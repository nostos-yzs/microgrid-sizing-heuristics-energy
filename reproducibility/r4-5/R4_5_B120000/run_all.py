"""One-command R4.5 campaign: preregister -> tune -> freeze target -> test -> report."""
from __future__ import annotations
import argparse, hashlib, importlib.util, itertools, json, os, platform, random, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from worker import BASE, ENTRIES, VARIANTS, save_json


def stamp(): return datetime.now(timezone.utc).isoformat()
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fingerprint(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def preflight():
    missing=[m for m in ['numpy','pandas','torch','scipy','matplotlib','openpyxl'] if importlib.util.find_spec(m) is None]
    if missing: raise RuntimeError('Missing packages: '+', '.join(missing)+'. Install requirements.txt with this Python interpreter.')
    expected=json.loads((BASE/'source_manifest.json').read_text(encoding='utf-8'))
    for rel,sha in expected.items():
        if digest(BASE/'source_snapshot'/rel)!=sha: raise RuntimeError('Source snapshot changed: '+rel)
    for i in range(5):
        hashes={digest(BASE/'source_snapshot'/v[0]/'data'/f'typicalday{i}.csv') for v in ENTRIES.values()}
        if len(hashes)!=1: raise RuntimeError('Typical-day inputs are not shared.')
    ems={digest(BASE/'source_snapshot'/v[0]/'microgrid_env.py') for v in ENTRIES.values()}
    if len(ems)!=1: raise RuntimeError('EMS implementations differ.')
    return expected


def execute(root, protocol, stage, alg, variant, candidate, seed, budget, target=None):
    name=f'{alg}_{variant}_{candidate}_seed{seed}'
    job=root/'runs'/stage/name;job.mkdir(parents=True,exist_ok=True)
    spec=dict(stage=stage,algorithm=alg,variant=variant,candidate=candidate,seed=seed,
              budget=budget,target=target,protocol=protocol)
    spec_hash=fingerprint(spec);result_path=job/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if result['spec_hash']!=spec_hash: raise RuntimeError('Resume parameters do not match '+str(job))
        print(f'[resume] {stage} {name}',flush=True);return result
    number=len(list(job.glob('attempt_*')))+1
    attempt=job/f'attempt_{number:03d}';attempt.mkdir()
    save_json(attempt/'spec.json',spec)
    env=os.environ.copy();env.update(PYTHONHASHSEED=str(seed),PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',
                                     OMP_NUM_THREADS=str(protocol['torch_threads']),MKL_NUM_THREADS=str(protocol['torch_threads']))
    print(f'[{stage}] {alg:3s} {variant:8s} seed={seed:4d} candidate={candidate:11s} B={budget}',flush=True)
    t=time.perf_counter();code=None
    try:
        with (attempt/'console.log').open('w',encoding='utf-8') as log:
            child=subprocess.run([sys.executable,str(BASE/'worker.py'),'--spec',str(attempt/'spec.json'),
                                  '--directory',str(attempt)],stdout=log,stderr=subprocess.STDOUT,env=env)
        code=child.returncode
    finally:
        elapsed=time.perf_counter()-t
        save_json(attempt/'attempt_timing.json',dict(process_wall_s=elapsed,exit_code=code,ended_utc=stamp(),
                 stage=stage,algorithm=alg,variant=variant,seed=seed,candidate=candidate))
    if code!=0:
        tail=(attempt/'console.log').read_text(encoding='utf-8',errors='replace')[-4500:]
        raise RuntimeError(f'Worker failed. Log: {attempt / "console.log"}\n{tail}')
    result=json.loads((attempt/'worker_result.json').read_text(encoding='utf-8'))
    result.update(process_wall_s=elapsed,spec_hash=spec_hash,artifact_dir=str(attempt.relative_to(root)))
    save_json(result_path,result)
    print(f'  feasible={result["feasible"]} LCOE={result["lcoe"]} EMS={result["ems_calls"]} '
          f'SAC_updates={result["sac_updates"]} wall={elapsed:.2f}s',flush=True)
    return result


def jobs(protocol, stage):
    seeds=protocol[{'tuning':'tuning_seeds','pilot':'pilot_seeds','formal':'seeds'}[stage]]
    candidates=protocol['candidates'] if stage=='tuning' else [None]
    rows=list(itertools.product(protocol['algorithms'],protocol['variants'],candidates,seeds))
    # Interleave methods to reduce confounding by execution order / machine drift.
    random.Random(protocol['schedule_seed']+{'tuning':1,'pilot':2,'formal':3}[stage]).shuffle(rows)
    return rows


def validate_pairs(rows, protocol):
    for alg in protocol['algorithms']:
        for seed in protocol['seeds']:
            group=[r for r in rows if r['algorithm']==alg and r['seed']==seed]
            if len(group)!=len(protocol['variants']): raise RuntimeError('Incomplete paired block.')
            if len({r['initialization_sha256'] for r in group})!=1:
                raise RuntimeError(f'Initialization differs within {alg}, seed {seed}.')
            if any(r['ems_calls']!=protocol['budget'] for r in group): raise RuntimeError('Unequal budgets.')
            if not protocol.get('validation_only') and any(r['variant']=='sac' and r['sac_updates']==0 for r in group):
                raise RuntimeError('A SAC formal run did not learn; inspect frozen batch size and budget.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=BASE/'results_B120000')
    ap.add_argument('--config',type=Path,default=BASE/'experiment_config.json')
    ap.add_argument('--check',action='store_true',help='Check environment/snapshot only; no optimization')
    ap.add_argument('--smoke',action='store_true',help='Tiny isolated validation campaign; never paper results')
    ap.add_argument('--analyze-only',action='store_true')
    ap.add_argument('--prepare-only',action='store_true',help='Complete tuning and independent pilot review, freeze target, then stop before formal seeds')
    args=ap.parse_args();expected=preflight()
    if args.check:
        print('Preflight OK. Python:',sys.executable,'; frozen files:',len(expected));return
    root=args.output.resolve()
    if args.smoke:
        if root==BASE/'results_B120000': root=BASE/'validation'/'smoke_B120000_v2'
    root.mkdir(parents=True,exist_ok=True)
    if args.analyze_only:
        from analysis import analyze
        analyze(root);return
    protocol=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.smoke:
        protocol.update(seeds=[0,42],budget=1200,tuning_seeds=[1001],tuning_budget=600,
                        pilot_seeds=[2001],pilot_budget=1200,sac_batch_size=2,checkpoint_every=100,
                        validation_only=True)
    sets=[set(protocol[k]) for k in ['seeds','tuning_seeds','pilot_seeds']]
    if any(a&b for a,b in itertools.combinations(sets,2)): raise ValueError('Training/pilot/test seeds overlap.')
    if any(len(protocol[k])!=len(set(protocol[k])) for k in ['seeds','tuning_seeds','pilot_seeds']):raise ValueError('Duplicate seeds.')
    if any(protocol[k]<1 for k in ['budget','pilot_budget','tuning_budget']) or protocol['target_relative_margin']<0:raise ValueError('Invalid budget/target margin.')
    if protocol['pilot_budget'] != protocol['budget']:raise ValueError('Pilot and formal budgets must match before freezing the shared target.')
    code_hashes={p.name:digest(p) for p in [BASE/name for name in ['run_all.py','worker.py','analysis.py','table3_reporting.py','budget_review.py','table3_reference.csv']]}
    identity=dict(protocol=protocol,source_hashes=expected,runner_hashes=code_hashes)
    campaign_hash=fingerprint(identity)
    frozen=root/'preregistration.json'
    if frozen.exists():
        saved=json.loads(frozen.read_text(encoding='utf-8'))
        if saved['campaign_hash']!=campaign_hash:
            raise RuntimeError('Frozen settings/code changed. Use a new --output folder; existing results are preserved.')
    else:
        save_json(frozen,dict(created_utc=stamp(),campaign_hash=campaign_hash,**identity,
            plan='EMS-call budget; best evaluated feasible incumbent; separate tuning and pilot seeds; target=(1+margin)*best pilot feasible LCOE; paired comparisons; Holm correction; unknown historical tuning excluded explicitly.'))
    started=time.perf_counter();session_start=stamp()
    tuned=root/'tuning_selection.json'
    if not tuned.exists():
        results=[]
        for alg,var,cand,seed in jobs(protocol,'tuning'):
            results.append(execute(root,protocol,'tuning',alg,var,cand,seed,protocol['tuning_budget']))
        selection={}
        for alg,var in itertools.product(protocol['algorithms'],protocol['variants']):
            ranked=[]
            for cand in protocol['candidates']:
                group=[r for r in results if (r['algorithm'],r['variant'],r['candidate'])==(alg,var,cand)]
                feasible=[r['lcoe'] for r in group if r['feasible']]
                score=(-len(feasible),sum(feasible)/len(feasible) if feasible else float('inf'),protocol['candidates'].index(cand))
                ranked.append((score,cand))
            selection[alg+'_'+var]=min(ranked)[1]
        save_json(tuned,dict(created_utc=stamp(),selection=selection,criterion='maximize feasible runs, then minimize conditional mean LCOE, then original candidate'))
    selection=json.loads(tuned.read_text(encoding='utf-8'))['selection']
    target_path=root/'target_frozen.json'
    if not target_path.exists():
        pilot=[]
        for alg,var,_,seed in jobs(protocol,'pilot'):
            pilot.append(execute(root,protocol,'pilot',alg,var,selection[alg+'_'+var],seed,protocol['pilot_budget']))
        from budget_review import review_pilot
        review_pilot(root,protocol,selection)
        feasible=[r['lcoe'] for r in pilot if r['feasible']]
        if not feasible:
            raise RuntimeError('No feasible pilot solution. Do not use formal results to pick a target. Revise preregistered pilot budget in a NEW output folder.')
        target=(1+protocol['target_relative_margin'])*min(feasible)
        save_json(target_path,dict(frozen_utc=stamp(),target=target,best_pilot_feasible=min(feasible),
                  relative_margin=protocol['target_relative_margin'],pilot_seeds=protocol['pilot_seeds']))
    if not (root/'pilot_budget_review.json').exists():
        from budget_review import review_pilot
        review_pilot(root,protocol,selection)
    target=json.loads(target_path.read_text(encoding='utf-8'))['target']
    print(f'Frozen common target = {target:.10g}; formal runs = {len(jobs(protocol,"formal"))}',flush=True)
    if args.prepare_only:
        print('Preparation complete. Inspect pilot_budget_review.csv/json; rerun without --prepare-only to continue with the frozen protocol.',flush=True)
        return
    formal=[]
    for alg,var,_,seed in jobs(protocol,'formal'):
        formal.append(execute(root,protocol,'formal',alg,var,selection[alg+'_'+var],seed,protocol['budget'],target))
    validate_pairs(formal,protocol)
    save_json(root/'formal_complete.json',dict(completed_utc=stamp(),n=len(formal),campaign_hash=campaign_hash,
                                             initialization_and_budget_checks='passed'))
    from analysis import analyze
    report_start=time.perf_counter();analyze(root)
    save_json(root/'last_session.json',dict(started_utc=session_start,completed_utc=stamp(),
              active_session_wall_s=time.perf_counter()-started,analysis_export_s=time.perf_counter()-report_start,
              note='Session wall includes resumed work only; all attempt costs across sessions are in the workbook.'))
    print('\nCompleted:',root/'R4_5_statistics.xlsx',flush=True)


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt:
        print('Interrupted. Completed runs are retained; rerun the same command to resume.',file=sys.stderr);sys.exit(130)
    except Exception as exc:
        print('ERROR:',exc,file=sys.stderr);sys.exit(1)
