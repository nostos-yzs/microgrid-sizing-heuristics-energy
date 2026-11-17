from pathlib import Path
import sys, json,random, collections, time,platform, argparse,hashlib
import numpy as np,pandas as pd, torch
from environment import FastEnv,module,BASE
ENTRIES={'GA':('ga_repair','ga_core','config_ga'),'PSO':('pso_repair','pso_core_sac','config_sac'),'SA':('sa_repair','sa_core','config_sa'),'TS':('ts_repair','ts_core','config_tabu')}

def load(alg):
 folder,core,cfg=ENTRIES[alg];d=BASE/'patched'/folder
 sys.path.insert(0,str(d))
 try:c=module(alg+'_config',d/(cfg+'.py')).get_config();m=module(alg+'_core',d/(core+'.py'));agent=module(alg+'_agent',d/'sac_agent.py')
 finally:sys.path.pop(0)
 if alg=='SA':c['sac_agent_params']=c['experiments']['sac_guided']['sac_agent_params']
 c['sac_agent_params']['device']='cpu'
 helper=module(alg+'_helpers',BASE/'code'/('legacy_'+alg.lower()+'_helpers.py'))
 return c,m,agent,helper

def reward_parts(old,new,diag,first,alg,arm,c):
 p=c['sac_agent_params'];imp=(old-new)/old if new<old and 0<old<1e6 else 0.
 progress=float(p['rewards']['gbest_improvement']*np.tanh(10*imp)) if new<old else float(p['rewards']['stagnation'])
 bonus=50. if first else 0.;pen=float(p['penalties'].get((diag or {}).get('status','success'),0));override=0.
 if arm!='revised' and first and alg in ['SA','TS']:
  override=-progress
  if alg=='TS':override-=pen
 return dict(r_progress=progress,r_first=bonus,r_failure=pen,r_override=override,reward=progress+bonus+pen+override,relative_improvement=imp)

def state_fn(alg,s,i,n,h,helper,pop=None,diag=None):
 if alg=='GA':return helper.get_state_ga(pop,i,n,list(h),diag)
 if alg=='PSO':return helper.get_state(s,i,n,h)
 if alg=='SA':return helper.get_state_sa(s,n)
 return helper.get_state_ts(s,i,n,h)

def run(alg,arm,seed,smoke=False):
 np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 c,core,agentmod,helper=load(alg);c['sac_agent_params']['corrected_policy']=(arm=='revised')
 # Source search budgets preserved. Only smoke overrides them for test.
 n={'GA':250,'PSO':200,'SA':3600,'TS':200}[alg]
 if smoke:n=12
 folder=ENTRIES[alg][0];env=FastEnv(c,folder);agent=agentmod.SACAgent(c);h=collections.deque(maxlen=10 if alg=='PSO' else 20);rows=[];first_seen=False;diag=None;pop=None
 if alg=='GA':pop,lo,hi=core.initialize_population(c);best=np.inf;sol=None;s=None
 elif alg=='PSO':
  s=core.initialize_swarm(c);best=s['gbest_fitness'];sol=None
  if arm in ['revised','transition_only']:core.evaluate_moved_swarm(s,env);best=s['gbest_fitness'];first_seen=best<1e6
 elif alg=='SA':
  # Both members of each pair use the explicit run seed for local proposal RNG.
  s=core.initialize_sa_state(env,c['min_bounds'],c['max_bounds'],c['sa_params']['schedule'],seed,c['sac_agent_params']);best=s['f_best'];first_seen=(s['diag_best'].get('status')=='success');sol=s['x_best']
 else:s=core.initialize_tabu_state(c,env,seed=seed);best=s['best_f'];first_seen=best<1e6;sol=s['best_x']
 obs=state_fn(alg,s,0,n,h,helper,pop,diag);t0=time.perf_counter()
 for t in range(n):
  mapped=agent.choose_action(obs);meta=agent.last_action;applied=mapped.copy()
  if alg=='SA' and arm=='legacy':applied=helper._map_to_bounds(mapped,c['sac_agent_params']['action_bounds'])
  old=best;step_calls=env.calls
  if alg=='GA':
   ev=core.evaluate_population(pop,env);b=min(ev,key=lambda x:x['fitness'])
   if b['fitness']<best and b['fitness']<1e6 and b['diagnostics']['status']=='success':best=b['fitness'];sol=b['individual'].copy();diag=b['diagnostics']
   pop=core.run_ga_iteration_guided(pop,ev,applied,lo,hi,env);ev=core.evaluate_population(pop,env);b=min(ev,key=lambda x:x['fitness'])
   if b['fitness']<best and b['fitness']<1e6 and b['diagnostics']['status']=='success':best=b['fitness'];sol=b['individual'].copy();diag=b['diagnostics']
   current=b['fitness'];currentdiag=b['diagnostics'];rdiag=currentdiag;proposal=current;propdiag=currentdiag
   first=(not first_seen) and b['fitness']<1e6 and currentdiag['status']=='success'
  elif alg=='PSO':
   s,info=core.run_pso_iteration_guided(s,applied,env)
   if arm in ['revised','transition_only']:current,currentdiag=core.evaluate_moved_swarm(s,env)
   else:current=s['gbest_fitness'];currentdiag=s['gbest_diagnostics']
   best=s['gbest_fitness'];sol=s['gbest_position'];diag=s['gbest_diagnostics'];rdiag=currentdiag if arm=='revised' else diag;proposal=current;propdiag=currentdiag;first=not first_seen and best<1e6
  elif alg=='SA':
   sch=c['sa_params']['schedule'];s['T']=max(sch['T0']*sch['alpha']**(t//60),sch['Tmin']);s=core.run_sa_iteration(s,applied,env)
   best=s['f_best'];sol=s['x_best'];diag=s['diag_best'];current=s['f'];currentdiag=s['diag'];rdiag=currentdiag if arm=='revised' else diag;proposal=s['proposal_f'];propdiag=s['proposal_diag'];first=not first_seen and diag['status']=='success'
  else:
   s=core.run_tabu_iteration_guided(s,applied,env);best=s['best_f'];sol=s['best_x'];diag=s['best_diag'];current=s['f'];currentdiag=s['diag'];rdiag=currentdiag if arm=='revised' else diag;proposal=current;propdiag=currentdiag;first=not first_seen and best<1e6
  if first:first_seen=True
  parts=reward_parts(old,best,rdiag,first,alg,arm,c)
  if arm!='revised':
   fn=helper.calculate_reward_sa if alg=='SA' else helper.calculate_reward;cfg=c['sac_agent_params'] if alg=='SA' else c
   assert np.isclose(parts['reward'],fn(old,best,rdiag,first,cfg)),(alg,parts)
  nxt=state_fn(alg,s,t+1,n,h,helper,pop,diag)
  agent.replay_buffer.add(obs,applied,parts['reward'],nxt,t==n-1);agent.learn()
  pidx=1 if alg=='SA' else 0;kidx=1-pidx
  row=dict(algorithm=alg,arm=arm,seed=seed,step=t+1,old_best=old,best_objective=best,lcoe=best if best<1e6 else np.nan,current_objective=current,current_status=(currentdiag or {}).get('status'),proposal_objective=proposal,proposal_status=(propdiag or {}).get('status'),reward_status=(rdiag or {}).get('status'),best_status=(diag or {}).get('status'),p_mapped=float(mapped[pidx]),k_mapped=float(mapped[kidx]),p_applied=float(applied[pidx]),k_applied=float(applied[kidx]),evaluation_requests=env.calls,unique_evaluations=env.unique,step_evaluations=env.calls-step_calls,learn_updates=agent.learn_updates,first_feasible=first,**parts)
  if alg=='TS':row['guided_quota']=int(np.floor(applied[0]*c['ts_params']['neighbors']));row['effective_guided_fraction']=row['guided_quota']/c['ts_params']['neighbors']
  for j in range(2):row.update({f'z{j}':float(meta['z'][j]),f'u{j}':float(meta['u'][j]),f'mapped{j}':float(mapped[j]),f'applied{j}':float(applied[j])})
  for j in range(4):row[f's{j}']=float(obs[j]);row[f'next_s{j}']=float(nxt[j])
  rows.append(row);obs=nxt;h.append(best)
 elapsed=time.perf_counter()-t0
 # Reevaluate the reported final design with the original Python EMS, not just the accelerator.
 if sol is not None:
  rf,rd=env.ref.evaluate_solution(sol);assert np.isclose(rf,best,rtol=1e-9,atol=1e-7),(alg,arm,seed,rf,best);assert rd['status']==(diag or s.get('gbest_diagnostics'))['status']
 else:rf=np.inf;rd={'status':'none'}
 summary=dict(algorithm=alg,arm=arm,seed=seed,steps=n,final_objective=best,lcoe=best if best<1e6 else None,feasible=best<1e6,final_original_python=rf,final_status=rd['status'],solution=np.asarray(sol).tolist() if sol is not None else None,evaluation_requests=env.calls,unique_evaluations=env.unique,learn_updates=agent.learn_updates,seconds=elapsed)
 out=BASE/'results'/('smoke' if smoke else 'runs');out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/f'{alg}_{arm}_{seed}.csv',index=False);(out/f'{alg}_{arm}_{seed}.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 print(json.dumps(summary),flush=True);return summary

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--algorithms',nargs='+',default=list(ENTRIES));ap.add_argument('--arms',nargs='+',default=['legacy','revised']);ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46]);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 for alg in a.algorithms:
  for seed in a.seeds:
   for arm in a.arms:run(alg,arm,seed,a.smoke)
if __name__=='__main__':main()
