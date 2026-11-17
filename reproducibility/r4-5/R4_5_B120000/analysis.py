"""Auditable statistical analysis and publication plots for completed R4.5 runs."""
from __future__ import annotations
import itertools, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from worker import save_json
from table3_reporting import expand_equipment, design_tables, EQUIPMENT

ALGS=['GA','PSO','SA','TS']; VARS=['baseline','static','sac']
LABELS={'baseline':'Baseline','static':'Static repair','sac':'SAC repair'}
COLORS={'baseline':'#777777','static':'#397EAB','sac':'#CE8541'}


def mean_ci(values, confidence=.95):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)];n=len(x)
    if not n:return (0,np.nan,np.nan,np.nan,np.nan)
    mean=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else np.nan
    half=float(stats.t.ppf((1+confidence)/2,n-1)*sd/math.sqrt(n)) if n>1 else np.nan
    return n,mean,sd,mean-half,mean+half


def wilson(successes,n,confidence=.95):
    if not n:return np.nan,np.nan
    z=stats.norm.ppf((1+confidence)/2);q=successes/n;den=1+z*z/n
    center=(q+z*z/(2*n))/den;half=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
    return float(center-half),float(center+half)


def holm(values):
    p=np.asarray(values,dtype=float);result=np.full(len(p),np.nan);idx=np.where(np.isfinite(p))[0]
    order=idx[np.argsort(p[idx])];running=0.
    for i,j in enumerate(order):
        running=max(running,(len(order)-i)*p[j]);result[j]=min(1.,running)
    return result


def wilcoxon_p(differences):
    d=np.round(np.asarray(differences,dtype=float),12)
    if len(d)<2:return np.nan
    if np.all(d==0):return 1.
    # Zero differences discarded; average ranks for ties; two-sided normal approximation.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(stats.wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='approx').pvalue)


def summaries(raw, confidence):
    result=[]
    for (alg,var),g in raw.groupby(['algorithm','variant'],sort=False):
        n=len(g);nf=int(g.feasible.sum());nh=int(g.target_reached.sum())
        row=dict(algorithm=alg,variant=var,n_runs=n,n_feasible=nf,feasibility_rate=nf/n,
                 n_target_hits=nh,target_hit_rate=nh/n,budget=int(g.budget.iloc[0]))
        row['feasibility_ci_low'],row['feasibility_ci_high']=wilson(nf,n,confidence)
        row['target_hit_ci_low'],row['target_hit_ci_high']=wilson(nh,n,confidence)
        for label,vals in [('lcoe',g.loc[g.feasible,'lcoe']),('formal_process_s',g.process_wall_s),
                           ('search_s',g.search_wall_s),('sac_s',g.sac_total_s),
                           ('capped_target_calls',g.target_observation_calls),
                           ('hit_target_calls',g.loc[g.target_reached,'target_first_calls']),
                           ('hit_target_search_s',g.loc[g.target_reached,'target_first_search_s']),
                           ('amortized_total_s',g.amortized_total_s)]:
            nn,m,sd,lo,hi=mean_ci(vals,confidence)
            row.update({label+'_n':nn,label+'_mean':m,label+'_sd':sd,label+'_ci_low':lo,label+'_ci_high':hi})
        row['target_calls_median_among_hits']=g.loc[g.target_reached,'target_first_calls'].median()
        row['sac_updates_min']=g.sac_updates.min();row['sac_updates_mean']=g.sac_updates.mean()
        result.append(row)
    return pd.DataFrame(result)


def comparisons(raw, confidence):
    groups={(a,v):g.set_index('seed') for (a,v),g in raw.groupby(['algorithm','variant'])}
    rows=[]
    for a,b in itertools.combinations(sorted(groups),2):
        x,y=groups[a],groups[b];seeds=x.index.intersection(y.index);x=x.loc[seeds];y=y.loc[seeds]
        base=dict(method_A='_'.join(a),method_B='_'.join(b),within_algorithm=a[0]==b[0],n_seed_pairs=len(seeds))
        for metric in ['lcoe','process_wall_s','target_observation_calls','feasible']:
            if metric=='lcoe':
                keep=x.feasible & y.feasible
                difference=x.loc[keep,'lcoe'].to_numpy()-y.loc[keep,'lcoe'].to_numpy()
            else:difference=x[metric].to_numpy(dtype=float)-y[metric].to_numpy(dtype=float)
            n,mean,sd,low,high=mean_ci(difference,confidence)
            if metric=='feasible':
                only_a=int(((x.feasible)&(~y.feasible)).sum());only_b=int(((~x.feasible)&y.feasible).sum())
                p=float(stats.binomtest(only_a,only_a+only_b,.5).pvalue) if only_a+only_b else 1.
                test='Exact McNemar (discordant-pair binomial)'
            else:p=wilcoxon_p(difference);test='Wilcoxon signed-rank, two-sided, normal approximation'
            rows.append(dict(**base,metric=metric,n_used=n,mean_A_minus_B=mean,sd_difference=sd,
                             ci_low=low,ci_high=high,p_raw=p,test=test,
                             conditioning='both runs feasible' if metric=='lcoe' else 'all paired seeds'))
    df=pd.DataFrame(rows)
    for metric,idx in df.groupby('metric').groups.items():
        df.loc[idx,'p_holm']=holm(df.loc[idx,'p_raw'])
        df.loc[idx,'correction_family']=f'all {len(idx)} method pairs for {metric}'
    df['significant_0_05']=df.p_holm<.05
    return df


def collect(root):
    rows=[]
    for p in sorted((root/'runs').glob('*/*/result.json')):
        row=json.loads(p.read_text(encoding='utf-8'));rows.append(row)
    allruns=pd.DataFrame(rows)
    attempts=[]
    for p in (root/'runs').glob('*/*/attempt_*/attempt_timing.json'):
        r=json.loads(p.read_text(encoding='utf-8'));r['path']=str(p.relative_to(root));attempts.append(r)
    return allruns,pd.DataFrame(attempts)


def timing_table(allruns,attempts,n_seeds):
    rows=[]
    for alg,var in itertools.product(ALGS,VARS):
        g=allruns[(allruns.algorithm==alg)&(allruns.variant==var)]
        a=attempts[(attempts.algorithm==alg)&(attempts.variant==var)]
        stages={stage:float(a.loc[a.stage==stage,'process_wall_s'].sum()) for stage in ['tuning','pilot','formal']}
        successful=float(g.process_wall_s.sum());total=sum(stages.values())
        row=dict(algorithm=alg,variant=var,tuning_total_s=stages['tuning'],pilot_total_s=stages['pilot'],
                 formal_total_s=stages['formal'],campaign_method_total_s=total,
                 preparation_amortized_per_seed_s=(stages['tuning']+stages['pilot'])/n_seeds,
                 retry_or_failed_attempt_s=max(0.,total-successful),
                 campaign_total_per_seed_s=total/n_seeds)
        for stage in ['tuning','pilot','formal']:
            z=g[g.stage==stage]
            for field in ['sac_init_s','sac_action_s','sac_replay_s','sac_learn_s','sac_total_s','search_wall_s','ems_s']:
                row[stage+'_'+field]=float(z[field].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_table(root,raw,budget,confidence):
    grid=np.unique(np.r_[1,np.linspace(0,budget,151).astype(int)[1:],budget])
    rows=[];matrices={}
    for alg,var in itertools.product(ALGS,VARS):
        g=raw[(raw.algorithm==alg)&(raw.variant==var)];values=[]
        for _,r in g.iterrows():
            imp=pd.read_csv(root/r.artifact_dir/'improvements.csv')
            vals=np.full(len(grid),np.nan)
            if len(imp):
                idx=np.searchsorted(imp.ems_calls.to_numpy(),grid,side='right')-1
                keep=idx>=0;vals[keep]=imp.best_feasible_lcoe.to_numpy()[idx[keep]]
            values.append(vals)
        matrix=np.asarray(values);matrices[alg,var]=(grid,matrix)
        for i,call in enumerate(grid):
            n,m,sd,lo,hi=mean_ci(matrix[:,i],confidence)
            rows.append(dict(algorithm=alg,variant=var,ems_calls=int(call),n_runs=len(g),n_feasible=n,
                     feasibility_rate=n/len(g),conditional_lcoe_mean=m,ci_low=lo,ci_high=hi))
    return pd.DataFrame(rows),matrices


def figures(root,raw,summary,pairs,timing,curves,confidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7})
    out=root/'figures';out.mkdir(exist_ok=True);rng=np.random.default_rng(9001)
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(12.6,6.3),layout='constrained')
    for col,alg in enumerate(ALGS):
        for i,var in enumerate(VARS):
            g=raw[(raw.algorithm==alg)&(raw.variant==var)];s=summary[(summary.algorithm==alg)&(summary.variant==var)].iloc[0]
            vals=g.loc[g.feasible,'lcoe'].to_numpy()
            axes[0,col].scatter(i+rng.uniform(-.12,.12,len(vals)),vals,s=17,color=COLORS[var],alpha=.7)
            if len(vals)>1:
                axes[0,col].errorbar(i,s.lcoe_mean,yerr=[[s.lcoe_mean-s.lcoe_ci_low],[s.lcoe_ci_high-s.lcoe_mean]],
                                     color='black',marker='_',capsize=4,linewidth=1.3)
            elif len(vals)==1: axes[0,col].plot(i,vals[0],'k_')
            axes[0,col].text(i,.98,f'n={len(vals)}/{len(g)}',transform=axes[0,col].get_xaxis_transform(),ha='center',va='top',fontsize=8)
            axes[1,col].errorbar(i,s.feasibility_rate,yerr=[[s.feasibility_rate-s.feasibility_ci_low],
                        [s.feasibility_ci_high-s.feasibility_rate]],fmt='o',color=COLORS[var],capsize=4)
        axes[0,col].set_title(alg);axes[1,col].set_ylim(-.05,1.10)
        for ax in axes[:,col]:ax.set_xticks(range(3),['Base','Static','SAC']);ax.set_xlim(-.45,2.45)
    axes[0,0].set_ylabel('Feasible LCOE (mean and 95% CI)');axes[1,0].set_ylabel('Feasibility rate (95% Wilson CI)')
    fig.suptitle('Equal EMS budgets; independent seeds; infeasible runs excluded only from LCOE')
    save(fig,'01_outcomes')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,alg in zip(axes.flat,ALGS):
        for var in VARS:
            g=curves[(curves.algorithm==alg)&(curves.variant==var)]
            ax.plot(g.ems_calls,g.conditional_lcoe_mean,color=COLORS[var],label=LABELS[var])
            ax.fill_between(g.ems_calls,g.ci_low,g.ci_high,color=COLORS[var],alpha=.15)
        ax.set_title(alg);ax.set_xlabel('EMS evaluations');ax.set_ylabel('Best feasible LCOE')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Conditional mean and pointwise 95% CI; available feasible counts in source data')
    save(fig,'02_convergence')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained');budget=int(raw.budget.iloc[0])
    for ax,alg in zip(axes.flat,ALGS):
        for var in VARS:
            g=raw[(raw.algorithm==alg)&(raw.variant==var)]
            hits=np.sort(g.loc[g.target_reached,'target_first_calls'].to_numpy())
            xx=np.r_[0,hits,budget];yy=np.r_[0,np.arange(1,len(hits)+1)/len(g),len(hits)/len(g)]
            ax.step(xx,yy,where='post',color=COLORS[var],label=LABELS[var])
        ax.set_title(alg);ax.set_xlabel('EMS evaluations');ax.set_ylabel('Fraction reaching common target');ax.set_ylim(0,1.04)
    axes.flat[0].legend(fontsize=8);fig.suptitle(f'Common target = {raw.target.iloc[0]:.6g}; non-hits censored at B = {budget:,}')
    save(fig,'03_target_attainment')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    cost_colors=['#777777','#397EAB','#CE8541','#A5B790','#B9A1C4']
    for ax,alg in zip(axes.flat,ALGS):
        for i,var in enumerate(VARS):
            g=raw[(raw.algorithm==alg)&(raw.variant==var)];t=timing[(timing.algorithm==alg)&(timing.variant==var)].iloc[0]
            vals=[g.search_non_sac_s.mean(),g.sac_total_s.mean(),
                  (g.process_wall_s-g.search_wall_s).mean(),t.tuning_total_s/len(g),t.pilot_total_s/len(g)]
            bottom=0
            for val,color,label in zip(vals,cost_colors,['Search excluding SAC','SAC (nested in search)','Setup / process / export','Tuning / seed','Pilot / seed']):
                ax.bar(i,val,bottom=bottom,color=color,label=label if i==0 else None,width=.65);bottom+=val
        ax.set_title(alg);ax.set_xticks(range(3),['Base','Static','SAC']);ax.set_ylabel('Mean seconds / formal seed')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Complete measured cost; SAC is included once (failed attempts listed separately)')
    save(fig,'04_timing')
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    for ax,alg in zip(axes.flat,ALGS):
        g=pairs[(pairs.metric=='lcoe')&pairs.within_algorithm&pairs.method_A.str.startswith(alg+'_')]
        labels=[]
        for i,(_,r) in enumerate(g.iterrows()):
            labels.append(r.method_A.split('_',1)[1]+' - '+r.method_B.split('_',1)[1]+f' (n={r.n_used})')
            if np.isfinite(r.ci_low):
                ax.errorbar(r.mean_A_minus_B,i,xerr=[[r.mean_A_minus_B-r.ci_low],[r.ci_high-r.mean_A_minus_B]],fmt='o',capsize=4,color='#397EAB')
            elif np.isfinite(r.mean_A_minus_B):ax.plot(r.mean_A_minus_B,i,'o',color='#397EAB')
        ax.axvline(0,color='gray',ls='--',lw=.8);ax.set_yticks(range(len(labels)),labels,fontsize=8)
        ax.set_title(alg);ax.set_xlabel('Paired mean LCOE difference (95% CI)')
    fig.suptitle('Jointly feasible seed pairs; Holm-adjusted Wilcoxon tests in workbook')
    save(fig,'05_paired_effects')


def workbook(root,frames):
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table,TableStyleInfo
    path=root/'R4_5_statistics.xlsx'
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        for name,df in frames.items():
            export=df.copy()
            for col in export.columns:
                export[col]=export[col].map(lambda v:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v)
            export.to_excel(writer,sheet_name=name,index=False)
        wb=writer.book
        for ws in wb.worksheets:
            ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
            for c in ws[1]:c.fill=PatternFill('solid',fgColor='203864');c.font=Font(color='FFFFFF',bold=True);c.alignment=Alignment(wrap_text=True,vertical='center')
            ws.row_dimensions[1].height=45
            for i,col in enumerate(ws.columns,1):
                width=min(44,max(13,max(len(str(c.value or '')) for c in list(col)[:40])+2))
                ws.column_dimensions[get_column_letter(i)].width=width
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value,float):c.number_format='0.000000'
            if ws.max_row>1:
                table=Table(displayName='T_'+ws.title,ref=ws.dimensions)
                table.tableStyleInfo=TableStyleInfo(name='TableStyleMedium2',showRowStripes=True);ws.add_table(table)
        # Actual formulas make the basic consistency checks inspectable in Excel.
        checks=wb['Checks'];raw=wb['Raw_runs'];headers={c.value:c.column_letter for c in raw[1]};last=raw.max_row
        checks.append(['raw_row_count_formula',f'=COUNTA(Raw_runs!A2:A{last})',f'Expected {len(frames["Raw_runs"])}'])
        checks.append(['total_EMS_calls_formula',f'=SUM(Raw_runs!{headers["ems_calls"]}2:{headers["ems_calls"]}{last})',f'Expected {int(frames["Raw_runs"].budget.sum()):,}'])
        checks.append(['budget_mismatch_formula',f'=SUMPRODUCT(--(Raw_runs!{headers["budget"]}2:{headers["budget"]}{last}<>Raw_runs!{headers["ems_calls"]}2:{headers["ems_calls"]}{last}))','Expected 0'])
    return path


def analyze(root):
    root=Path(root)
    if not (root/'formal_complete.json').exists():raise RuntimeError('Formal campaign is incomplete; no final inferential report is generated.')
    protocol=json.loads((root/'preregistration.json').read_text(encoding='utf-8'))['protocol']
    allruns,attempts=collect(root);raw=allruns[allruns.stage=='formal'].copy()
    expected=len(protocol['seeds'])*len(protocol['algorithms'])*len(protocol['variants'])
    if len(raw)!=expected or raw.duplicated(['algorithm','variant','seed']).any():raise RuntimeError('Missing/duplicate formal results.')
    confidence=protocol['confidence'];timing=timing_table(allruns,attempts,len(protocol['seeds']))
    raw=raw.merge(timing[['algorithm','variant','preparation_amortized_per_seed_s']],on=['algorithm','variant'],validate='many_to_one')
    raw['amortized_total_s']=raw.process_wall_s+raw.preparation_amortized_per_seed_s
    raw=raw.sort_values(['algorithm','variant','seed']).reset_index(drop=True)
    raw=expand_equipment(raw)
    design_frames=design_tables(raw)
    representatives=design_frames['Representative_designs'].set_index(['algorithm','variant'])
    summary=summaries(raw,confidence);pairs=comparisons(raw,confidence)
    curves,_=convergence_table(root,raw,protocol['budget'],confidence)
    fmt=lambda m,l,h: 'NA' if not np.isfinite(m) else f'{m:.6f} [{l:.6f}, {h:.6f}]'
    table=[]
    for _,r in summary.iterrows():
        design=representatives.loc[(r.algorithm,r.variant)]
        table.append({'Algorithm':r.algorithm,'Variant':LABELS[r.variant],
         'LCOE mean [95% CI]':fmt(r.lcoe_mean,r.lcoe_ci_low,r.lcoe_ci_high),
         'Representative design seed':design['seed'],
         'Representative design LCOE':design['lcoe'],
         **{key+' (representative)':design[key] for key in EQUIPMENT},
         'Feasible / all':f'{r.n_feasible}/{r.n_runs}',
         'Feasibility % [95% CI]':f'{100*r.feasibility_rate:.1f} [{100*r.feasibility_ci_low:.1f}, {100*r.feasibility_ci_high:.1f}]',
         'Target hits / all':f'{r.n_target_hits}/{r.n_runs}',
         'Eval-to-target among hits mean [95% CI]':fmt(r.hit_target_calls_mean,r.hit_target_calls_ci_low,r.hit_target_calls_ci_high),
         'Time-to-target search s among hits mean [95% CI]':fmt(r.hit_target_search_s_mean,r.hit_target_search_s_ci_low,r.hit_target_search_s_ci_high),
         'Restricted target calls mean [95% CI]':fmt(r.capped_target_calls_mean,r.capped_target_calls_ci_low,r.capped_target_calls_ci_high),
         'Formal wall s mean [95% CI]':fmt(r.formal_process_s_mean,r.formal_process_s_ci_low,r.formal_process_s_ci_high),
         'With amortized preparation s [95% CI]':fmt(r.amortized_total_s_mean,r.amortized_total_s_ci_low,r.amortized_total_s_ci_high)})
    notes=[('Experiment',json.dumps(protocol)),
     ('Design columns','Six integer equipment counts of an actual lower-median feasible LCOE run, ties by seed. Its seed and LCOE are shown separately from the across-seed mean. Infeasible designs stay blank.'),
     ('Historical Table 3','Printed values transcribed from original manuscript PDF page 21. Seed42_vs_original is descriptive only; it is not proof of historical reproduction. Historical target/time definitions differ.'),
     ('Target time among hits','Actual elapsed search wall seconds at first target evaluation, including online SAC work; conditional on hitting the target. Read with target hit rate and censored calls; never replace non-hits with zero.'),
     ('Budget review','Independent pilot diagnostics are in Pilot_budget_review. Exact budgets and nonzero SAC updates are execution checks; late improvements do not certify convergence.'),
     ('Sample unit','One independent seed. Same algorithm/seed shares initialization across its three variants.'),
     ('LCOE endpoint','Best feasible design evaluated anywhere within B EMS calls; includes repair candidates. Success status AND finite objective < 1e6.'),
     ('LCOE uncertainty','Two-sided Student-t confidence interval of the conditional mean among feasible runs; not standard deviation.'),
     ('Pairwise effect CI','Student-t interval for same-seed mean differences; LCOE uses jointly feasible pairs only; selection may bias conditional comparisons.'),
     ('Pairwise tests','Two-sided Wilcoxon signed-rank with normal approximation, differences rounded to 12 decimals, zero differences removed; all-zero pairs p=1. Interpretation assumes symmetric difference distribution.'),
     ('Multiplicity','Holm adjustment over all 66 method-pair tests separately for each metric; effect CIs are unadjusted.'),
     ('Feasibility','Wilson 95% confidence interval; exact McNemar/binomial test for paired feasibility.'),
     ('Common target','1.05 times the best feasible LCOE of independent all-method pilot runs; frozen before formal data.'),
     ('Censoring','Non-hits retain blank target_first_calls; observation_calls=B and censored=True. Restricted mean min(T,B) is not the uncensored expected time-to-target.'),
     ('Timing','process_wall_s includes imports, setup, agent initialization, search, logging and export; SAC sub-times are nested, not added again. Tuning/pilot overhead is allocated equally over 22 formal seeds for each method.'),
     ('Historical tuning','Previous manual tuning outside this campaign is UNKNOWN, never reported as zero. Complete cost covers this reproducible campaign only.'),
     ('Algorithm freeze','Original local operators/action mappings/rewards retained. SA legacy main performs additional action mapping; this protocol does not silently change it.'),
     ('Budget adaptations','Exact EMS-call stop; SAC progress state uses EMS/B; SA original 60-layer endpoints stretched across budget; all SAC batch sizes=32 (smoke=2).'),
     ('Training','SAC starts from scratch each run. Actual update counts are exported; no pretrained model is reused.'),
     ('Figures','Quantitative comparison without a preselected winner. Pointwise CI bands, conditional feasible counts exported. PDF/SVG vector text; PNG 300 dpi.'),
     ('Sources','Statistical definitions follow scipy.stats.t, scipy.stats.wilcoxon, scipy.stats.binomtest; Wilson interval and Holm step-down implemented explicitly.'),
     ('Validation only',str(protocol.get('validation_only',False)))]
    checks=pd.DataFrame([('formal_runs',len(raw),expected),('budget_mismatches',int((raw.ems_calls!=raw.budget).sum()),0),
       ('paired_initialization_mismatches',int((raw.groupby(['algorithm','seed']).initialization_sha256.nunique()!=1).sum()),0),
       ('target_nonhit_values_present',int(raw.loc[~raw.target_reached,'target_first_calls'].notna().sum()),0),
       ('sac_runs_without_updates',int(((raw.variant=='sac')&(raw.sac_updates==0)).sum()),0)],columns=['check','observed','expected'])
    frames={'Table3':pd.DataFrame(table),**design_frames,'Raw_runs':raw,'Statistics':summary,'Paired_tests':pairs,'Timing':timing,
            'Tuning_pilot':allruns[allruns.stage!='formal'],'Convergence':curves,'Attempts':attempts,
            'Protocol':pd.DataFrame(notes,columns=['item','definition']),'Checks':checks}
    review_path=root/'pilot_budget_review.csv'
    if review_path.exists():frames['Pilot_budget_review']=pd.read_csv(review_path)
    for name,df in frames.items():df.to_csv(root/(name+'.csv'),index=False,encoding='utf-8-sig')
    figures(root,raw,summary,pairs,timing,curves,confidence);workbook(root,frames)
    save_json(root/'analysis_checks.json',dict(n_formal=len(raw),expected=expected,n_pairs=len(pairs),
                all_checks_pass=bool((checks.observed==checks.expected).all()),
                measured_attempt_wall_s=float(attempts.process_wall_s.sum()),
                note='Do not claim superiority from a mean alone; inspect feasibility, paired tests, and complete timing.'))
    (root/'FIGURE_CAPTIONS.txt').write_text('\n'.join([
       'Fig. 1 | Final outcomes under equal EMS-call budgets. Individual feasible runs, mean and two-sided 95% t CI; Wilson intervals for feasibility. Infeasible runs remain in feasibility denominators.',
       'Fig. 2 | Best evaluated feasible LCOE versus EMS calls. Conditional mean and pointwise 95% t CI; number of feasible seeds at each checkpoint is in Convergence.csv.',
       'Fig. 3 | Empirical cumulative target attainment. All seeds in denominator; non-hits administratively censored at the common budget.',
       'Fig. 4 | Mean complete measured computing cost. SAC subcomponents are included once. Preparation is amortized across formal seeds; failed/retried attempts are separately reported in Timing.',
       'Fig. 5 | Within-algorithm paired LCOE differences. Jointly feasible seed pairs only; unadjusted 95% mean-difference CIs. Holm-adjusted signed-rank p values are in Paired_tests.']),encoding='utf-8')
    print('Exported workbook, CSV source data, and 5 PDF/SVG/PNG figure sets.',flush=True)
