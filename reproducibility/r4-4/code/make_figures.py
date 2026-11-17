from pathlib import Path
import json, numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
BASE=Path(__file__).resolve().parents[1];F=BASE/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':400})
C={'legacy':'#737373','revised':'#0072B2','mapping_only':'#D55E00','transition_only':'#009E73'}
def save(fig,name):
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.svg'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=400,bbox_inches='tight');plt.close(fig)
def panel(ax,label):ax.text(-.13,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=10)

def main():
 summaries=[json.loads(p.read_text('utf8')) for p in (BASE/'results/runs').glob('*.json')];S=pd.DataFrame(summaries)
 assert len(S)==50,('need all 50 runs',len(S))
 S.to_csv(BASE/'results/run_summary.csv',index=False)
 agg=S.groupby(['algorithm','arm']).agg(n=('seed','size'),feasible=('feasible','sum'),mean=('lcoe','mean'),sd=('lcoe','std'),minimum=('lcoe','min'),maximum=('lcoe','max'),mean_evaluations=('evaluation_requests','mean'),mean_unique=('unique_evaluations','mean'),learn_updates=('learn_updates','first')).reset_index();agg.to_csv(BASE/'results/summary_statistics.csv',index=False)
 def log(a,arm='revised',seed=42):return pd.read_csv(BASE/'results/runs'/f'{a}_{arm}_{seed}.csv')
 # Invariants: no empirical performance interpretation in hand check.
 fig,axs=plt.subplots(2,2,figsize=(7.05,4.9),layout='constrained')
 d=pd.read_csv(BASE/'validation/pso_hand_update.csv');ax=axs[0,0];ax.plot(np.arange(12),d.manual_position,'o',ms=4,mfc='none',color=C['revised'],label='Hand calculation');ax.plot(np.arange(12),d.actual_position,'x',ms=3,color='#D55E00',label='Executed update');ax.set(xlabel='Particle-coordinate index',ylabel='Updated coordinate',title='PSO velocity applied once');ax.legend(fontsize=7);panel(ax,'a')
 ax=axs[0,1];k=np.linspace(0,4,100);ax.plot(k,k,color=C['revised'],label=r'Implemented $\Delta x(k)/\Delta x(1)=k$');ax.plot(k,k*k,'--',color='#D55E00',label=r'Incorrect outer scaling: $k^2$');ax.set(xlabel='Guidance strength k',ylabel='Relative pre-bound displacement',title='SA correction already includes k');ax.legend(fontsize=7);panel(ax,'b')
 ax=axs[1,0];p=np.linspace(.26,.46,100);ax.plot(p,.26+(p+1)*.1,color='#737373',label='Legacy second mapping');ax.plot(p,p,color=C['revised'],label='Revised identity');ax.set(xlabel='Agent physical probability',ylabel='Applied probability',title='SA action contract');ax.legend(fontsize=7);panel(ax,'c')
 ax=axs[1,1];d=log('SA','legacy');sub=d[d.step<=127];ax.plot(sub.step,sub.p_mapped,color=C['revised'],lw=1,label='Agent output');ax.plot(sub.step,sub.p_applied,color='#D55E00',lw=1,label='Legacy applied p');ax.set(xlabel='Iteration (before learning starts)',ylabel='Probability',title='Observed SA mapping distortion');ax.set_ylim(.25,.47);ax.legend(fontsize=7);panel(ax,'d');save(fig,'Fig_R44_1_update_and_mapping_checks')
 # 4 x3 action and reward, one prespecified seed not selected best.
 fig,axs=plt.subplots(4,3,figsize=(7.05,7.15),layout='constrained')
 for i,a in enumerate(['GA','PSO','SA','TS']):
  d=log(a)
  for j,key in enumerate(['p_applied','k_applied','reward']):
   ax=axs[i,j];ax.plot(d.step,d[key],color=C['revised'],lw=.55);panel(ax,chr(97+i*3+j));ax.set_xlabel(a+' iteration');ax.set_ylabel(['Applied p','Applied k','Reward'][j]);ax.axvline(128,color='#999999',ls=':',lw=.7)
   if j==0 and a=='TS':ax.plot(d.step,d.effective_guided_fraction,color='#D55E00',lw=.6,label=r'$\lfloor16p\rfloor/16$');ax.legend(fontsize=6,loc='lower right')
   if j==2:
    ax.plot(d.step,d.r_progress,color='#E69F00',alpha=.65,lw=.45,label='Progress/stagnation')
    ax.scatter(d.loc[d.r_first!=0,'step'],d.loc[d.r_first!=0,'reward'],marker='^',s=14,color='#CC79A7',zorder=4)
  axs[i,0].text(.03,.92,a,transform=axs[i,0].transAxes,fontweight='bold',fontsize=9)
 save(fig,'Fig_R44_2_action_reward_trajectories')
 # Mean convergence displayed with exact paired final results, no unjustified error bands.
 fig,axs=plt.subplots(2,4,figsize=(7.15,4.8),layout='constrained')
 for i,a in enumerate(['GA','PSO','SA','TS']):
  ax=axs[0,i];arms=['legacy','revised'];
  if a=='SA':arms=['legacy','mapping_only','revised']
  if a=='PSO':arms=['legacy','transition_only','revised']
  for arm in arms:
   ds=[log(a,arm,int(seed)) for seed in [42,43,44,45,46]];arr=np.array([d.lcoe.values for d in ds]);mean=np.nanmean(arr,axis=0)
   ax.plot(ds[0].step,mean,color=C[arm],lw=1,label={'legacy':'Legacy','revised':'Revised','mapping_only':'Map only','transition_only':'Timing only'}[arm])
  ax.set_title(a,fontweight='bold');ax.set_xlabel('Iteration');ax.set_ylabel('Feasible-run mean\nlegacy-model LCOE ($/kWh)');ax.legend(fontsize=6);panel(ax,chr(97+i))
  ax=axs[1,i]
  for seed in [42,43,44,45,46]:
   vals=[S[(S.algorithm==a)&(S.arm==arm)&(S.seed==seed)].lcoe.iloc[0] for arm in arms];ax.plot(range(len(arms)),vals,'o-',ms=3,lw=.7,color='#999999',alpha=.8)
  ax.plot(range(len(arms)),[S[(S.algorithm==a)&(S.arm==arm)].lcoe.mean() for arm in arms], 'D',ms=5,color=C['revised'],label='Mean')
  ax.set_xticks(range(len(arms)),[{'legacy':'Legacy','revised':'Revised','mapping_only':'Map\nonly','transition_only':'Timing\nonly'}[a] for a in arms],fontsize=7);ax.set_ylabel('Final legacy-model\nLCOE ($/kWh)');panel(ax,chr(101+i))
 save(fig,'Fig_R44_3_paired_reexecution')
 # Every record consistency and seeded action comparison before learner is active.
 reports=[];total=0;maxmap=0.;maxrew=0.
 for p in (BASE/'results/runs').glob('*.csv'):
  d=pd.read_csv(p);total+=len(d);rerr=float(np.max(np.abs(d.reward-d[['r_progress','r_first','r_failure','r_override']].sum(axis=1))));maxrew=max(maxrew,rerr)
  if '_revised_' in p.name or '_mapping_only_' in p.name:
   me=float(np.max(abs(d.p_applied-d.p_mapped)));maxmap=max(maxmap,me);assert me<1e-7
  assert rerr<1e-10
  reports.append({'file':p.name,'steps':len(d),'reward_sum_error':rerr,'learn_updates':int(d.learn_updates.iloc[-1]),'status_channel_mismatches':int((d.current_status!=d.reward_status).sum())})
 (BASE/'validation/log_consistency.json').write_text(json.dumps({'runs':len(reports),'logged_transitions':total,'max_reward_sum_error':maxrew,'max_corrected_action_error':maxmap,'details':reports},indent=2),'utf8')
 # Paired differences are descriptive and confidence intervals exploratory (n=5).
 pairs=[]
 for a in ['GA','PSO','SA','TS']:
  for arm in ['revised']+(['mapping_only'] if a=='SA' else ['transition_only'] if a=='PSO' else []):
   b=S[(S.algorithm==a)&(S.arm=='legacy')].set_index('seed').lcoe.sort_index();r=S[(S.algorithm==a)&(S.arm==arm)].set_index('seed').lcoe.sort_index();delta=r-b;se=delta.std(ddof=1)/np.sqrt(5);ci=stats.t.ppf(.975,4)*se;pairs.append(dict(algorithm=a,comparison=arm+' minus legacy',mean_delta=float(delta.mean()),sd_delta=float(delta.std(ddof=1)),ci_low=float(delta.mean()-ci),ci_high=float(delta.mean()+ci),wins=int((delta<-1e-9).sum()),ties=int((abs(delta)<=1e-9).sum()),losses=int((delta>1e-9).sum())))
 pd.DataFrame(pairs).to_csv(BASE/'results/paired_differences.csv',index=False)
 print(agg.to_string(index=False));print('Checked',total,'logged transitions')
if __name__=='__main__':main()
