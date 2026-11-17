from pathlib import Path
import json,copy,hashlib,difflib,platform,sys,time,random
import numpy as np,pandas as pd,torch
from environment import FastEnv,module,BASE
from run_experiments import load,reward_parts
report={};rng=np.random.RandomState(20261111)
# Numerical equivalence: all algorithms use their own original configs/day files.
errors=[];cases=0;feasible=0;diagmax=0.;objmax=0.
for alg in ['GA','PSO','SA','TS']:
 c,core,agentmod,helper=load(alg);env=FastEnv(c,{'GA':'ga_repair','PSO':'pso_repair','SA':'sa_repair','TS':'ts_repair'}[alg])
 xs=[rng.uniform(0,[1000,1000,200,200,1000,1000]) for _ in range(150)]
 xs += [np.array(x,float) for x in [[0]*6,[457,1,49,29,1,0],[500,0,60,30,1,0],[300,50,100,30,2,0]]]
 xs += [rng.uniform([200,0,30,15,1,0],[700,20,100,70,12,200]) for _ in range(100)]
 for x in xs:
  f,d=env.evaluate_solution(x);fr,dr=env.ref.evaluate_solution(x);cases+=1;feasible+=dr['status']=='success';objmax=max(objmax,abs(f-fr))
  bad=not np.isclose(f,fr,rtol=1e-10,atol=1e-8) or d['status']!=dr['status']
  for k in ['delta','delta_P','delta_E']:
   if k in dr:
    diagmax=max(diagmax,abs(d.get(k,np.inf)-dr[k]));bad|=not np.isclose(d.get(k,np.inf),dr[k],rtol=1e-9,atol=1e-8)
  if bad:errors.append({'alg':alg,'x':x.tolist(),'fast':[f,d],'python':[fr,dr]})
report['evaluator']={'cases':cases,'feasible_cases':feasible,'max_abs_objective_error':objmax,'max_abs_diagnostic_error':diagmax,'errors':errors}
# Explicit PSO one-step hand calculation, preserve exact random draw order.
c,core,_,_=load('PSO')
class Toy:
 config=c
 def evaluate_solution(self,x):return float(np.sum(x*x)),{'status':'fail_supply','delta_P':2.}
e=Toy();x=np.array([[10.,20,30,40,50,60],[22,28,36,45,53,66.] ]);v=np.array([[1,2,3,4,5,6],[-2,-1,0,1,2,3.]])
s=dict(particles=x.copy(),velocities=v.copy(),pbest_positions=x.copy()+5,pbest_fitness=np.array([1.,2.]),gbest_position=x[0].copy()-3,gbest_fitness=.5,gbest_diagnostics={'status':'success'})
seed=814;rng=np.random.RandomState(seed);expected=[];manual=[]
for i in range(2):
 r1,r2=rng.rand(2);activate=rng.rand()<1.;rD=rng.rand(6);uD=rng.uniform(.5,1.5,6)
 inert=.6*v[i];cog=1.5*r1*(s['pbest_positions'][i]-x[i]);soc=1.5*r2*(s['gbest_position']-x[i]);fb=2.*2.*uD;vel=inert+cog+soc+fb;expected.append(np.maximum(0,x[i]+vel))
 for j in range(6):manual.append(dict(particle=i,dimension=j,x=float(x[i,j]),old_velocity=float(v[i,j]),r1=r1,r2=r2,inertia=inert[j],cognitive=cog[j],social=soc[j],feedback=fb[j],new_velocity=vel[j],manual_position=expected[-1][j]))
np.random.seed(seed);actual,info=core.run_pso_iteration_guided(copy.deepcopy(s),np.array([1.,2.]),e)
err=float(np.max(abs(actual['particles']-expected)));assert err<1e-12
for row in manual:row['actual_position']=actual['particles'][row['particle'],row['dimension']]
pd.DataFrame(manual).to_csv(BASE/'validation/pso_hand_update.csv',index=False);report['pso_hand']={'dimensions':12,'max_abs_error':err,'velocity_used_once':True}
# SA strength linearity prior to bounds, same random numbers; wrong k^2 candidate differs.
c,core,_,_=load('SA');x=np.full(6,100.);lo=np.zeros(6);hi=np.ones(6)*1000;out=[]
for status in ['success','fail_supply','fail_storage_overcharge','fail_storage_overdischarge']:
 d={'status':status,'delta_E':.4,'delta_P':2.};a=core._guided_step_paper_style(x,d,1.,lo,hi,np.random.RandomState(312));b=core._guided_step_paper_style(x,d,2.,lo,hi,np.random.RandomState(312));er=float(max(abs((b-x)-2*(a-x))));assert er<1e-12
 for j in range(6):out.append(dict(status=status,dimension=j,x=x[j],k1_candidate=a[j],k2_candidate=b[j],expected_k2=x[j]+2*(a[j]-x[j]),wrong_outer_k_candidate=x[j]+2*(b[j]-x[j]),error=er))
pd.DataFrame(out).to_csv(BASE/'validation/sa_hand_update.csv',index=False);report['sa_hand']={'cases':24,'max_abs_error':max(r['error'] for r in out),'strength_used_once':True}
# Agent bounds, supplied physical action vs old SA double map.
actions=[]
for alg in ['GA','PSO','SA','TS']:
 c,_,agentmod,helper=load(alg);bounds=c['sac_agent_params']['action_bounds'];keys=list(bounds)
 for u in [-1.,0.,1.]:
  mapped=np.array([lo+(u+1)*.5*(hi-lo) for lo,hi in bounds.values()]);applied=helper._map_to_bounds(mapped,bounds) if alg=='SA' else mapped.copy()
  for j,k in enumerate(keys):actions.append(dict(algorithm=alg,name=k,u=u,mapped=mapped[j],legacy_applied=applied[j],revised_applied=mapped[j]))
pd.DataFrame(actions).to_csv(BASE/'validation/action_mapping.csv',index=False)
# Reward arithmetic: cases all algorithms / versions including overriding legacy bonuses.
r=[]
for alg in ['GA','PSO','SA','TS']:
 c,_,_,_=load(alg)
 for arm in ['legacy','revised']:
  for label,old,new,d,first in [('improve',1.,.99,{'status':'success'},False),('stagnate',1.,1.,{'status':'success'},False),('fail',1.,1.,{'status':'fail_supply'},False),('first',np.inf,.9,{'status':'success'},True),('infeasible_improve',1000020.,1000010.,{'status':'fail_supply'},False)]:
   z=reward_parts(old,new,d,first,alg,arm,c);assert abs(z['reward']-sum(z[k] for k in ['r_progress','r_first','r_failure','r_override']))<1e-12;r.append(dict(algorithm=alg,arm=arm,case=label,**z))
pd.DataFrame(r).to_csv(BASE/'validation/reward_cases.csv',index=False);report['reward_cases']=len(r)
# Source learner blocks do not reparameterize; validate detached vs pathwise gradient.
a=torch.tensor([.3],requires_grad=True);b=torch.distributions.Normal(a,torch.ones_like(a));z=b.sample();hasgrad_before=z.requires_grad;z=b.rsample();z.sum().backward();report['policy_pathwise']={'legacy_sample_requires_grad':hasgrad_before,'corrected_rsample_mean_grad':a.grad.item()};assert a.grad.item()==1.
report['all_passed']=not errors
(BASE/'validation/tests.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2));assert not errors
