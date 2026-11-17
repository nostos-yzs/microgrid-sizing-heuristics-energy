from pathlib import Path
import sys,json,difflib,platform,hashlib,subprocess
import numpy as np,pandas as pd,torch,matplotlib,scipy
from environment import BASE
from run_experiments import load,ENTRIES
rows=[]
for alg in ENTRIES:
 c,_,_,_=load(alg);p=c['sac_agent_params'];d=dict(algorithm=alg,steps={'GA':250,'PSO':200,'SA':3600,'TS':200}[alg],seeds='42,43,44,45,46',action_order=list(p['action_bounds']),action_bounds=p['action_bounds'],batch_size=p['batch_size'],buffer_size=p['buffer_size'],gamma=p['gamma'],tau=p['tau'],alpha=p['alpha'],actor_lr=p['actor_lr'],critic_lr=p['critic_lr'],device=p['device'],core_settings=c.get('ga_params',c.get('pso_core_params',c.get('sa_params',c.get('ts_params')))),representative_weights=c['typical_day_distribution'],stochasticity=c['stochasticity_params'])
 rows.append(d)
def default(x):return x.tolist() if isinstance(x,np.ndarray) else str(x)
(BASE/'audit/settings.json').write_text(json.dumps(rows,indent=2,default=default),encoding='utf8')
env={'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'matplotlib':matplotlib.__version__,'scipy':scipy.__version__,'cuda_available':torch.cuda.is_available(),'torch_threads':1,'deterministic_algorithms':True,'seed_scopes':['Python random','NumPy global','Torch CPU','SA/TS local RandomState'],'wall_time_caveat':'Runs were executed concurrently in separate processes; timings include learner and evaluator but are not controlled runtime benchmarks. Do not reuse as R4.5 speedup evidence.'}
(BASE/'audit/environment.json').write_text(json.dumps(env,indent=2),encoding='utf8')
(BASE/'code/requirements.txt').write_text('\n'.join([f'numpy=={np.__version__}',f'pandas=={pd.__version__}',f'torch=={torch.__version__}',f'matplotlib=={matplotlib.__version__}',f'scipy=={scipy.__version__}'])+'\n',encoding='utf8')
diff=[];changed=[]
for p in (BASE/'patched').rglob('*.py'):
 rel=p.relative_to(BASE/'patched');old=BASE/'source_snapshot'/rel
 if old.exists() and old.read_bytes()!=p.read_bytes():
  diff.extend(difflib.unified_diff(old.read_text('utf8').splitlines(True),p.read_text('utf8').splitlines(True),fromfile='source_snapshot/'+str(rel),tofile='patched/'+str(rel)));changed.append(str(rel))
(BASE/'audit/before_after.patch').write_text(''.join(diff),encoding='utf8');(BASE/'audit/changed_files.json').write_text(json.dumps(changed,indent=2),encoding='utf8')
# Hash every delivered source/input for provenance.
files=[]
for folder in ['source_snapshot','patched','inputs','code']:
 for p in (BASE/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:files.append({'file':str(p.relative_to(BASE)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(BASE/'audit/file_manifest.json').write_text(json.dumps(files,indent=2),encoding='utf8');print('Settings, environment, patch and hashes exported')
