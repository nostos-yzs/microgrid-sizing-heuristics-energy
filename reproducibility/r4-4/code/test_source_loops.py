import ast,os,sys,collections,csv,random,json,contextlib,io
from pathlib import Path
import numpy as np,pandas as pd,torch
from environment import BASE,FastEnv,module
from run_experiments import load,run,ENTRIES
os.chdir(BASE/'validation')
class NoLogger:
 def __init__(self,*a,**k):pass
 def log_iter(self,*a,**k):pass
 def save_trace_df(self,*a,**k):pass
report=[]
for alg,main,fn in [('GA','main_ga_comparison_patched.py','run_sac_guided_ga'),('PSO','main_comparison.py','run_sac_guided_pso'),('SA','main_sa_comparison_replaced_v3.py','run_sa_rl_guided'),('TS','main_tabu_comparison_REPLACEMENT.py','run_sac_guided_ts')]:
 c,core,am,helper=load(alg);c['sac_agent_params']['corrected_policy']=False
 if alg=='SA':c['simulation_params'].update(n_layers=1,inner_loops=12)
 folder=ENTRIES[alg][0];text=(BASE/'source_snapshot'/folder/main).read_text('utf8');tree=ast.parse(text)
 ns={'np':np,'torch':torch,'random':random,'pd':pd,'collections':collections,'csv':csv,'os':os,'time':__import__('time'),'SACAgent':am.SACAgent,'ExperimentLogger':NoLogger,'ga_core':core,'pso_core_sac':core,'ts_core':core,'initialize_sa':getattr(core,'initialize_sa_state',None),'run_sa_iteration':getattr(core,'run_sa_iteration',None)}
 ns.update(vars(helper))
 funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_read_layers',fn]]
 for n in funcs:exec(compile(ast.Module(body=[n],type_ignores=[]),str(main),'exec'),ns)
 np.random.seed(42);random.seed(42);torch.manual_seed(42);torch.set_num_threads(1)
 with contextlib.redirect_stdout(io.StringIO()):out=ns[fn](c,FastEnv(c,folder),12,42)
 expected=pd.read_csv(BASE/'results/smoke'/f'{alg}_legacy_42.csv').best_objective.values
 actual=np.array(out['fitness_history']);err=float(max(abs(actual-expected)));assert err<1e-8,(alg,err)
 report.append({'algorithm':alg,'selected_main_function':fn,'steps':12,'max_abs_history_difference':err,'seed':42,'scope':'source loop replay vs instrumented legacy harness; common explicit RNG seeds'})
(BASE/'validation/source_loop_equivalence.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report,indent=2))
