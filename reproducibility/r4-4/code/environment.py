from pathlib import Path
import importlib.util, ctypes, json,sys
import numpy as np, pandas as pd
BASE=Path(__file__).resolve().parents[1]
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
orig=module('reference_ems',BASE/'source_snapshot/microgrid_env.py')
P=ctypes.POINTER(ctypes.c_double)
class FastEnv:
 def __init__(self,config,folder='ga_repair'):
  self.config=config;self.params=config['microgrid_params'];self.days=[pd.read_csv(BASE/'source_snapshot'/folder/'data'/f'typicalday{i}.csv') for i in range(5)]
  self.ref=orig.MicrogridEnv(config,self.days);p=self.params
  self.pv=np.concatenate([self.ref._calculate_pv_power(d,1) for d in self.days]);self.wt=np.concatenate([self.ref._calculate_wind_power(d,1) for d in self.days]);self.load=np.concatenate([d.Load.values for d in self.days]);self.weights=np.array(config['typical_day_distribution'],float)
  keys=['single_ec_rated_power','single_fc_rated_power','single_tank_capacity_kg','C_BESS_initial','SOH_min','SOH_max','SOC_min','SOC_max','eta_EC_initial','eta_FC_initial','eta_ch','eta_dis','HHV','P_ch_rated','P_dis_rated','C_EC_OM_var','C_FC_OM_var','C_BESS_OM_var']
  self.par=np.array([p[k] for k in keys]+[p['fc_deg'][k] for k in ['s_chg','s_low','s_high','s_ss']]+[p['ec_deg'][k] for k in ['s_op','s_ss']],float)
  self.lib=ctypes.CDLL(str(BASE/'code/ems_kernel.dll'));self.fn=self.lib.simulate;self.fn.argtypes=[P,P,P,P,P,P,ctypes.c_int,P,P];self.fn.restype=ctypes.c_int
  self.cache={};self.calls=0;self.unique=0;self.last=None
 def evaluate_solution(self,x):
  self.calls+=1;key=tuple(np.maximum(0,np.ceil(x)).astype(int))
  if key not in self.cache:
   self.unique+=1;a=np.array(key,float);o=np.zeros(10);self.fn(*[v.ctypes.data_as(P) for v in [a,self.pv,self.wt,self.load,self.weights,self.par]],0,o.ctypes.data_as(P),None)
   code=int(o[0]);statuses=['success','fail_supply','fail_storage_overcharge','fail_storage_overdischarge'];d={'status':statuses[code]}
   if code==1:d['delta_P']=float(o[1]);f=float(o[5])
   elif code:d.update(which=['H2','BESS'][int(o[2])],delta=float(o[1]));f=float(o[5])
   else:
    d.update(delta_E=float((a[0]*self.pv[-24:].sum()+a[1]*self.wt[-24:].sum()-self.load[-24:].sum())/self.load[-24:].sum()),delta_P=0.)
    metrics={'bess_throughput':o[6],'fc_fade':o[7],'ec_fade':o[8],'bess_cal_deg':0,'bess_cyc_deg':0};repl=self.ref._calculate_replacement_costs(a,metrics);f=float(self.ref._calculate_lcoe(a,o[9],0,repl))
   self.cache[key]=(f,d)
  f,d=self.cache[key];self.last=(f,dict(d));return f,dict(d)
