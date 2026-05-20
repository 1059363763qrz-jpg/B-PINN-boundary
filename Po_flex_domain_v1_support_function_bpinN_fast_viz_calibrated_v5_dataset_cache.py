# -*- coding: utf-8 -*-
"""
flex-domain v1 support-function B-PINN (experiment-ready v2)
推荐调试顺序:
1) RUN_MODE="smoke": 检查 ready-check/active_records_match/max_abs_h_res/训练无 NaN/inf/support_res_raw≈0
2) RUN_MODE="small": 检查 val_boundary_sup、val_phys_flex、val_raw_pcc_p/q、P0/Q0/t 误差是否下降，CDF 是否可输出
3) RUN_MODE="fast": 正式快速实验
# 第一次跑：DATASET_CACHE_MODE="auto" 或 "rebuild"
# 后续复现实验：DATASET_CACHE_MODE="load_only"
# 若只看训练过程：RUN_FULL_OPF_EVAL=False
"""
import math, dataclasses, csv
from pathlib import Path
import json, pickle
from datetime import datetime
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

import gurobipy as gp
from gurobipy import GRB

RUN_MODE = "fast_plus"
# 可选: "smoke" | "small" | "fast" | "full"
if RUN_MODE == "smoke":
    NUM_SCENARIOS, MC_PER_SCENARIO, N_THETA, EPOCHS = 2, 3, 4, 2
    RUN_MULTI_TEST=False; MC_EVAL_MULTI=50; EVAL_THETA_SAMPLES=10
    USE_H_QUANTILE_LOSS=False; LAM_H_QUANTILE=0.0
elif RUN_MODE == "small":
    NUM_SCENARIOS, MC_PER_SCENARIO, N_THETA, EPOCHS = 20, 10, 4, 50
    RUN_MULTI_TEST=False; MC_EVAL_MULTI=100; EVAL_THETA_SAMPLES=20
    USE_H_QUANTILE_LOSS=True; LAM_H_QUANTILE=0.01
elif RUN_MODE == "fast":
    NUM_SCENARIOS, MC_PER_SCENARIO, N_THETA, EPOCHS = 300, 30, 12, 400
    RUN_MULTI_TEST=True; MC_EVAL_MULTI=400; EVAL_THETA_SAMPLES=40
    USE_H_QUANTILE_LOSS=True; LAM_H_QUANTILE=0.05
elif RUN_MODE == "fast_plus":
    NUM_SCENARIOS, MC_PER_SCENARIO, N_THETA, EPOCHS = 300, 30, 12, 200
    RUN_MULTI_TEST=False; MC_EVAL_MULTI=400; EVAL_THETA_SAMPLES=40  # first run disabled for speed; enable after calibration is satisfactory
    USE_H_QUANTILE_LOSS=True; LAM_H_QUANTILE=0.05
elif RUN_MODE == "full":
    NUM_SCENARIOS, MC_PER_SCENARIO, N_THETA, EPOCHS = 1000, 60, 16, 700
    RUN_MULTI_TEST=True; MC_EVAL_MULTI=400; EVAL_THETA_SAMPLES=40
    USE_H_QUANTILE_LOSS=True; LAM_H_QUANTILE=0.10
else:
    raise ValueError(f"Unsupported RUN_MODE={RUN_MODE}")
THETA_LIST = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
FAST_TRAINING = True
QUANTILE_EVERY_N_BATCHES = 5
H_QUANTILE_START_EPOCH = 30
FAST_EPOCHS = 150
if FAST_TRAINING and RUN_MODE == "fast":
    EPOCHS = FAST_EPOCHS
BATCH_SIZE, LR = (4096 if torch.cuda.is_available() else 2048), 1e-3
LAM_BOUNDARY_SUP, LAM_DISPATCH_SUP, LAM_PHYS_FLEX, LAM_SUPPORT_CONSIST, LAM_T_SUP = 0.05, 0.05, 0.08, 0.01, 0.05
H_QUANTILE_TAUS = [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
N_GMM_COMPONENTS = 6
GMM_SIGMA_FLOOR = 1e-3
USE_H_CDF_LOSS = True
Z_CDF_GRID = np.array([-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5],dtype=float)
FAST_STAGE_A_EPOCHS = 80
FAST_STAGE_B_EPOCHS = 70
FAST_STAGE_C_EPOCHS = 50
FAST_TOTAL_EPOCHS = FAST_STAGE_A_EPOCHS + FAST_STAGE_B_EPOCHS + FAST_STAGE_C_EPOCHS
USE_THETA_REWEIGHT = True
THETA_REWEIGHT_START_EPOCH = 100
THETA_REWEIGHT_UPDATE_EVERY = 25
THETA_REWEIGHT_POWER = 1.5
THETA_REWEIGHT_MAX = 5.0
THETA_REWEIGHT_MIN = 0.5
USE_ADAPTIVE_CDF_GRID = True
CDF_GRID_TAUS = [0.01,0.03,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.97,0.99]
CDF_STANDARD_GRID = [-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5]

ATOM_MASS_THRESHOLD = 0.20
POSSIBLE_ATOM_MASS_THRESHOLD = 0.10
HIGH_ARMS_THRESHOLD = 10.0
STABILIZE_OPF_DISPATCH, SUPPORT_EPS, W_QG_STAB, W_PG_STAB, W_PQ0_STAB = True, 1e-5, 1.0, 0.01, 1e-5
PRIOR_SIGMA, INIT_RHO, BETA_KL_MAX, KL_WARMUP_EPOCHS = 1.0, -5.0, 1.0, 500
RUN_SANITY_CHECKS = True
N_TEST_SCENARIOS = 20
SEED_DATA, SEED_TRAIN, SEED_EVAL = 0, 0, 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_DATASET_CACHE = True
DATASET_CACHE_MODE = "auto"  # "auto" | "rebuild" | "load_only"
DATASET_CACHE_DIR = "dataset_cache"
DATASET_CACHE_TAG = "flex33_fastplus_300scen_30mc_12theta_seed0_v1"
DATASET_CACHE_NPZ = f"{DATASET_CACHE_DIR}/{DATASET_CACHE_TAG}.npz"
DATASET_CACHE_PICKLE = f"{DATASET_CACHE_DIR}/{DATASET_CACHE_TAG}_active.pkl"
DATASET_CACHE_META = f"{DATASET_CACHE_DIR}/{DATASET_CACHE_TAG}_meta.json"
DATASET_VERSION = "flex_support_dataset_v1"
SAVE_TRAINING_RESULT = True
TRAINING_RESULT_DIR = "training_results"
TRAINING_RUN_TAG = "fastplus_gmm6_stageABC_seed0"
RUN_FULL_OPF_EVAL = True
POLYGON_COORD_MAX = 20.0
POLYGON_CONVEX_CLEANUP = True
H_QUANTILE_BATCH_SCENARIOS = 8
H_QUANTILE_BATCH_THETAS = 4
if (not FAST_TRAINING) and RUN_MODE in ["fast","full"]:
    H_QUANTILE_BATCH_SCENARIOS = 32
    H_QUANTILE_BATCH_THETAS = None
THETA_LIST = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)

@dataclasses.dataclass
class GridCase:
    n_bus:int; root:int; from_bus:np.ndarray; to_bus:np.ndarray; r:np.ndarray; x:np.ndarray
    vmin:float; vmax:float; fmax_p:np.ndarray; fmax_q:np.ndarray; pd_base:np.ndarray; qd_base:np.ndarray
    pv_buses:np.ndarray; pv_pmax:np.ndarray; pv_pf:float; gen_buses:np.ndarray
    pg_min:np.ndarray; pg_max:np.ndarray; qg_min:np.ndarray; qg_max:np.ndarray
    children:List[List[int]]; parent_branch:np.ndarray; parent_bus:np.ndarray
    topo_order:List[int]; rev_topo_order:List[int]; in_branches:List[List[int]]; out_branches:List[List[int]]

def _build_radial_topology(n_bus, root, fb, tb):
    n_br = fb.size; children=[[] for _ in range(n_bus)]
    parent_branch=-np.ones(n_bus,dtype=int); parent_bus=-np.ones(n_bus,dtype=int)
    for l in range(n_br):
        i,j=int(fb[l]),int(tb[l]); children[i].append(j); parent_branch[j]=l; parent_bus[j]=i
    topo=[]; st=[root]
    while st:
        i=st.pop(); topo.append(i); st.extend(children[i])
    rev=topo[::-1]
    inb=[[] for _ in range(n_bus)]; outb=[[] for _ in range(n_bus)]
    for l in range(n_br):
        i,j=int(fb[l]),int(tb[l]); outb[i].append(l); inb[j].append(l)
    return children,parent_branch,parent_bus,topo,rev,inb,outb

def build_ieee33_case():
    fb=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,18,19,20,2,22,23,5,25,26,27,28,29,30,31],dtype=int)
    tb=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],dtype=int)
    r=np.array([0.0922,0.4930,0.3660,0.3811,0.8190,0.1872,0.7114,1.0300,1.0440,0.1966,0.3744,1.4680,0.5416,0.5910,0.7463,1.2890,0.7320,0.1640,1.5042,0.4095,0.7089,0.4512,0.8980,0.8960,0.2030,0.2842,1.0590,0.8042,0.5075,0.9744,0.3105,0.3410],dtype=float)*0.01
    x=np.array([0.0470,0.2511,0.1864,0.1941,0.7070,0.6188,0.2351,0.7400,0.7400,0.0650,0.1238,1.1550,0.7129,0.5260,0.5450,1.7210,0.5740,0.1565,1.3554,0.4784,0.9373,0.3083,0.7091,0.7011,0.1034,0.1447,0.9337,0.7006,0.2585,0.9630,0.3619,0.5302],dtype=float)*0.01
    nb=33; nl=32
    pd=np.array([0,0.10,0.09,0.12,0.06,0.06,0.20,0.20,0.06,0.06,0.045,0.06,0.06,0.12,0.06,0.06,0.06,0.09,0.09,0.09,0.09,0.09,0.09,0.42,0.42,0.06,0.06,0.06,0.12,0.20,0.15,0.21,0.06],dtype=float)
    qd=np.array([0,0.06,0.04,0.08,0.03,0.02,0.10,0.10,0.02,0.02,0.03,0.035,0.035,0.08,0.01,0.02,0.02,0.04,0.04,0.04,0.04,0.04,0.05,0.20,0.20,0.025,0.025,0.02,0.07,0.60,0.07,0.10,0.04],dtype=float)
    pv_b=np.array([6,13,23,29],dtype=int); pv_max=np.array([0.35,0.30,0.60,0.50],dtype=float)
    gen_b=np.array([12,17,24,32],dtype=int)
    pg_min=np.zeros(4); pg_max=np.array([0.45,0.40,0.90,0.75],dtype=float)
    qg_min=np.array([-1.20,-1.00,-1.80,-1.50],dtype=float); qg_max=np.array([1.20,1.00,1.80,1.50],dtype=float)
    c,pb,pbu,to,rt,inb,outb=_build_radial_topology(nb,0,fb,tb)
    return GridCase(nb,0,fb,tb,r,x,0.95,1.05,np.full(nl,5.0),np.full(nl,5.0),pd,qd,pv_b,pv_max,0.98,gen_b,pg_min,pg_max,qg_min,qg_max,c,pb,pbu,to,rt,inb,outb)

def sample_trunc_normal(mu,sigma,lo=0.0,hi=None):
    x=np.random.normal(mu,sigma); x=max(lo,x)
    if hi is not None: x=min(hi,x)
    return float(x)

def sample_scenario_means(case,rng):
    pd=case.pd_base.copy(); qd=case.qd_base.copy()
    for i in range(case.n_bus):
        if pd[i]>1e-9:
            s=rng.uniform(0.75,1.25); pd[i]*=s; qd[i]*=s
    pr=np.zeros(case.n_bus)
    for k,b in enumerate(case.pv_buses): pr[b]=rng.uniform(0.15,0.95)*case.pv_pmax[k]
    qr=pr*math.tan(math.acos(case.pv_pf)); return pd,qd,pr,qr

def make_feature_vector(case,pd,pr):
    return np.concatenate([pd,pr[case.pv_buses],np.array([pd.sum(),pr.sum()],dtype=float)])

def solve_flex_support_gurobi_33bus(case,pd,qd,pr,qr,alpha,beta,return_detail=False,stabilize_dispatch=STABILIZE_OPF_DISPATCH):
    nb,nl=case.n_bus,case.from_bus.size
    def build_model():
        m=gp.Model(); m.Params.OutputFlag=0
        P=m.addVars(nl,lb=-GRB.INFINITY,ub=GRB.INFINITY,name='P'); Q=m.addVars(nl,lb=-GRB.INFINITY,ub=GRB.INFINITY,name='Q')
        V=m.addVars(nb,lb=case.vmin**2,ub=case.vmax**2,name='V')
        P0=m.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,name='P0'); Q0=m.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,name='Q0')
        Pg=m.addVars(len(case.gen_buses),lb=-GRB.INFINITY,ub=GRB.INFINITY,name='Pg'); Qg=m.addVars(len(case.gen_buses),lb=-GRB.INFINITY,ub=GRB.INFINITY,name='Qg')
        return m,P,Q,V,P0,Q0,Pg,Qg
    def add_constraints(m,P,Q,V,P0,Q0,Pg,Qg):
        for g in range(len(case.gen_buses)):
            m.addConstr(Pg[g]>=case.pg_min[g]); m.addConstr(Pg[g]<=case.pg_max[g]); m.addConstr(Qg[g]>=case.qg_min[g]); m.addConstr(Qg[g]<=case.qg_max[g])
        for l in range(nl):
            m.addConstr(P[l]<=case.fmax_p[l]); m.addConstr(P[l]>=-case.fmax_p[l]); m.addConstr(Q[l]<=case.fmax_q[l]); m.addConstr(Q[l]>=-case.fmax_q[l])
        root_out=case.out_branches[case.root]
        m.addConstr(gp.quicksum(P[l] for l in root_out)==P0); m.addConstr(gp.quicksum(Q[l] for l in root_out)==Q0)
        bus_to_gen={int(b):g for g,b in enumerate(case.gen_buses)}
        for i in range(case.n_bus):
            if i==case.root: continue
            inb,outb=case.in_branches[i],case.out_branches[i]
            pg_i=Pg[bus_to_gen[i]] if i in bus_to_gen else 0.0; qg_i=Qg[bus_to_gen[i]] if i in bus_to_gen else 0.0
            pinj=pr[i]-pd[i]+pg_i; qinj=qr[i]-qd[i]+qg_i
            m.addConstr(gp.quicksum(P[l] for l in inb)-gp.quicksum(P[l] for l in outb)+pinj==0)
            m.addConstr(gp.quicksum(Q[l] for l in inb)-gp.quicksum(Q[l] for l in outb)+qinj==0)
        m.addConstr(V[case.root]==1.0)
        for l in range(nl):
            i,j=int(case.from_bus[l]),int(case.to_bus[l]); m.addConstr(V[j]==V[i]-2*(case.r[l]*P[l]+case.x[l]*Q[l]))
    m,P,Q,V,P0,Q0,Pg,Qg=build_model(); add_constraints(m,P,Q,V,P0,Q0,Pg,Qg)
    m.setObjective(alpha*P0+beta*Q0,GRB.MAXIMIZE); m.optimize()
    if m.Status!=GRB.OPTIMAL: return {'ok':False} if return_detail else float('nan')
    h_stage1=float(alpha*P0.X+beta*Q0.X)
    stage1_out={'ok':True,'h':h_stage1,'h_stage1':h_stage1,'P0':float(P0.X),'Q0':float(Q0.X),'P':np.array([P[l].X for l in range(nl)]),'Q':np.array([Q[l].X for l in range(nl)]),'V':np.array([V[i].X for i in range(nb)]),'Pg':np.array([Pg[g].X for g in range(len(case.gen_buses))]),'Qg':np.array([Qg[g].X for g in range(len(case.gen_buses))]),'alpha':float(alpha),'beta':float(beta),'stabilized':False}
    if not stabilize_dispatch:
        return stage1_out if return_detail else h_stage1
    m2,P2,Q2,V2,P0_2,Q0_2,Pg2,Qg2=build_model(); add_constraints(m2,P2,Q2,V2,P0_2,Q0_2,Pg2,Qg2)
    m2.addConstr(alpha*P0_2+beta*Q0_2>=h_stage1-SUPPORT_EPS)
    obj=gp.QuadExpr()
    for g in range(len(case.gen_buses)): obj += W_QG_STAB*Qg2[g]*Qg2[g] + W_PG_STAB*Pg2[g]*Pg2[g]
    obj += W_PQ0_STAB*(P0_2*P0_2+Q0_2*Q0_2)
    m2.setObjective(obj,GRB.MINIMIZE); m2.optimize()
    if m2.Status!=GRB.OPTIMAL: return stage1_out if return_detail else h_stage1
    h_final=float(alpha*P0_2.X+beta*Q0_2.X)
    stage2_out={'ok':True,'h':h_final,'h_stage1':h_stage1,'P0':float(P0_2.X),'Q0':float(Q0_2.X),'P':np.array([P2[l].X for l in range(nl)]),'Q':np.array([Q2[l].X for l in range(nl)]),'V':np.array([V2[i].X for i in range(nb)]),'Pg':np.array([Pg2[g].X for g in range(len(case.gen_buses))]),'Qg':np.array([Qg2[g].X for g in range(len(case.gen_buses))]),'alpha':float(alpha),'beta':float(beta),'stabilized':True}
    if abs(stage2_out['h']-(alpha*stage2_out['P0']+beta*stage2_out['Q0']))>1e-5: stage2_out['h']=alpha*stage2_out['P0']+beta*stage2_out['Q0']
    return stage2_out if return_detail else stage2_out['h']

def get_active_constraint_signature(case,sol,tol=1e-4):
    pg,qg,v,p,q=sol['Pg'],sol['Qg'],sol['V'],sol['P'],sol['Q']; names=[]; bits=[]
    for g in range(len(case.gen_buses)): names += [f'Pg_min_g{g}',f'Pg_max_g{g}',f'Qg_min_g{g}',f'Qg_max_g{g}']; bits += [int(pg[g]<=case.pg_min[g]+tol),int(pg[g]>=case.pg_max[g]-tol),int(qg[g]<=case.qg_min[g]+tol),int(qg[g]>=case.qg_max[g]-tol)]
    for i in range(case.n_bus): names += [f'Vmin_bus{i+1:02d}',f'Vmax_bus{i+1:02d}']; bits += [int(v[i]<=case.vmin**2+tol),int(v[i]>=case.vmax**2-tol)]
    for l in range(case.from_bus.size): names += [f'Pline_pos_l{l:02d}',f'Pline_neg_l{l:02d}',f'Qline_pos_l{l:02d}',f'Qline_neg_l{l:02d}']; bits += [int(p[l]>=case.fmax_p[l]-tol),int(p[l]<=-case.fmax_p[l]+tol),int(q[l]>=case.fmax_q[l]-tol),int(q[l]<=-case.fmax_q[l]+tol)]
    sig=tuple(bits); act=[n for n,b in zip(names,bits) if b==1]; return sig,act,names

def summarize_active_patterns(active_records,all_names,top_k=10):
    from collections import Counter, defaultdict
    if not active_records: return
    cnt=Counter([r['signature'] for r in active_records]); n=len(active_records)
    print(f'[active] total={n}, unique={len(cnt)}')
    rows=[]
    YH_eval=[]
    XMU_eval=[]
    for i,(sig,c) in enumerate(cnt.most_common(top_k),1):
        act=';'.join([all_names[j] for j,b in enumerate(sig) if b==1]); rows.append((i,c,100*c/n,act)); print(i,c,act)
    with open('active_constraint_patterns_flex_v1.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['pattern_id','count','percentage','active_names']); [w.writerow([a,b,f'{c:.6f}',d]) for a,b,c,d in rows]
    single=np.zeros(len(all_names),dtype=int)
    for r in active_records: single += np.array(r['signature'])
    with open('active_constraint_single_rates_flex_v1.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['constraint_name','count','active_rate'])
        for i,nm in enumerate(all_names): w.writerow([nm,int(single[i]),f'{single[i]/max(1,n):.8f}'])
    by_theta=defaultdict(list)
    for r in active_records: by_theta[r['theta_idx']].append(r['signature'])
    with open('active_constraint_by_theta_flex_v1.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['theta_idx','unique_patterns','top1_ratio','fallback_ratio'])
        for k,v in sorted(by_theta.items()):
            c=Counter(v); top=c.most_common(1)[0][1]/len(v); fr=np.mean([1.0 if r.get('fallback',False) else 0.0 for r in active_records if r['theta_idx']==k]); w.writerow([k,len(c),f'{top:.6f}',f'{fr:.6f}'])
    with open('active_constraint_records_flex_v1.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['scenario_idx','mc_idx','theta_idx','theta','alpha','beta','fallback','fallback_type','active_names'])
        for r in active_records: w.writerow([r.get('scenario_idx',-1),r.get('mc_idx',-1),r.get('theta_idx',-1),r.get('theta',np.nan),r.get('alpha',np.nan),r.get('beta',np.nan),r.get('fallback',False),r.get('fallback_type','none'),';'.join(r.get('active_names',[]))])

def generate_flex_dataset(case,num_scenarios=NUM_SCENARIOS,mc_per_scenario=MC_PER_SCENARIO,theta_list=THETA_LIST,seed=SEED_DATA):
    rng=np.random.default_rng(seed); np.random.seed(seed)
    T=len(theta_list); theta_feat=np.stack([np.cos(theta_list),np.sin(theta_list)],axis=1)
    XMU=[]; XREAL=[]; YH=[]; YP0=[]; YQ0=[]; YPG=[]; YQG=[]; active=[]; alln=None; drop=0
    fallback_fill_count_total = 0; mean_fallback_count_total = 0; infeasible_opf_count_total = 0; dropped_scenarios = 0
    for s in range(num_scenarios):
        pd_mu,qd_mu,pr_mu,qr_mu=sample_scenario_means(case,rng); xmu=make_feature_vector(case,pd_mu,pr_mu)
        xr=[]; yh=np.full((mc_per_scenario,T),np.nan); yp0=np.full((mc_per_scenario,T),np.nan); yq0=np.full((mc_per_scenario,T),np.nan); ypg=np.full((mc_per_scenario,T,len(case.gen_buses)),np.nan); yqg=np.full((mc_per_scenario,T,len(case.gen_buses)),np.nan); active_grid=[[None for _ in range(T)] for _ in range(mc_per_scenario)]
        ok_scene=True
        for m in range(mc_per_scenario):
            pd=pd_mu.copy(); pr=pr_mu.copy(); std_pd=0.10*np.maximum(pd_mu,1e-3); std_pr=0.12*np.maximum(pr_mu,1e-3)
            for i in range(case.n_bus):
                if pd_mu[i]>1e-9: pd[i]=sample_trunc_normal(pd_mu[i],std_pd[i],0.0,None)
            for k,b in enumerate(case.pv_buses): pr[b]=sample_trunc_normal(pr_mu[b],std_pr[b],0.0,float(case.pv_pmax[k]))
            qd=qd_mu*(pd/np.maximum(pd_mu,1e-6)); qr=pr*math.tan(math.acos(case.pv_pf)); xr.append(make_feature_vector(case,pd,pr))
            for j,th in enumerate(theta_list):
                sol=solve_flex_support_gurobi_33bus(case,pd,qd,pr,qr,float(np.cos(th)),float(np.sin(th)),return_detail=True,stabilize_dispatch=STABILIZE_OPF_DISPATCH)
                if not sol['ok']:
                    infeasible_opf_count_total += 1
                    continue
                yh[m,j],yp0[m,j],yq0[m,j]=sol['h'],sol['P0'],sol['Q0']; ypg[m,j,:]=sol['Pg']; yqg[m,j,:]=sol['Qg']
                sig,an,alln=get_active_constraint_signature(case,sol); active_grid[m][j]={'signature':sig,'active_names':an,'theta_idx':j,'theta':float(th),'alpha':float(np.cos(th)),'beta':float(np.sin(th)),'fallback':False,'fallback_type':'none','scenario_idx':s,'mc_idx':m}
            if not ok_scene: break
        # per-theta fallback fill
        for j in range(T):
            valid=np.where(np.isfinite(yh[:,j]))[0]; miss=np.where(~np.isfinite(yh[:,j]))[0]
            if miss.size==0: continue
            if valid.size>0:
                pick=rng.choice(valid,size=miss.size,replace=True)
                for mm,pp in zip(miss,pick):
                    yh[mm,j],yp0[mm,j],yq0[mm,j]=yh[pp,j],yp0[pp,j],yq0[pp,j]
                    ypg[mm,j,:],yqg[mm,j,:]=ypg[pp,j,:],yqg[pp,j,:]
                    active_grid[mm][j]=dict(active_grid[pp][j]); active_grid[mm][j]['fallback']=True; active_grid[mm][j]['fallback_type']='copy_same_theta'; active_grid[mm][j]['scenario_idx']=s; active_grid[mm][j]['mc_idx']=mm
                    fallback_fill_count_total += 1
            else:
                solm=solve_flex_support_gurobi_33bus(case,pd_mu,qd_mu,pr_mu,qr_mu,float(np.cos(theta_list[j])),float(np.sin(theta_list[j])),return_detail=True,stabilize_dispatch=STABILIZE_OPF_DISPATCH)
                if solm['ok']:
                    yh[:,j]=solm['h']; yp0[:,j]=solm['P0']; yq0[:,j]=solm['Q0']; ypg[:,j,:]=solm['Pg']; yqg[:,j,:]=solm['Qg']
                    sig,an,alln=get_active_constraint_signature(case,solm)
                    for mm in range(mc_per_scenario):
                        active_grid[mm][j]={'signature':sig,'active_names':an,'theta_idx':j,'theta':float(theta_list[j]),'alpha':float(np.cos(theta_list[j])),'beta':float(np.sin(theta_list[j])),'fallback':True,'fallback_type':'mean_same_theta','scenario_idx':s,'mc_idx':mm}
                        mean_fallback_count_total += 1
                else:
                    ok_scene=False
        scene_finite = np.all(np.isfinite(yh)) and np.all(np.isfinite(yp0)) and np.all(np.isfinite(yq0)) and np.all(np.isfinite(ypg)) and np.all(np.isfinite(yqg))
        scene_active_ok = all(active_grid[m][j] is not None for m in range(mc_per_scenario) for j in range(T))
        if (not ok_scene) or (not scene_finite) or (not scene_active_ok):
            drop+=1; dropped_scenarios += 1; continue
        XMU.append(xmu); XREAL.append(np.stack(xr)); YH.append(yh); YP0.append(yp0); YQ0.append(yq0); YPG.append(ypg); YQG.append(yqg)
        for mm in range(mc_per_scenario):
            for jj in range(T):
                active.append(active_grid[mm][jj])
    XMU=np.array(XMU); XREAL=np.array(XREAL); YH=np.array(YH); YP0=np.array(YP0); YQ0=np.array(YQ0); YPG=np.array(YPG); YQG=np.array(YQG)
    print(f'[flex-dataset] XMU shape={XMU.shape}'); print(f'[flex-dataset] XREAL shape={XREAL.shape}'); print(f'[flex-dataset] THETA_FEAT shape={theta_feat.shape}')
    print(f'[flex-dataset] YH shape={YH.shape}'); print(f'[flex-dataset] YP0/YQ0/YPG/YQG shape={YP0.shape}/{YQ0.shape}/{YPG.shape}/{YQG.shape}')
    expected=max(0,XMU.shape[0])*mc_per_scenario*T
    print(f'[flex-dataset] total_OPF_labels = {expected}, dropped_scenarios={drop}')
    print(f'[flex-dataset] dropped_scenarios = {dropped_scenarios}')
    print(f'[flex-dataset] infeasible_opf_count_total = {infeasible_opf_count_total}')
    print(f'[flex-dataset] fallback_fill_count_total = {fallback_fill_count_total}')
    print(f'[flex-dataset] mean_fallback_count_total = {mean_fallback_count_total}')
    print(f'[flex-dataset] active_records count = {len(active)}')
    print(f'[flex-dataset] active_records_match = {len(active)==expected}')
    if XMU.size>0: print(f'[flex-dataset] h/P0/Q0/Pg/Qg ranges = {YH.min():.4f}-{YH.max():.4f} / {YP0.min():.4f}-{YP0.max():.4f} / {YQ0.min():.4f}-{YQ0.max():.4f} / {YPG.min():.4f}-{YPG.max():.4f} / {YQG.min():.4f}-{YQG.max():.4f}')
    return XMU,XREAL,theta_feat,YH,YP0,YQ0,YPG,YQG,active,alln

def flatten_flex_dataset(XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG):
    N,M,T=YH.shape; d=XMU.shape[1]; ng=YPG.shape[-1]
    xmu=np.repeat(XMU[:,None,None,:],M*T,axis=1).reshape(-1,d)
    xreal=np.repeat(XREAL[:,:,None,:],T,axis=2).reshape(-1,d)
    theta=np.repeat(THETA_FEAT[None,None,:,:],N*M,axis=0).reshape(-1,2)
    yh=YH.reshape(-1,1); yp0=YP0.reshape(-1,1); yq0=YQ0.reshape(-1,1); ypg=YPG.reshape(-1,ng); yqg=YQG.reshape(-1,ng)
    yt = -theta[:,1:2]*yp0 + theta[:,0:1]*yq0
    mask=np.isfinite(yh[:,0]) & np.isfinite(yp0[:,0]) & np.isfinite(yq0[:,0])
    print(f'[flex-dataset] flattened samples={mask.sum()}')
    return xmu[mask],xreal[mask],theta[mask],yh[mask],yp0[mask],yq0[mask],ypg[mask],yqg[mask],yt[mask]

class BayesLinear(nn.Module):
    def __init__(self,i,o,prior_sigma=1.0,init_rho=-5.0):
        super().__init__(); self.prior_sigma=prior_sigma
        self.w_mu=nn.Parameter(torch.empty(o,i)); self.w_rho=nn.Parameter(torch.empty(o,i)); self.b_mu=nn.Parameter(torch.empty(o)); self.b_rho=nn.Parameter(torch.empty(o))
        nn.init.kaiming_uniform_(self.w_mu,a=math.sqrt(5)); nn.init.constant_(self.w_rho,init_rho); nn.init.uniform_(self.b_mu,-1/math.sqrt(i),1/math.sqrt(i)); nn.init.constant_(self.b_rho,init_rho)
    def _sigma(self,r): return torch.nn.functional.softplus(r)+1e-6
    def forward(self,x,sample=True):
        if sample: w=self.w_mu+self._sigma(self.w_rho)*torch.randn_like(self.w_mu); b=self.b_mu+self._sigma(self.b_rho)*torch.randn_like(self.b_mu)
        else: w,b=self.w_mu,self.b_mu
        return torch.nn.functional.linear(x,w,b)
    def kl_divergence(self):
        pv=self.prior_sigma**2; ws=self._sigma(self.w_rho); bs=self._sigma(self.b_rho)
        kl=(math.log(self.prior_sigma)-torch.log(ws)).sum()+0.5*((ws**2+self.w_mu**2)/pv).sum()-0.5*self.w_mu.numel()
        kl+=(math.log(self.prior_sigma)-torch.log(bs)).sum()+0.5*((bs**2+self.b_mu**2)/pv).sum()-0.5*self.b_mu.numel(); return kl

def gmm_log_prob(y,w,mu,s):
    y=y.view(-1,1); z=(y-mu)/(s+1e-12)
    lp=-0.5*z**2-torch.log(s+1e-12)-0.5*math.log(2*math.pi)
    return torch.logsumexp(torch.log(w.clamp(min=1e-12))+lp,dim=1,keepdim=True)

def gmm_cdf(z,w,mu,s):
    zz=np.asarray(z).reshape(-1,1)
    return np.sum(w.reshape(1,-1)*norm.cdf((zz-mu.reshape(1,-1))/(s.reshape(1,-1)+1e-12)),axis=1)

def gmm_quantile_torch(w,mu,s,taus,n_iter=50):
    B,Kmix=w.shape; tt=torch.tensor(taus,device=w.device,dtype=w.dtype).view(1,-1).repeat(B,1); Ktau=tt.shape[1]
    lo=(mu-8*s).min(dim=1,keepdim=True)[0].repeat(1,Ktau); hi=(mu+8*s).max(dim=1,keepdim=True)[0].repeat(1,Ktau)
    for _ in range(n_iter):
        md=(lo+hi)/2
        cdf=(w.unsqueeze(1)*(0.5*(1+torch.erf((md.unsqueeze(-1)-mu.unsqueeze(1))/(s.unsqueeze(1)+1e-12)/math.sqrt(2))))).sum(dim=2)
        go=cdf<tt; lo=torch.where(go,md,lo); hi=torch.where(go,hi,md)
    return (lo+hi)/2

class BayesFlexGMM2SupportNet(nn.Module):
    def __init__(self,in_dim,case,hidden=160,depth=3):
        super().__init__(); self.n_gen=len(case.gen_buses); self.x_dim=in_dim
        self.layers=nn.ModuleList(); d=in_dim+2
        for _ in range(depth): self.layers.append(BayesLinear(d,hidden,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO)); d=hidden
        self.gmm_out=BayesLinear(d,3*N_GMM_COMPONENTS,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO)
        self.rec_out=BayesLinear(d+in_dim+2+1,1+2*self.n_gen,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO)
        self.act=nn.ReLU(); self.register_buffer('pg_min_t',torch.tensor(case.pg_min).view(1,-1).float()); self.register_buffer('pg_max_t',torch.tensor(case.pg_max).view(1,-1).float()); self.register_buffer('qg_min_t',torch.tensor(case.qg_min).view(1,-1).float()); self.register_buffer('qg_max_t',torch.tensor(case.qg_max).view(1,-1).float())
    def encode_gmm(self,x_mu_theta,sample=True):
        h=x_mu_theta
        for l in self.layers: h=self.act(l(h,sample=sample))
        return h
    def gmm_head(self,h,sample=True):
        o=self.gmm_out(h,sample=sample)
        K=N_GMM_COMPONENTS
        w=torch.softmax(o[:,:K],dim=1); mu=o[:,K:2*K]; s=torch.nn.functional.softplus(o[:,2*K:3*K])+GMM_SIGMA_FLOOR
        return w,mu,s
    def forward_gmm(self,x_mu_norm,theta_feat,sample=True):
        h=self.encode_gmm(torch.cat([x_mu_norm,theta_feat],dim=1),sample=sample); return h,*self.gmm_head(h,sample=sample)
    def recover_boundary_dispatch_from_h_theta(self,h_mu_theta,x_real_norm,theta_feat,h_label,h_mean,h_std,t_mean,t_std,sample=True,bound_t_output=True):
        h_norm=(h_label-h_mean)/(h_std+1e-9)
        o=self.rec_out(torch.cat([h_mu_theta,x_real_norm,theta_feat,h_norm],dim=1),sample=sample)
        raw_t=o[:,0:1]; pgr=o[:,1:1+self.n_gen]; qgr=o[:,1+self.n_gen:1+2*self.n_gen]
        t_norm=5.0*torch.tanh(raw_t/5.0) if bound_t_output else raw_t
        t=t_mean+t_std*t_norm
        alpha=theta_feat[:,0:1]; beta=theta_feat[:,1:2]
        p0=alpha*h_label-beta*t; q0=beta*h_label+alpha*t
        pg=self.pg_min_t.to(o)+torch.sigmoid(pgr)*(self.pg_max_t.to(o)-self.pg_min_t.to(o)); qg=self.qg_min_t.to(o)+torch.sigmoid(qgr)*(self.qg_max_t.to(o)-self.qg_min_t.to(o))
        return p0,q0,pg,qg,t
    def kl_divergence(self): return sum(l.kl_divergence() for l in self.layers)+self.gmm_out.kl_divergence()+self.rec_out.kl_divergence()

def recover_flows_from_flex_dispatch_batch(case,x_real_raw,p0_hat,q0_hat,pg_hat,qg_hat):
    B=x_real_raw.shape[0]; nb=case.n_bus; nl=case.from_bus.size; n_pv=len(case.pv_buses)
    pd=x_real_raw[:,:nb]; pr_pv=x_real_raw[:,nb:nb+n_pv]; pr=torch.zeros((B,nb),device=x_real_raw.device,dtype=x_real_raw.dtype); pr[:,torch.tensor(case.pv_buses,device=x_real_raw.device)]=pr_pv
    qr=pr*math.tan(math.acos(case.pv_pf)); qd=pd*torch.tensor(np.divide(case.qd_base,np.maximum(case.pd_base,1e-6)),device=x_real_raw.device,dtype=x_real_raw.dtype).view(1,-1)
    pinj=pr-pd; qinj=qr-qd; gb=torch.tensor(case.gen_buses,device=x_real_raw.device)
    pinj[:,gb]+=pg_hat; qinj[:,gb]+=qg_hat
    P=torch.zeros((B,nl),device=x_real_raw.device); Q=torch.zeros((B,nl),device=x_real_raw.device); sp=pinj.clone(); sq=qinj.clone()
    for j in case.rev_topo_order:
        if j==case.root: continue
        l=int(case.parent_branch[j]); p=int(case.parent_bus[j]); P[:,l]=-sp[:,j]; Q[:,l]=-sq[:,j]; sp[:,p]+=sp[:,j]; sq[:,p]+=sq[:,j]
    V=torch.zeros((B,nb),device=x_real_raw.device); V[:,case.root]=1.0
    rt=torch.tensor(case.r,device=x_real_raw.device,dtype=x_real_raw.dtype); xt=torch.tensor(case.x,device=x_real_raw.device,dtype=x_real_raw.dtype)
    for j in case.topo_order:
        if j==case.root: continue
        l=int(case.parent_branch[j]); p=int(case.parent_bus[j]); V[:,j]=V[:,p]-2*(rt[l]*P[:,l]+xt[l]*Q[:,l])
    return {"P":P,"Q":Q,"V":V,"pinj":pinj,"qinj":qinj,"pg":pg_hat,"qg":qg_hat,"pr":pr}

def physics_loss_flex(case,x_real_raw,p0_hat,q0_hat,pg_hat,qg_hat,return_parts=False,p_scale=1.0,q_scale=1.0):
    rec=recover_flows_from_flex_dispatch_batch(case,x_real_raw,p0_hat,q0_hat,pg_hat,qg_hat); P,Q,V,pinj,qinj,pg,qg,pr=rec['P'],rec['Q'],rec['V'],rec['pinj'],rec['qinj'],rec['pg'],rec['qg'],rec['pr']; relu=torch.relu
    p_scale_t=p_scale+1e-9; q_scale_t=q_scale+1e-9
    root_out=case.out_branches[case.root]; raw_pcc_p=(P[:,root_out].sum(dim=1,keepdim=True)-p0_hat).pow(2).mean(); raw_pcc_q=(Q[:,root_out].sum(dim=1,keepdim=True)-q0_hat).pow(2).mean()
    raw_global_p=(p0_hat+pinj.sum(dim=1,keepdim=True)).pow(2).mean(); raw_global_q=(q0_hat+qinj.sum(dim=1,keepdim=True)).pow(2).mean()
    pccp=(((P[:,root_out].sum(dim=1,keepdim=True)-p0_hat)/p_scale_t)**2).mean(); pccq=(((Q[:,root_out].sum(dim=1,keepdim=True)-q0_hat)/q_scale_t)**2).mean()
    gp=(((p0_hat+pinj.sum(dim=1,keepdim=True))/p_scale_t)**2).mean(); gq=(((q0_hat+qinj.sum(dim=1,keepdim=True))/q_scale_t)**2).mean()
    v=((relu(case.vmin**2-V)**2)+(relu(V-case.vmax**2)**2)).mean(); fp=torch.tensor(case.fmax_p,device=x_real_raw.device).view(1,-1); fq=torch.tensor(case.fmax_q,device=x_real_raw.device).view(1,-1)
    lp=(relu(P.abs()-fp)**2).mean(); lq=(relu(Q.abs()-fq)**2).mean(); pgmn=torch.tensor(case.pg_min,device=x_real_raw.device).view(1,-1); pgmx=torch.tensor(case.pg_max,device=x_real_raw.device).view(1,-1); qgmn=torch.tensor(case.qg_min,device=x_real_raw.device).view(1,-1); qgmx=torch.tensor(case.qg_max,device=x_real_raw.device).view(1,-1)
    lpg=((relu(pgmn-pg)**2)+(relu(pg-pgmx)**2)).mean(); lqg=((relu(qgmn-qg)**2)+(relu(qg-qgmx)**2)).mean(); pv=(relu(pr[:,torch.tensor(case.pv_buses,device=x_real_raw.device)]-torch.tensor(case.pv_pmax,device=x_real_raw.device).view(1,-1))**2).mean()
    kcl=[]
    for i in range(case.n_bus):
        if i==case.root: continue
        kcl.append((P[:,case.in_branches[i]].sum(dim=1,keepdim=True)-P[:,case.out_branches[i]].sum(dim=1,keepdim=True)+pinj[:,i:i+1]).pow(2))
        kcl.append((Q[:,case.in_branches[i]].sum(dim=1,keepdim=True)-Q[:,case.out_branches[i]].sum(dim=1,keepdim=True)+qinj[:,i:i+1]).pow(2))
    lkcl=torch.cat(kcl,dim=1).mean(); loss=pccp+pccq+gp+gq+v+lp+lq+lpg+lqg+pv+0.1*lkcl
    if return_parts: return loss,{"pcc_p":pccp.detach(),"pcc_q":pccq.detach(),"global_p":gp.detach(),"global_q":gq.detach(),"raw_pcc_p":raw_pcc_p.detach(),"raw_pcc_q":raw_pcc_q.detach(),"raw_global_p":raw_global_p.detach(),"raw_global_q":raw_global_q.detach(),"voltage":v.detach(),"line_p":lp.detach(),"line_q":lq.detach(),"pg":lpg.detach(),"qg":lqg.detach(),"pv":pv.detach(),"kcl":lkcl.detach()}
    return loss

def precompute_h_empirical_quantiles(YH, taus):
    return np.quantile(YH, taus, axis=1).transpose(1,2,0)

def compute_h_quantile_loss(net,XMU_scen,THETA_FEAT,QH_EMP,x_mu_mean,x_mu_std,h_mean,h_std,taus=H_QUANTILE_TAUS,n_scenarios_sample=H_QUANTILE_BATCH_SCENARIOS,n_thetas_sample=H_QUANTILE_BATCH_THETAS,sample=True,rng=None,theta_sampling_weights=None):
    if XMU_scen.shape[0]==0: return torch.tensor(0.0,device=DEVICE)
    if rng is None: rng=np.random.default_rng()
    n_scen=XMU_scen.shape[0]; n_theta=THETA_FEAT.shape[0]
    scen_idx=rng.choice(n_scen,size=min(n_scenarios_sample,n_scen),replace=False)
    if n_thetas_sample is None:
        theta_idx=np.arange(n_theta)
    else:
        p=None if theta_sampling_weights is None else np.asarray(theta_sampling_weights,dtype=float)/np.sum(theta_sampling_weights)
        theta_idx=rng.choice(n_theta,size=min(n_thetas_sample,n_theta),replace=False,p=p)
    s_grid,t_grid=np.meshgrid(scen_idx,theta_idx,indexing='ij')
    s_flat=s_grid.reshape(-1); t_flat=t_grid.reshape(-1)
    xmu_q=((XMU_scen[s_flat]-x_mu_mean)/x_mu_std)
    theta_q=THETA_FEAT[t_flat]
    q_emp=QH_EMP[s_flat,t_flat,:]
    q_emp_norm=(q_emp-h_mean)/(h_std+1e-9)
    xmu_t=torch.tensor(xmu_q,dtype=torch.float32,device=DEVICE)
    th_t=torch.tensor(theta_q,dtype=torch.float32,device=DEVICE)
    q_emp_t=torch.tensor(q_emp_norm,dtype=torch.float32,device=DEVICE)
    with torch.set_grad_enabled(sample):
        _,w,mu,s=net.forward_gmm(xmu_t,th_t,sample=sample)
        q_pred=gmm_quantile_torch(w,mu,s,taus)
        return ((q_pred-q_emp_t)**2).mean()

def compute_h_cdf_loss(net,XMU_scen,THETA_FEAT,YH_scen,x_mu_mean,x_mu_std,h_mean,h_std,n_scenarios_sample=8,n_thetas_sample=4,sample=True,rng=None,theta_sampling_weights=None):
    if (not USE_H_CDF_LOSS) or XMU_scen.shape[0]==0: return torch.tensor(0.0,device=DEVICE)
    if rng is None: rng=np.random.default_rng()
    n_scen,n_theta=XMU_scen.shape[0],THETA_FEAT.shape[0]
    scen_idx=rng.choice(n_scen,size=min(n_scenarios_sample,n_scen),replace=False)
    if n_thetas_sample is None:
        theta_idx=np.arange(n_theta)
    else:
        p=None if theta_sampling_weights is None else np.asarray(theta_sampling_weights,dtype=float)/np.sum(theta_sampling_weights)
        theta_idx=rng.choice(n_theta,size=min(n_thetas_sample,n_theta),replace=False,p=p)
    s_grid,t_grid=np.meshgrid(scen_idx,theta_idx,indexing='ij'); s_flat=s_grid.reshape(-1); t_flat=t_grid.reshape(-1)
    xmu_t=torch.tensor((XMU_scen[s_flat]-x_mu_mean)/x_mu_std,dtype=torch.float32,device=DEVICE); th_t=torch.tensor(THETA_FEAT[t_flat],dtype=torch.float32,device=DEVICE)
    z_fix=np.array(CDF_STANDARD_GRID,dtype=float)
    hs=YH_scen[s_flat,:,t_flat]; hs_n=(hs-h_mean)/(h_std+1e-9)
    if USE_ADAPTIVE_CDF_GRID:
        z_emp=np.quantile(hs_n,np.array(CDF_GRID_TAUS),axis=1).T
        z_all=np.concatenate([z_emp,np.tile(z_fix.reshape(1,-1),(hs_n.shape[0],1))],axis=1)
    else:
        z_all=np.tile(z_fix.reshape(1,-1),(hs_n.shape[0],1))
    emp=np.mean(hs_n[:,:,None] <= z_all[:,None,:],axis=1)
    emp_t=torch.tensor(emp,dtype=torch.float32,device=DEVICE)
    with torch.set_grad_enabled(sample):
        _,w,mu,s=net.forward_gmm(xmu_t,th_t,sample=sample)
        z=torch.tensor(z_all,dtype=torch.float32,device=DEVICE).unsqueeze(-1)
        pred=(w.unsqueeze(1)*(0.5*(1+torch.erf((z-mu.unsqueeze(1))/(s.unsqueeze(1)+1e-12)/math.sqrt(2))))).sum(dim=2)
        return ((pred-emp_t)**2).mean()

def train_bayes_flex_gmm2(case,XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG):
    rng=np.random.default_rng(SEED_TRAIN); n_scen=XMU.shape[0]; n_val=max(1,int(0.1*n_scen)); idx=rng.permutation(n_scen); tr,va=idx[:-n_val],idx[-n_val:]
    def split(arr): return arr[tr],arr[va]
    XMU_tr,XMU_va=split(XMU); XREAL_tr,XREAL_va=split(XREAL); YH_tr,YH_va=split(YH); YP0_tr,YP0_va=split(YP0); YQ0_tr,YQ0_va=split(YQ0); YPG_tr,YPG_va=split(YPG); YQG_tr,YQG_va=split(YQG)
    xmu_mean=XMU_tr.mean(0,keepdims=True); xmu_std=XMU_tr.std(0,keepdims=True)+1e-9; xr_mean=XREAL_tr.reshape(-1,XREAL_tr.shape[-1]).mean(0,keepdims=True); xr_std=XREAL_tr.reshape(-1,XREAL_tr.shape[-1]).std(0,keepdims=True)+1e-9
    h_mean=np.array([[YH_tr.mean()]]); h_std=np.array([[YH_tr.std()+1e-9]]); p0_mean=np.array([[YP0_tr.mean()]]); p0_std=np.array([[YP0_tr.std()+1e-9]]); q0_mean=np.array([[YQ0_tr.mean()]]); q0_std=np.array([[YQ0_tr.std()+1e-9]])
    t_train = -THETA_FEAT[None,None,:,1]*YP0_tr + THETA_FEAT[None,None,:,0]*YQ0_tr
    t_mean=np.array([[t_train.mean()]]); t_std=np.array([[t_train.std()+1e-9]])
    QH_EMP_TR=precompute_h_empirical_quantiles(YH_tr,H_QUANTILE_TAUS); QH_EMP_VAL=precompute_h_empirical_quantiles(YH_va,H_QUANTILE_TAUS)
    trf=flatten_flex_dataset(XMU_tr,XREAL_tr,THETA_FEAT,YH_tr,YP0_tr,YQ0_tr,YPG_tr,YQG_tr); vaf=flatten_flex_dataset(XMU_va,XREAL_va,THETA_FEAT,YH_va,YP0_va,YQ0_va,YPG_va,YQG_va)
    xmu_f,xr_f,th_f,yh_f,yp0_f,yq0_f,ypg_f,yqg_f,yt_f=trf; xmu_v,xr_v,th_v,yh_v,yp0_v,yq0_v,ypg_v,yqg_v,yt_v=vaf
    to=lambda a:torch.tensor(a,dtype=torch.float32,device=DEVICE)
    xmu_t,xr_t,th_t,yh_t,yp0_t,yq0_t,ypg_t,yqg_t,yt_t=map(to,[ (xmu_f-xmu_mean)/xmu_std, (xr_f-xr_mean)/xr_std, th_f,yh_f,yp0_f,yq0_f,ypg_f,yqg_f,yt_f ])
    xr_raw_t=to(xr_f)
    xmu_tv,xr_tv,th_tv,yh_tv,yp0_tv,yq0_tv,ypg_tv,yqg_tv,yt_tv=map(to,[ (xmu_v-xmu_mean)/xmu_std, (xr_v-xr_mean)/xr_std, th_v,yh_v,yp0_v,yq0_v,ypg_v,yqg_v,yt_v ])
    xr_raw_v=to(xr_v)
    net=BayesFlexGMM2SupportNet(XMU.shape[1],case).to(DEVICE); opt=torch.optim.Adam(net.parameters(),lr=LR); n=xmu_t.shape[0]; nb=(n+BATCH_SIZE-1)//BATCH_SIZE
    hm,hs,p0m,p0s,q0m,q0s,tm,ts=to(h_mean),to(h_std),to(p0_mean),to(p0_std),to(q0_mean),to(q0_std),to(t_mean),to(t_std)
    print(f"[train-speed] DEVICE={DEVICE}")
    print(f"[train-speed] BATCH_SIZE={BATCH_SIZE}")
    print(f"[train-speed] h_quantile every N batches={QUANTILE_EVERY_N_BATCHES}")
    print(f"[train-speed] h_quantile starts at epoch={H_QUANTILE_START_EPOCH}")
    print(f"[train-speed] h_quantile sampled scenarios={H_QUANTILE_BATCH_SCENARIOS}")
    print(f"[train-speed] h_quantile sampled thetas={H_QUANTILE_BATCH_THETAS}")
    print('=== 开始训练 33-bus VI-BPINN Flex-Domain (support-function formulation) ===')
    best_state=None; best_score=np.inf; patience=25; no_improve=0
    theta_sampling_weights=np.ones(THETA_FEAT.shape[0],dtype=float)
    for ep in range(EPOCHS):
        if (ep+1)<=FAST_STAGE_A_EPOCHS: stage="A"
        elif (ep+1)<=FAST_STAGE_A_EPOCHS+FAST_STAGE_B_EPOCHS: stage="B"
        else: stage="C"
        cur_q_every = 5 if stage=="A" else 1
        cur_q_scen = 8 if stage=="A" else (32 if stage=="B" else min(48,max(1,XMU_tr.shape[0])))
        cur_q_theta = 4 if stage=="A" else None
        cur_lam_hq = 0.03 if stage=="A" else (0.15 if stage=="B" else 0.20)
        cur_lam_hcdf = 0.00 if stage=="A" else (0.15 if stage=="B" else 0.25)
        cur_lr = LR if stage=="A" else (LR*0.3 if stage=="B" else LR*0.15)
        reweight_enabled=USE_THETA_REWEIGHT and (ep+1)>=THETA_REWEIGHT_START_EPOCH and stage in ["B","C"]
        for g in opt.param_groups: g['lr']=cur_lr
        perm=rng.permutation(n); beta=BETA_KL_MAX*min(1.0,(ep+1)/KL_WARMUP_EPOCHS); acc=np.zeros(10)
        for b in range(nb):
            ii=perm[b*BATCH_SIZE:min((b+1)*BATCH_SIZE,n)]
            xmu, xr, xr_raw, th, yh, yp0, yq0, ypg, yqg, yt = xmu_t[ii], xr_t[ii], xr_raw_t[ii], th_t[ii], yh_t[ii], yp0_t[ii], yq0_t[ii], ypg_t[ii], yqg_t[ii], yt_t[ii]
            henc,w,mu,s=net.forward_gmm(xmu,th,sample=True); yh_n=(yh-hm)/(hs+1e-9); nll=(-gmm_log_prob(yh_n,w,mu,s)).mean()
            p0h,q0h,pgh,qgh,thh=net.recover_boundary_dispatch_from_h_theta(henc,xr,th,yh,hm,hs,tm,ts,sample=True)
            p0q0_sup=((p0h-yp0)/p0s).pow(2).mean()+((q0h-yq0)/q0s).pow(2).mean(); t_sup=((thh-yt)/(ts+1e-9)).pow(2).mean(); bsup=p0q0_sup+LAM_T_SUP*t_sup
            dsup=((pgh-ypg)/(net.pg_max_t-net.pg_min_t+1e-6)).pow(2).mean()+((qgh-yqg)/(net.qg_max_t-net.qg_min_t+1e-6)).pow(2).mean()
            phys=physics_loss_flex(case,xr_raw,p0h,q0h,pgh,qgh,p_scale=p0s,q_scale=q0s)
            h_hat=th[:,0:1]*p0h+th[:,1:2]*q0h; scons=(((h_hat-yh)/(hs+1e-9))**2).mean()
            hq=torch.tensor(0.0,device=DEVICE)
            if USE_H_QUANTILE_LOSS and (ep+1)>=H_QUANTILE_START_EPOCH and cur_lam_hq>0 and (b % cur_q_every == 0):
                hq_rng=np.random.default_rng(SEED_TRAIN + 100000*ep + b)
                hq = compute_h_quantile_loss(net,XMU_tr,THETA_FEAT,QH_EMP_TR,xmu_mean,xmu_std,h_mean,h_std,taus=H_QUANTILE_TAUS,n_scenarios_sample=cur_q_scen,n_thetas_sample=cur_q_theta,sample=True,rng=hq_rng,theta_sampling_weights=(theta_sampling_weights if reweight_enabled else None))
            hcdf = compute_h_cdf_loss(net,XMU_tr,THETA_FEAT,YH_tr,xmu_mean,xmu_std,h_mean,h_std,n_scenarios_sample=cur_q_scen,n_thetas_sample=cur_q_theta,sample=True,rng=np.random.default_rng(SEED_TRAIN+200000*ep+b),theta_sampling_weights=(theta_sampling_weights if reweight_enabled else None)) if cur_lam_hcdf>0 else torch.tensor(0.0,device=DEVICE)
            kl=net.kl_divergence(); loss=nll+LAM_BOUNDARY_SUP*bsup+LAM_DISPATCH_SUP*dsup+LAM_PHYS_FLEX*phys+LAM_SUPPORT_CONSIST*scons+cur_lam_hq*hq+cur_lam_hcdf*hcdf+beta*kl/n
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
            acc += np.array([float(loss.detach().cpu()),float(nll.detach().cpu()),float(bsup.detach().cpu()),float(p0q0_sup.detach().cpu()),float(t_sup.detach().cpu()),float(dsup.detach().cpu()),float(phys.detach().cpu()),float(scons.detach().cpu()),float(hq.detach().cpu()),float(hcdf.detach().cpu())])
        if ep==0 or (ep+1)%50==0 or ep+1==EPOCHS:
            with torch.no_grad():
                henc,w,mu,s=net.forward_gmm(xmu_tv,th_tv,sample=False); nllv=(-gmm_log_prob((yh_tv-hm)/(hs+1e-9),w,mu,s)).mean()
                p0h,q0h,pgh,qgh,thh=net.recover_boundary_dispatch_from_h_theta(henc,xr_tv,th_tv,yh_tv,hm,hs,tm,ts,sample=False)
                p0q0sv=((p0h-yp0_tv)/p0s).pow(2).mean()+((q0h-yq0_tv)/q0s).pow(2).mean(); tsv=((thh-yt_tv)/(ts+1e-9)).pow(2).mean(); bsv=p0q0sv+LAM_T_SUP*tsv; dsv=((pgh-ypg_tv)/(net.pg_max_t-net.pg_min_t+1e-6)).pow(2).mean()+((qgh-yqg_tv)/(net.qg_max_t-net.qg_min_t+1e-6)).pow(2).mean()
                phv,parts=physics_loss_flex(case,xr_raw_v,p0h,q0h,pgh,qgh,return_parts=True,p_scale=p0s,q_scale=q0s); h_hat_v=th_tv[:,0:1]*p0h+th_tv[:,1:2]*q0h; scv=(((h_hat_v-yh_tv)/(hs+1e-9))**2).mean(); val_hq_rng=np.random.default_rng(SEED_TRAIN + 999); hqv=compute_h_quantile_loss(net,XMU_va,THETA_FEAT,QH_EMP_VAL,xmu_mean,xmu_std,h_mean,h_std,taus=H_QUANTILE_TAUS,n_scenarios_sample=min(cur_q_scen,max(1,XMU_va.shape[0])),n_thetas_sample=cur_q_theta,sample=False,rng=val_hq_rng) if (USE_H_QUANTILE_LOSS and cur_lam_hq>0) else torch.tensor(0.0,device=DEVICE); hcdfv=compute_h_cdf_loss(net,XMU_va,THETA_FEAT,YH_va,xmu_mean,xmu_std,h_mean,h_std,n_scenarios_sample=min(cur_q_scen,max(1,XMU_va.shape[0])),n_thetas_sample=cur_q_theta,sample=False,rng=np.random.default_rng(SEED_TRAIN+9999)) if cur_lam_hcdf>0 else torch.tensor(0.0,device=DEVICE)
                th_cdf=[]
                for j in range(THETA_FEAT.shape[0]):
                    th_cdf.append(float(compute_h_cdf_loss(net,XMU_va,THETA_FEAT,YH_va,xmu_mean,xmu_std,h_mean,h_std,n_scenarios_sample=min(16,max(1,XMU_va.shape[0])),n_thetas_sample=1,sample=False,rng=np.random.default_rng(SEED_TRAIN+7000+j),theta_sampling_weights=np.eye(THETA_FEAT.shape[0])[j])))
                th_cdf=np.array(th_cdf)
                if reweight_enabled and ((ep+1)%THETA_REWEIGHT_UPDATE_EVERY==0):
                    w=(th_cdf+1e-9)**THETA_REWEIGHT_POWER; w=w/(w.mean()+1e-9); w=np.clip(w,THETA_REWEIGHT_MIN,THETA_REWEIGHT_MAX); theta_sampling_weights=w
                    print(f"[theta-reweight] weights = {np.round(theta_sampling_weights,3).tolist()}")
                    print(f"[theta-reweight] worst_theta_idx = {int(np.argmax(theta_sampling_weights))}")
                    print(f"[theta-reweight] max_weight = {float(np.max(theta_sampling_weights)):.3f}")
                lv=nllv+LAM_BOUNDARY_SUP*bsv+LAM_DISPATCH_SUP*dsv+LAM_PHYS_FLEX*phv+LAM_SUPPORT_CONSIST*scv+cur_lam_hq*hqv+cur_lam_hcdf*hcdfv+beta*net.kl_divergence()/n
            print(f"Epoch {ep+1:4d} | stage={stage} | theta_reweight_enabled={reweight_enabled} | lr={cur_lr:.2e} | gmmK={N_GMM_COMPONENTS} | current_LAM_H_QUANTILE={cur_lam_hq:.3f} current_LAM_H_CDF={cur_lam_hcdf:.3f} | avg_loss={acc[0]/nb:.6f} avg_nll_h={acc[1]/nb:.6f} avg_boundary_sup={acc[2]/nb:.6f} avg_t_sup={acc[4]/nb:.6f} avg_phys_flex={acc[6]/nb:.6f} avg_h_quantile={acc[8]/nb:.6f} avg_h_cdf={acc[9]/nb:.6f} | val_h_quantile={hqv.item():.6f} val_h_cdf={hcdfv.item():.6f} val_boundary_sup={bsv.item():.6f} val_phys_flex={phv.item():.6f} val_raw_pcc_p={parts['raw_pcc_p'].item():.3e} val_raw_pcc_q={parts['raw_pcc_q'].item():.3e}")
            if (ep+1)%50==0:
                print(f"[theta-val] mean_h_cdf_by_theta = {np.round(th_cdf,4).tolist()}")
            print(f"[theta-val] max_theta_cdf = {float(np.max(th_cdf)):.6f}")
            print(f"[theta-val] worst_theta_idx = {int(np.argmax(th_cdf))}")
            vscore=float(bsv.item()+phv.item()+hqv.item())
            if vscore < best_score - 1e-6:
                best_score=vscore; no_improve=0; best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"[early-stop] no improvement for {patience} eval checkpoints, stop at epoch={ep+1}, best_score={best_score:.6f}")
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    norm={'x_mu_mean':xmu_mean,'x_mu_std':xmu_std,'x_real_mean':xr_mean,'x_real_std':xr_std,'h_mean':h_mean,'h_std':h_std,'p0_mean':p0_mean,'p0_std':p0_std,'q0_mean':q0_mean,'q0_std':q0_std,'t_mean':t_mean,'t_std':t_std}
    return net,norm

def flex_opf_sanity_check(case,n_samples=3,seed=2026):
    rng=np.random.default_rng(seed); print('\n=== flex OPF sanity check ===')
    for s in range(n_samples):
        pd,qd,pr,qr=sample_scenario_means(case,rng)
        for th in [0,math.pi/2,math.pi,3*math.pi/2]:
            sol=solve_flex_support_gurobi_33bus(case,pd,qd,pr,qr,math.cos(th),math.sin(th),return_detail=True,stabilize_dispatch=STABILIZE_OPF_DISPATCH)
            if not sol['ok']: print('infeasible'); continue
            h=math.cos(th)*sol['P0']+math.sin(th)*sol['Q0']; pccp=sol['P'][case.out_branches[case.root]].sum(); pccq=sol['Q'][case.out_branches[case.root]].sum()
            print(f"[flex-opf] s{s} th={th:.2f} h={sol['h']:.4f} check={h:.4f} pccPerr={pccp-sol['P0']:+.2e} pccQerr={pccq-sol['Q0']:+.2e} Vmin={sol['V'].min():.4f}")

def flex_realization_sanity_check(case,net,norm,theta_list,n_samples=3,seed=2030):
    rng=np.random.default_rng(seed); hm=torch.tensor(norm['h_mean'],dtype=torch.float32,device=DEVICE); hs=torch.tensor(norm['h_std'],dtype=torch.float32,device=DEVICE)
    for s in range(n_samples):
        pd,qd,pr,qr=sample_scenario_means(case,rng); xmu=make_feature_vector(case,pd,pr).reshape(1,-1)
        xmu_n=(xmu-norm['x_mu_mean'])/norm['x_mu_std']
        for th in [0,math.pi/2,math.pi,3*math.pi/2]:
            alpha,beta=math.cos(th),math.sin(th); sol=solve_flex_support_gurobi_33bus(case,pd,qd,pr,qr,alpha,beta,return_detail=True)
            if not sol['ok']: continue
            xr=make_feature_vector(case,pd,pr).reshape(1,-1); xr_n=(xr-norm['x_real_mean'])/norm['x_real_std']
            with torch.no_grad():
                xmu_t=torch.tensor(xmu_n,dtype=torch.float32,device=DEVICE); th_t=torch.tensor([[alpha,beta]],dtype=torch.float32,device=DEVICE); yh_t=torch.tensor([[sol['h']]],dtype=torch.float32,device=DEVICE)
                henc,*_=net.forward_gmm(xmu_t,th_t,sample=False)
                p0s=torch.tensor(norm['p0_std'],dtype=torch.float32,device=DEVICE); q0s=torch.tensor(norm['q0_std'],dtype=torch.float32,device=DEVICE); tm=torch.tensor(norm['t_mean'],dtype=torch.float32,device=DEVICE); ts=torch.tensor(norm['t_std'],dtype=torch.float32,device=DEVICE)
                p0h,q0h,pgh,qgh,thh=net.recover_boundary_dispatch_from_h_theta(henc,torch.tensor(xr_n,dtype=torch.float32,device=DEVICE),th_t,yh_t,hm,hs,tm,ts,sample=False)
                rec=physics_loss_flex(case,torch.tensor(xr,dtype=torch.float32,device=DEVICE),p0h,q0h,pgh,qgh,return_parts=True,p_scale=p0s,q_scale=q0s)[1]
            t_label=-beta*sol['P0']+alpha*sol['Q0']; t_hat=-beta*float(p0h)+alpha*float(q0h)
            print(f"[flex-real] th={th:.2f} P0_label={sol['P0']:.4f} P0_hat={float(p0h):.4f} P0_err={float(p0h)-sol['P0']:+.2e} Q0_label={sol['Q0']:.4f} Q0_hat={float(q0h):.4f} Q0_err={float(q0h)-sol['Q0']:+.2e} t_label={t_label:.4f} t_hat={t_hat:.4f} t_err={t_hat-t_label:+.2e} support_res_raw={(alpha*float(p0h)+ beta*float(q0h)-sol['h']):+.2e} support_res_norm={((alpha*float(p0h)+beta*float(q0h)-sol['h'])/(float(norm['h_std'][0,0])+1e-9)):+.2e} pcc_p={rec['pcc_p'].item():.2e} pcc_q={rec['pcc_q'].item():.2e} global_p={rec['global_p'].item():.2e} global_q={rec['global_q'].item():.2e} raw_pcc_p={rec['raw_pcc_p'].item():.2e} raw_pcc_q={rec['raw_pcc_q'].item():.2e} L1_Pg={np.abs(pgh.cpu().numpy().reshape(-1)-sol['Pg']).sum():.4f} L1_Qg={np.abs(qgh.cpu().numpy().reshape(-1)-sol['Qg']).sum():.4f}")

def eval_and_plot_direction_cdfs(case,net,norm,theta_list):
    reps=[0,np.pi/2,np.pi,3*np.pi/2]; idx=[int(np.argmin(np.abs(((theta_list-r+np.pi)%(2*np.pi))-np.pi))) for r in reps]
    fig,axs=plt.subplots(2,2,figsize=(10,8),dpi=120)
    for ax,j in zip(axs.ravel(),idx):
        th=theta_list[j]; alpha,beta=np.cos(th),np.sin(th); rng=np.random.default_rng(SEED_EVAL)
        pdm,qdm,prm,qrm=sample_scenario_means(case,rng); ys=[]
        for _ in range(500):
            pd=pdm.copy(); pr=prm.copy(); std_pd=0.10*np.maximum(pdm,1e-3); std_pr=0.12*np.maximum(prm,1e-3)
            for i in range(case.n_bus):
                if pdm[i]>1e-9: pd[i]=sample_trunc_normal(pdm[i],std_pd[i],0,None)
            for k,b in enumerate(case.pv_buses): pr[b]=sample_trunc_normal(prm[b],std_pr[b],0,float(case.pv_pmax[k]))
            qd=qdm*(pd/np.maximum(pdm,1e-6)); qr=pr*math.tan(math.acos(case.pv_pf)); v=solve_flex_support_gurobi_33bus(case,pd,qd,pr,qr,alpha,beta)
            if np.isfinite(v): ys.append(v)
        if len(ys) < 30:
            print(f"[direction-cdf] warning: theta={th:.3f}, feasible samples={len(ys)} < 30, skip this subplot")
            ax.set_title(f"theta={th:.2f}: insufficient MC samples")
            ax.axis("off")
            continue
        ys=np.sort(np.array(ys)); z=np.linspace(ys.min()-0.2,ys.max()+0.2,400); cdf_mc=np.searchsorted(ys,z,side='right')/len(ys)
        xmu=make_feature_vector(case,pdm,prm).reshape(1,-1); xmu_n=(xmu-norm['x_mu_mean'])/norm['x_mu_std']; xt=torch.tensor(xmu_n,dtype=torch.float32,device=DEVICE); th_t=torch.tensor([[alpha,beta]],dtype=torch.float32,device=DEVICE)
        cdfs=[]
        with torch.no_grad():
            for _ in range(EVAL_THETA_SAMPLES):
                _,w,mu,s=net.forward_gmm(xt,th_t,sample=True)
                z_norm=(z-float(norm['h_mean'][0,0]))/(float(norm['h_std'][0,0])+1e-9)
                cdfs.append(gmm_cdf(z_norm,w.cpu().numpy().reshape(-1),mu.cpu().numpy().reshape(-1),s.cpu().numpy().reshape(-1)))
        cdfs=np.array(cdfs); cm=cdfs.mean(0); clo=np.quantile(cdfs,0.025,0); chi=np.quantile(cdfs,0.975,0); arms=100*math.sqrt(np.mean((cm-cdf_mc)**2))
        ax.fill_between(z,clo,chi,alpha=0.2); ax.plot(z,cdf_mc,'k',lw=2); ax.plot(z,cm,'--',lw=2); ax.set_title(f'theta={th:.2f}, ARMS={arms:.2f}%')
    plt.tight_layout(); plt.savefig('FlexDomain_CDF_selected_directions_v2.png',dpi=260)

def eval_and_plot_flex_domain(case,net,norm,theta_list,mc_eval=500,seed=SEED_EVAL):
    rng=np.random.default_rng(seed); pdm,qdm,prm,qrm=sample_scenario_means(case,rng); xmu=make_feature_vector(case,pdm,prm).reshape(1,-1)
    h_mc_q05=[]; h_mc_q50=[]; h_mc_q95=[]; h_bn_q05=[]; h_bn_q50=[]; h_bn_q95=[]
    for th in theta_list:
        alpha,beta=np.cos(th),np.sin(th); ys=[]
        for _ in range(mc_eval):
            pd=pdm.copy(); pr=prm.copy(); std_pd=0.10*np.maximum(pdm,1e-3); std_pr=0.12*np.maximum(prm,1e-3)
            for i in range(case.n_bus):
                if pdm[i]>1e-9: pd[i]=sample_trunc_normal(pdm[i],std_pd[i],0,None)
            for k,b in enumerate(case.pv_buses): pr[b]=sample_trunc_normal(prm[b],std_pr[b],0,float(case.pv_pmax[k]))
            qd=qdm*(pd/np.maximum(pdm,1e-6)); qr=pr*math.tan(math.acos(case.pv_pf)); v=solve_flex_support_gurobi_33bus(case,pd,qd,pr,qr,alpha,beta)
            if np.isfinite(v): ys.append(v)
        if len(ys)<30:
            print(f"[flex-eval] warning: theta={th:.3f} has too few MC samples={len(ys)}")
            h_mc_q05.append(np.nan); h_mc_q50.append(np.nan); h_mc_q95.append(np.nan)
        else:
            ys=np.array(ys); h_mc_q05.append(np.quantile(ys,0.05)); h_mc_q50.append(np.quantile(ys,0.50)); h_mc_q95.append(np.quantile(ys,0.95))
        xmu_n=(xmu-norm['x_mu_mean'])/norm['x_mu_std']; xt=torch.tensor(xmu_n,dtype=torch.float32,device=DEVICE); th_t=torch.tensor([[alpha,beta]],dtype=torch.float32,device=DEVICE)
        with torch.no_grad():
            _,w,mu,s=net.forward_gmm(xt,th_t,sample=False)
            qn=gmm_quantile_torch(w,mu,s,[0.05,0.5,0.95]).cpu().numpy().reshape(-1)
            q=float(norm['h_mean'][0,0])+float(norm['h_std'][0,0])*qn
            h_bn_q05.append(q[0]); h_bn_q50.append(q[1]); h_bn_q95.append(q[2])
    poly_mc50=support_values_to_polygon(theta_list,np.array(h_mc_q50)); poly_bn50=support_values_to_polygon(theta_list,np.array(h_bn_q50)); poly_mc05=support_values_to_polygon(theta_list,np.array(h_mc_q05)); poly_mc95=support_values_to_polygon(theta_list,np.array(h_mc_q95)); poly_bn05=support_values_to_polygon(theta_list,np.array(h_bn_q05)); poly_bn95=support_values_to_polygon(theta_list,np.array(h_bn_q95))
    for nm,poly in [("mc50",poly_mc50),("bn50",poly_bn50),("mc05",poly_mc05),("mc95",poly_mc95),("bn05",poly_bn05),("bn95",poly_bn95)]:
        if len(poly)>=3:
            ar=polygon_area(poly)
            if ar<1e-6 or ar>1e3: print(f"[flex-eval] warning: polygon {nm} area abnormal {ar:.4e}")
    plt.figure(figsize=(10,8),dpi=260)
    def close(poly): return (np.r_[poly[:,0],poly[0,0]],np.r_[poly[:,1],poly[0,1]])
    if len(poly_mc95)>2: x,y=close(poly_mc95); plt.fill(x,y,color='#bfdbfe',alpha=0.18,label='MC 95% quantile domain')
    if len(poly_mc05)>2: x,y=close(poly_mc05); plt.fill(x,y,color='#60a5fa',alpha=0.30,label='MC 5% quantile domain')
    if len(poly_bn95)>2: x,y=close(poly_bn95); plt.fill(x,y,color='#fed7aa',alpha=0.18,hatch='//',edgecolor='#fb923c',label='B-PINN 95% quantile domain')
    if len(poly_bn05)>2: x,y=close(poly_bn05); plt.fill(x,y,color='#fdba74',alpha=0.30,hatch='//',edgecolor='#f97316',label='B-PINN 5% quantile domain')
    if len(poly_mc50)>2: x,y=close(poly_mc50); plt.plot(x,y,'-',color='#1d4ed8',lw=3.0,label='MC median domain')
    if len(poly_bn50)>2: x,y=close(poly_bn50); plt.plot(x,y,'--',color='#c2410c',lw=3.0,label='B-PINN median domain')
    if len(poly_mc50)>2 and len(poly_bn50)>2 and len(poly_mc50)==len(poly_bn50):
        d=np.sqrt(((poly_mc50-poly_bn50)**2).sum(1))
        print(f"[flex-domain-metrics] median mean vertex dist = {d.mean():.4f} MW/Mvar")
        print(f"[flex-domain-metrics] median max vertex dist = {d.max():.4f}")
    def area_err(a,b): return abs(a-b)/(abs(a)+1e-9)*100.0
    amc50,abp50,amc05,abp05,amc95,abp95=polygon_area(poly_mc50),polygon_area(poly_bn50),polygon_area(poly_mc05),polygon_area(poly_bn05),polygon_area(poly_mc95),polygon_area(poly_bn95)
    print(f"[flex-domain-metrics] median area err = {area_err(amc50,abp50):.3f} %")
    print(f"[flex-domain-metrics] q05 area err = {area_err(amc05,abp05):.3f} %")
    print(f"[flex-domain-metrics] q95 area err = {area_err(amc95,abp95):.3f} %")
    plt.grid(alpha=0.25,color='0.7'); plt.xlabel('P0 (MW)',fontsize=12); plt.ylabel('Q0 (Mvar)',fontsize=12); plt.title('MC vs B-PINN quantile flexibility domains'); plt.legend(loc='center left',bbox_to_anchor=(1.02,0.5),fontsize=10); plt.axis('equal'); plt.margins(0.08); plt.tight_layout(); plt.savefig('FlexDomain_overlay_quantile_comparison_v2.png',dpi=280)
    if len(poly_mc95)>2:
        plt.figure(figsize=(9,7),dpi=260); x,y=close(poly_mc95); plt.fill(x,y,color='#bfdbfe',alpha=0.18,label='MC 95% quantile domain')
        if len(poly_mc05)>2: x,y=close(poly_mc05); plt.fill(x,y,color='#60a5fa',alpha=0.30,label='MC 5% quantile domain')
        if len(poly_mc50)>2: x,y=close(poly_mc50); plt.plot(x,y,color='#1d4ed8',lw=3,label='MC median domain')
        plt.grid(alpha=0.25); plt.title('MC quantile flexibility domains under source-load uncertainty'); plt.xlabel('P0 (MW)',fontsize=12); plt.ylabel('Q0 (Mvar)',fontsize=12); plt.axis('equal'); plt.legend(); plt.tight_layout(); plt.savefig('FlexDomain_MC_quantile_domains_v2.png',dpi=280)
    if len(poly_bn95)>2:
        plt.figure(figsize=(9,7),dpi=260); x,y=close(poly_bn95); plt.fill(x,y,color='#fed7aa',alpha=0.18,hatch='//',edgecolor='#fb923c',label='B-PINN 95% quantile domain')
        if len(poly_bn05)>2: x,y=close(poly_bn05); plt.fill(x,y,color='#fdba74',alpha=0.30,hatch='//',edgecolor='#f97316',label='B-PINN 5% quantile domain')
        if len(poly_bn50)>2: x,y=close(poly_bn50); plt.plot(x,y,'--',color='#c2410c',lw=3,label='B-PINN median domain')
        plt.grid(alpha=0.25); plt.title('B-PINN predicted quantile flexibility domains'); plt.xlabel('P0 (MW)',fontsize=12); plt.ylabel('Q0 (Mvar)',fontsize=12); plt.axis('equal'); plt.legend(); plt.tight_layout(); plt.savefig('FlexDomain_BPINN_quantile_domains_v2.png',dpi=280)

def eval_multiple_flex_scenarios(case,net,norm,theta_list):
    rng=np.random.default_rng(SEED_EVAL+1000); all_arms=[]
    for s in range(N_TEST_SCENARIOS):
        pdm,qdm,prm,qrm=sample_scenario_means(case,rng); xmu=make_feature_vector(case,pdm,prm).reshape(1,-1)
        for th in theta_list:
            alpha,beta=np.cos(th),np.sin(th); ys=[]
            for _ in range(MC_EVAL_MULTI):
                pd=pdm.copy(); pr=prm.copy(); std_pd=0.10*np.maximum(pdm,1e-3); std_pr=0.12*np.maximum(prm,1e-3)
                for i in range(case.n_bus):
                    if pdm[i]>1e-9: pd[i]=sample_trunc_normal(pdm[i],std_pd[i],0,None)
                for k,b in enumerate(case.pv_buses): pr[b]=sample_trunc_normal(prm[b],std_pr[b],0,float(case.pv_pmax[k]))
                qd=qdm*(pd/np.maximum(pdm,1e-6)); qr=pr*math.tan(math.acos(case.pv_pf)); v=solve_flex_support_gurobi_33bus(case,pd,qd,pr,qr,alpha,beta)
                if np.isfinite(v): ys.append(v)
            if len(ys)<30: continue
            ys=np.sort(np.array(ys)); z=np.linspace(ys.min()-0.2,ys.max()+0.2,300); cdf_mc=np.searchsorted(ys,z,side='right')/len(ys)
            xt=torch.tensor((xmu-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE); th_t=torch.tensor([[alpha,beta]],dtype=torch.float32,device=DEVICE)
            with torch.no_grad():
                c=[]
                for _ in range(EVAL_THETA_SAMPLES):
                    _,w,mu,s=net.forward_gmm(xt,th_t,sample=True); z_norm=(z-float(norm['h_mean'][0,0]))/(float(norm['h_std'][0,0])+1e-9); c.append(gmm_cdf(z_norm,w.cpu().numpy().reshape(-1),mu.cpu().numpy().reshape(-1),s.cpu().numpy().reshape(-1)))
            arms=100*math.sqrt(np.mean((np.mean(np.array(c),axis=0)-cdf_mc)**2)); all_arms.append((s,th,arms))
    if len(all_arms) == 0:
        print("[multi-flex] warning: no valid ARMS samples, skip summary")
        return
    arr=np.array([a[2] for a in all_arms]); print(f'[multi-flex] overall mean={arr.mean():.4f}% median={np.median(arr):.4f}% q90={np.quantile(arr,0.9):.4f}% max={arr.max():.4f}%')
    worst=max(all_arms,key=lambda x:x[2]); print(f"[multi-flex] worst scenario={worst[0]}, theta={worst[1]:.4f}, ARMS={worst[2]:.4f}%")

def eval_all_theta_cdf_arms(case,net,norm,theta_list,mc_eval=400,save_cache_path='all_theta_eval_cache.npz'):
    rng=np.random.default_rng(SEED_EVAL+222); pdm,qdm,prm,qrm=sample_scenario_means(case,rng); xmu=make_feature_vector(case,pdm,prm).reshape(1,-1)
    rows=[]
    YH_by_theta={}
    for j,th in enumerate(theta_list):
        alpha,beta=np.cos(th),np.sin(th); ys=[]
        for _ in range(mc_eval):
            pd=pdm.copy(); pr=prm.copy()
            for i in range(case.n_bus):
                if pdm[i]>1e-9: pd[i]=sample_trunc_normal(pdm[i],0.10*max(pdm[i],1e-3),0,None)
            for k,b in enumerate(case.pv_buses): pr[b]=sample_trunc_normal(prm[b],0.12*max(prm[b],1e-3),0,float(case.pv_pmax[k]))
            qd=qdm*(pd/np.maximum(pdm,1e-6)); qr=pr*math.tan(math.acos(case.pv_pf)); v=solve_flex_support_gurobi_33bus(case,pd,qd,pr,qr,alpha,beta)
            if np.isfinite(v): ys.append(v)
        if len(ys)<30: continue
        YH_by_theta[j]=np.asarray(ys,dtype=float)
        ys=np.sort(np.array(ys)); z=np.linspace(ys.min()-0.2,ys.max()+0.2,250); cdf_mc=np.searchsorted(ys,z,side='right')/len(ys)
        xt=torch.tensor((xmu-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE); th_t=torch.tensor([[alpha,beta]],dtype=torch.float32,device=DEVICE)
        c=[]
        with torch.no_grad():
            for _ in range(EVAL_THETA_SAMPLES):
                _,w,mu,s=net.forward_gmm(xt,th_t,sample=True); zn=(z-float(norm['h_mean'][0,0]))/(float(norm['h_std'][0,0])+1e-9); c.append(gmm_cdf(zn,w.cpu().numpy().reshape(-1),mu.cpu().numpy().reshape(-1),s.cpu().numpy().reshape(-1)))
        arms=100*math.sqrt(np.mean((np.mean(np.array(c),axis=0)-cdf_mc)**2)); rows.append((j,float(th),float(arms)))
    if not rows: return None
    arr=np.array([r[2] for r in rows]); worst=max(rows,key=lambda x:x[2]); print(f"[all-theta-arms] mean={arr.mean():.4f}%"); print(f"[all-theta-arms] median={np.median(arr):.4f}%"); print(f"[all-theta-arms] max={arr.max():.4f}%"); print(f"[all-theta-arms] q90={np.quantile(arr,0.9):.4f}%"); print(f"[all-theta-arms] worst3={[w[0] for w in sorted(rows,key=lambda x:x[2],reverse=True)[:3]]}")
    reps={0,np.pi/2,np.pi,3*np.pi/2}
    with open('all_theta_cdf_arms_v1.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['theta_idx','theta','arms_pct','is_representative_direction']); [w.writerow([r[0],r[1],r[2],int(any(abs(r[1]-rp)<1e-6 for rp in reps))]) for r in rows]
    np.savez(save_cache_path,
             XMU_eval=xmu,
             theta_list=np.asarray(theta_list,dtype=float),
             arms_by_theta=np.asarray([r[2] for r in rows],dtype=float),
             theta_idx_by_row=np.asarray([r[0] for r in rows],dtype=int),
             **{f'YH_theta_{j}':YH_by_theta[j] for j in YH_by_theta})
    print(f"[all-theta-arms] eval cache saved to {save_cache_path}")
    colors=['#6b7280']*len(rows)
    worst3=sorted(rows,key=lambda x:x[2],reverse=True)[:3]
    for w3 in worst3: colors[w3[0]]='#dc2626'
    plt.figure(figsize=(9,4.2),dpi=260); plt.bar([r[0] for r in rows],[r[2] for r in rows],color=colors); plt.axhline(arr.mean(),color='#1d4ed8',ls='--',lw=1.5,label='mean'); plt.axhline(arr.max(),color='#ea580c',ls=':',lw=1.8,label='max'); plt.xlabel('theta idx',fontsize=12); plt.ylabel('ARMS (%)',fontsize=12); plt.title('All-theta CDF ARMS of support-function distribution'); plt.legend(); plt.tight_layout(); plt.savefig('FlexDomain_all_theta_ARMS_bar_v2.png',dpi=280)
    return {'rows':rows,'xmu':xmu,'cache':save_cache_path}

def eval_and_plot_flex_domain_posterior(case,net,norm,theta_list,n_post=30,mc_eval=200):
    rng=np.random.default_rng(SEED_EVAL+333); pdm,qdm,prm,qrm=sample_scenario_means(case,rng); xmu=make_feature_vector(case,pdm,prm).reshape(1,-1)
    polys=[]
    for _ in range(n_post):
        hs=[]
        for th in theta_list:
            xt=torch.tensor((xmu-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE); th_t=torch.tensor([[np.cos(th),np.sin(th)]],dtype=torch.float32,device=DEVICE)
            with torch.no_grad():
                _,w,mu,s=net.forward_gmm(xt,th_t,sample=True); qn=gmm_quantile_torch(w,mu,s,[0.5]).cpu().numpy().reshape(-1)[0]; hs.append(float(norm['h_mean'][0,0])+float(norm['h_std'][0,0])*qn)
        polys.append(support_values_to_polygon(theta_list,np.array(hs)))
    plt.figure(figsize=(8,7),dpi=140)
    for p in polys:
        if len(p)>2: plt.plot(np.r_[p[:,0],p[0,0]],np.r_[p[:,1],p[0,1]],color='#14b8a6',alpha=0.15,lw=0.8)
    if polys and len(polys[0])>2: plt.plot(np.r_[polys[0][:,0],polys[0][0,0]],np.r_[polys[0][:,1],polys[0][0,1]],color='#0f766e',lw=2.0,label='posterior sample median domain')
    pm=np.nanmean(np.array([np.interp(np.arange(len(theta_list)),np.arange(len(theta_list)),np.array([np.nan if len(pp)<3 else 0 for pp in polys]))]),axis=0) if False else None
    hs_det=[]; hs_mc=[]
    pdm,qdm,prm,qrm=sample_scenario_means(case,np.random.default_rng(SEED_EVAL+335))
    for th in theta_list:
        a,b=np.cos(th),np.sin(th)
        hs_mc.append(solve_flex_support_gurobi_33bus(case,pdm,qdm,prm,qrm,float(a),float(b)))
        xt=torch.tensor((xmu-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE); th_t=torch.tensor([[a,b]],dtype=torch.float32,device=DEVICE)
        with torch.no_grad():
            _,w,mu,s=net.forward_gmm(xt,th_t,sample=False); qn=gmm_quantile_torch(w,mu,s,[0.5]).cpu().numpy().reshape(-1)[0]; hs_det.append(float(norm['h_mean'][0,0])+float(norm['h_std'][0,0])*qn)
    poly_mc=support_values_to_polygon(theta_list,np.array(hs_mc)); poly_det=support_values_to_polygon(theta_list,np.array(hs_det))
    if len(poly_mc)>2: plt.plot(np.r_[poly_mc[:,0],poly_mc[0,0]],np.r_[poly_mc[:,1],poly_mc[0,1]],color='#1d4ed8',lw=2.6,label='MC median domain')
    if len(poly_det)>2: plt.plot(np.r_[poly_det[:,0],poly_det[0,0]],np.r_[poly_det[:,1],poly_det[0,1]],'--',color='#c2410c',lw=2.6,label='deterministic B-PINN median domain')
    if polys:
        hs_mean=np.mean([np.array([p[i,0]*np.cos(theta_list[i])+p[i,1]*np.sin(theta_list[i]) for i in range(min(len(theta_list),len(p)))]) if len(p)>=len(theta_list) else np.zeros(len(theta_list)) for p in polys],axis=0)
        poly_pm=support_values_to_polygon(theta_list,hs_mean)
        if len(poly_pm)>2: plt.plot(np.r_[poly_pm[:,0],poly_pm[0,0]],np.r_[poly_pm[:,1],poly_pm[0,1]],color='#0f766e',lw=3,label='posterior mean median domain')
    plt.title('B-PINN posterior uncertainty of median flexibility domain'); plt.xlabel('P0 (MW)'); plt.ylabel('Q0 (Mvar)'); plt.grid(alpha=0.25); plt.legend(); plt.axis('equal'); plt.tight_layout(); plt.savefig('FlexDomain_posterior_overlay_v2.png',dpi=280)

def eval_and_plot_realization_cloud(case,net,norm,theta_list,n_real=24):
    rng=np.random.default_rng(SEED_EVAL+444); pdm,qdm,prm,qrm=sample_scenario_means(case,rng); clouds=[]
    for _ in range(n_real):
        pd=pdm.copy(); pr=prm.copy()
        for i in range(case.n_bus):
            if pdm[i]>1e-9: pd[i]=sample_trunc_normal(pdm[i],0.10*max(pdm[i],1e-3),0,None)
        for k,b in enumerate(case.pv_buses): pr[b]=sample_trunc_normal(prm[b],0.12*max(prm[b],1e-3),0,float(case.pv_pmax[k]))
        qd=qdm*(pd/np.maximum(pdm,1e-6)); qr=pr*math.tan(math.acos(case.pv_pf))
        hs=[solve_flex_support_gurobi_33bus(case,pd,qd,pr,qr,float(np.cos(th)),float(np.sin(th))) for th in theta_list]
        clouds.append(support_values_to_polygon(theta_list,np.array(hs)))
    plt.figure(figsize=(8,7),dpi=140)
    for p in clouds:
        if len(p)>2: plt.plot(np.r_[p[:,0],p[0,0]],np.r_[p[:,1],p[0,1]],color='#94a3b8',alpha=0.18,lw=0.8)
    rng2=np.random.default_rng(SEED_EVAL+445); pdm,qdm,prm,qrm=sample_scenario_means(case,rng2)
    hs_mc=[]; hs_bp=[]
    for th in theta_list:
        a,b=np.cos(th),np.sin(th); hs_mc.append(solve_flex_support_gurobi_33bus(case,pdm,qdm,prm,qrm,float(a),float(b)))
        xt=torch.tensor((make_feature_vector(case,pdm,prm).reshape(1,-1)-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE); th_t=torch.tensor([[a,b]],dtype=torch.float32,device=DEVICE)
        with torch.no_grad():
            _,w,mu,s=net.forward_gmm(xt,th_t,sample=False); qn=gmm_quantile_torch(w,mu,s,[0.5]).cpu().numpy().reshape(-1)[0]; hs_bp.append(float(norm['h_mean'][0,0])+float(norm['h_std'][0,0])*qn)
    pmc=support_values_to_polygon(theta_list,np.array(hs_mc)); pbp=support_values_to_polygon(theta_list,np.array(hs_bp))
    if len(pmc)>2: plt.plot(np.r_[pmc[:,0],pmc[0,0]],np.r_[pmc[:,1],pmc[0,1]],color='#1d4ed8',lw=2.8,label='MC median domain')
    if len(pbp)>2: plt.plot(np.r_[pbp[:,0],pbp[0,0]],np.r_[pbp[:,1],pbp[0,1]],'--',color='#c2410c',lw=2.8,label='B-PINN median domain')
    plt.title('Realization cloud of flexibility domains under source-load uncertainty'); plt.xlabel('P0 (MW)'); plt.ylabel('Q0 (Mvar)'); plt.grid(alpha=0.25); plt.legend(['MC realization domains','MC median domain','B-PINN median domain']); plt.axis('equal'); plt.tight_layout(); plt.savefig('FlexDomain_realization_cloud_v2.png',dpi=280)



def polygon_area(poly):
    if poly is None or len(poly)<3: return 0.0
    x=poly[:,0]; y=poly[:,1]
    return 0.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))

def support_values_to_polygon(theta_list,h_values,eps=1e-9,do_convex_cleanup=True):
    h_values=np.asarray(h_values,dtype=float)
    if h_values.ndim!=1 or len(h_values)!=len(theta_list): return np.zeros((0,2))
    valid_h=np.isfinite(h_values)
    pts=[]
    for j in range(len(theta_list)):
        k=(j+1)%len(theta_list)
        if (not valid_h[j]) or (not valid_h[k]): continue
        d1=np.array([np.cos(theta_list[j]),np.sin(theta_list[j])]); d2=np.array([np.cos(theta_list[k]),np.sin(theta_list[k])])
        A=np.vstack([d1,d2]); b=np.array([h_values[j],h_values[k]])
        if np.linalg.cond(A)>1e10: continue
        try: y=np.linalg.solve(A,b)
        except Exception: continue
        pts.append(y)
    if len(pts)<3: return np.zeros((0,2))
    pts=np.array(pts,dtype=float)
    pts=pts[np.all(np.isfinite(pts),axis=1)]
    pts=pts[(np.abs(pts[:,0])<=POLYGON_COORD_MAX)&(np.abs(pts[:,1])<=POLYGON_COORD_MAX)]
    if len(pts)<3: return np.zeros((0,2))
    pts=np.unique(np.round(pts,10),axis=0)
    if do_convex_cleanup and len(pts)>=3:
        try:
            from scipy.spatial import ConvexHull
            hull=ConvexHull(pts); pts=pts[hull.vertices]
        except Exception:
            ang=np.arctan2(pts[:,1],pts[:,0]); pts=pts[np.argsort(ang)]
    area=polygon_area(pts)
    if area<1e-8 or area>1e4: print(f"[polygon] warning: suspicious area={area:.4e}")
    return pts

def experiment_ready_check(case, XMU, XREAL, THETA_FEAT, YH, YP0, YQ0, YPG, YQG, active_records):
    N,M,T=YH.shape; ng=len(case.gen_buses)
    finite = np.all(np.isfinite(XMU)) and np.all(np.isfinite(XREAL)) and np.all(np.isfinite(YH)) and np.all(np.isfinite(YP0)) and np.all(np.isfinite(YQ0)) and np.all(np.isfinite(YPG)) and np.all(np.isfinite(YQG))
    shape_ok = (YP0.shape==YH.shape) and (YQ0.shape==YH.shape) and (YPG.shape==(N,M,T,ng)) and (YQG.shape==(N,M,T,ng))
    active_ok = (len(active_records)==N*M*T)
    al=THETA_FEAT[:,0].reshape(1,1,T); be=THETA_FEAT[:,1].reshape(1,1,T)
    max_abs_h_res=float(np.max(np.abs(YH-(al*YP0+be*YQ0))))
    print(f"[ready-check] finite={finite and shape_ok}")
    print(f"[ready-check] active_records_match={active_ok}")
    print(f"[ready-check] max_abs_h_res={max_abs_h_res:.3e}")
    print(f"[ready-check] h range={YH.min():.4f}..{YH.max():.4f}")
    print(f"[ready-check] P0/Q0 range={YP0.min():.4f}..{YP0.max():.4f} / {YQ0.min():.4f}..{YQ0.max():.4f}")
    print(f"[ready-check] Pg/Qg range={YPG.min():.4f}..{YPG.max():.4f} / {YQG.min():.4f}..{YQG.max():.4f}")
    if (not finite) or (not shape_ok) or (not active_ok):
        print("[ready-check] warning: dataset failed readiness check.")
        raise RuntimeError("experiment_ready_check failed")

def build_dataset_cache_meta(case, theta_list):
    return {
        "run_mode": RUN_MODE,
        "num_scenarios": NUM_SCENARIOS,
        "mc_per_scenario": MC_PER_SCENARIO,
        "n_theta": N_THETA,
        "seed_data": SEED_DATA,
        "theta_list": [float(x) for x in theta_list],
        "n_bus": int(case.n_bus),
        "n_branch": int(case.from_bus.size),
        "pv_buses": [int(x) for x in case.pv_buses.tolist()],
        "gen_buses": [int(x) for x in case.gen_buses.tolist()],
        "pg_min": [float(x) for x in case.pg_min.tolist()],
        "pg_max": [float(x) for x in case.pg_max.tolist()],
        "qg_min": [float(x) for x in case.qg_min.tolist()],
        "qg_max": [float(x) for x in case.qg_max.tolist()],
        "vmin": float(case.vmin),
        "vmax": float(case.vmax),
        "fmax_p_first": float(case.fmax_p[0]),
        "fmax_q_first": float(case.fmax_q[0]),
        "timestamp": datetime.utcnow().isoformat(),
        "script_dataset_version": DATASET_VERSION,
    }

def dataset_cache_meta_matches(meta, case, theta_list):
    keys = {
        "run_mode": RUN_MODE, "num_scenarios": NUM_SCENARIOS, "mc_per_scenario": MC_PER_SCENARIO,
        "n_theta": N_THETA, "seed_data": SEED_DATA, "script_dataset_version": DATASET_VERSION,
        "n_bus": int(case.n_bus), "n_branch": int(case.from_bus.size),
    }
    for k,v in keys.items():
        if meta.get(k) != v: return False
    if meta.get("theta_list") != [float(x) for x in theta_list]: return False
    if meta.get("pv_buses") != [int(x) for x in case.pv_buses.tolist()]: return False
    if meta.get("gen_buses") != [int(x) for x in case.gen_buses.tolist()]: return False
    return True

def save_flex_dataset_cache(XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln,meta):
    Path(DATASET_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    np.savez(DATASET_CACHE_NPZ,XMU=XMU,XREAL=XREAL,THETA_FEAT=THETA_FEAT,THETA_LIST=THETA_LIST,YH=YH,YP0=YP0,YQ0=YQ0,YPG=YPG,YQG=YQG)
    with open(DATASET_CACHE_PICKLE,"wb") as f: pickle.dump({"active":active,"alln":alln},f)
    with open(DATASET_CACHE_META,"w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,indent=2)
    print(f"[dataset-cache] saved numeric arrays to {DATASET_CACHE_NPZ}")
    print(f"[dataset-cache] saved active records to {DATASET_CACHE_PICKLE}")
    print(f"[dataset-cache] saved metadata to {DATASET_CACHE_META}")
    print(f"[dataset-cache] XMU shape={XMU.shape}")
    print(f"[dataset-cache] YH shape={YH.shape}")
    print(f"[dataset-cache] active_records count={len(active)}")

def load_flex_dataset_cache(meta_matched):
    d=np.load(DATASET_CACHE_NPZ,allow_pickle=True)
    with open(DATASET_CACHE_PICKLE,"rb") as f: pp=pickle.load(f)
    XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG=d["XMU"],d["XREAL"],d["THETA_FEAT"],d["YH"],d["YP0"],d["YQ0"],d["YPG"],d["YQG"]
    active,alln=pp["active"],pp["alln"]
    print("[dataset-cache] loaded from cache")
    print(f"[dataset-cache] XMU shape={XMU.shape}")
    print(f"[dataset-cache] XREAL shape={XREAL.shape}")
    print(f"[dataset-cache] YH shape={YH.shape}")
    print(f"[dataset-cache] YP0/YQ0/YPG/YQG shape={YP0.shape}/{YQ0.shape}/{YPG.shape}/{YQG.shape}")
    print(f"[dataset-cache] active_records count={len(active)}")
    print(f"[dataset-cache] metadata matched={meta_matched}")
    return XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln

def get_or_build_flex_dataset_cache(case):
    Path(DATASET_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    meta_now=build_dataset_cache_meta(case,THETA_LIST)
    exists=Path(DATASET_CACHE_NPZ).exists() and Path(DATASET_CACHE_PICKLE).exists() and Path(DATASET_CACHE_META).exists()
    if DATASET_CACHE_MODE=="rebuild":
        exists=False
    if exists:
        with open(DATASET_CACHE_META,"r",encoding="utf-8") as f: meta_old=json.load(f)
        matched=dataset_cache_meta_matches(meta_old,case,THETA_LIST)
        if not matched:
            if DATASET_CACHE_MODE=="load_only":
                raise RuntimeError("dataset cache unavailable or mismatched under load_only mode")
            print("[dataset-cache] metadata mismatch, rebuilding dataset.")
            exists=False
        else:
            print(f"[dataset-cache] using cached dataset: {DATASET_CACHE_NPZ}")
            print(f"[dataset-cache] mode={DATASET_CACHE_MODE}")
            print(f"[dataset-cache] dataset version={DATASET_VERSION}")
            return load_flex_dataset_cache(True)
    else:
        if DATASET_CACHE_MODE=="load_only":
            raise RuntimeError("dataset cache unavailable or mismatched under load_only mode")
    if USE_DATASET_CACHE:
        XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln=get_or_build_flex_dataset_cache(case)
    else:
        XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln=generate_flex_dataset(case,NUM_SCENARIOS,MC_PER_SCENARIO,THETA_LIST,SEED_DATA)
    save_flex_dataset_cache(XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln,meta_now)
    return XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln

def save_training_artifacts(net,norm,config_dict):
    if not SAVE_TRAINING_RESULT: return
    Path(TRAINING_RESULT_DIR).mkdir(parents=True,exist_ok=True)
    mp=f"{TRAINING_RESULT_DIR}/{TRAINING_RUN_TAG}_model.pt"
    npth=f"{TRAINING_RESULT_DIR}/{TRAINING_RUN_TAG}_norm.pkl"
    cp=f"{TRAINING_RESULT_DIR}/{TRAINING_RUN_TAG}_config.json"
    torch.save(net.state_dict(),mp)
    with open(npth,"wb") as f: pickle.dump(norm,f)
    with open(cp,"w",encoding="utf-8") as f: json.dump(config_dict,f,ensure_ascii=False,indent=2)
    print(f"[train-save] model saved to {mp}")
    print(f"[train-save] norm saved to {npth}")
    print(f"[train-save] config saved to {cp}")


class AtomGMMDesignNote:
    """
    如果边界诊断显示某些 theta 存在明显点质量，则下一步可将 GMM-K 改为：
    p(h|x_mu,theta) = pi_atom * delta(h-h_sat) + (1-pi_atom) * GMM_K(h)
    网络输出：
    - pi_atom(x_mu,theta)
    - h_sat(x_mu,theta) 或基于经验 mode 的半固定 h_sat
    - GMM_K weights/mu/sigma for continuous part

    CDF:
    F(h) = (1-pi_atom)*F_gmm(h) + pi_atom*I(h>=h_sat)

    NLL:
    if |h-h_sat| <= eps_atom:
        -log(pi_atom + (1-pi_atom)*p_gmm_mass_near_atom)
    else:
        -log((1-pi_atom)*p_gmm(h))

    但本版本只做诊断，不实现训练。
    """

def _mode_mass(samples, eps):
    x=np.sort(np.asarray(samples,dtype=float))
    n=len(x)
    if n==0: return np.nan,np.nan
    best_m,best_h=0.0,float(x[0])
    l=0
    for r in range(n):
        while x[r]-x[l] > 2*eps: l+=1
        m=(r-l+1)/n
        if m>best_m: best_m=m; best_h=float(x[(l+r)//2])
    return best_h,best_m

def diagnose_atom_active_constraints(YH,active_records,theta_idx,h_mode,eps=1e-2):
    N,M,T=YH.shape
    if (active_records is None) or (len(active_records)!=N*M*T):
        return {'atom_active_top1_pattern':'NA','atom_active_top1_ratio':np.nan,'atom_active_top_constraints':'NA','nonatom_active_top1_pattern':'NA','nonatom_active_top1_ratio':np.nan,'nonatom_active_top_constraints':'NA'}
    flat=YH.reshape(-1)
    idxs=[]
    base=[]
    for s in range(N):
        for m in range(M):
            i=(s*M*T)+(m*T)+theta_idx
            idxs.append(i)
    idxs=np.array(idxs,dtype=int)
    hs=flat[idxs]
    atom_mask=np.abs(hs-h_mode)<=eps
    from collections import Counter
    def stat(mask):
        ids=idxs[mask]
        if len(ids)==0: return 'NA',np.nan,'NA'
        pats=[str(active_records[i].get('signature','NA')) for i in ids]
        c=Counter(pats); top,ct=c.most_common(1)[0]
        names=[]
        for i in ids:
            names.extend(active_records[i].get('active_names',[]))
        cc=Counter(names)
        commons=';'.join([a for a,_ in cc.most_common(5)]) if cc else 'NA'
        return top,ct/max(1,len(ids)),commons
    a1,a2,a3=stat(atom_mask); b1,b2,b3=stat(~atom_mask)
    return {'atom_active_top1_pattern':a1,'atom_active_top1_ratio':a2,'atom_active_top_constraints':a3,'nonatom_active_top1_pattern':b1,'nonatom_active_top1_ratio':b2,'nonatom_active_top_constraints':b3}

def diagnose_support_distribution_atoms(YH,THETA_FEAT,theta_list,active_records=None,arms_csv_path=None,eps_list=(1e-4,1e-3,1e-2,5e-2),top_bins=3,save_prefix='support_atom_diagnostic'):
    N,M,T=YH.shape
    arms_map={}
    if arms_csv_path is not None and Path(arms_csv_path).exists():
        import csv as _csv
        with open(arms_csv_path,'r',encoding='utf-8') as f:
            r=_csv.DictReader(f)
            for row in r: arms_map[int(row['theta_idx'])]=float(row['arms_pct'])
    rows=[]
    YH_eval=[]
    XMU_eval=[]; scene_rows=[]
    YH_eval=[]
    XMU_eval=[]; active_rows=[]
    YH_eval=[]
    XMU_eval=[]
    for j in range(T):
        hs=np.asarray(YH[:,:,j].reshape(-1),dtype=float)
        hmin,hmax,hmean,hstd=np.min(hs),np.max(hs),np.mean(hs),np.std(hs)
        q=np.quantile(hs,[0.01,0.05,0.5,0.95,0.99])
        mode_mass={}
        for e in eps_list:
            md,ms=_mode_mass(hs,e); mode_mass[e]=(md,ms)
        m12=mode_mass[1e-2][1]; md12=mode_mass[1e-2][0]
        near_max=(abs(md12-hmax)<=1e-2) or (abs(md12-hmax)<=0.01*(hmax-hmin+1e-9))
        near_min=(abs(md12-hmin)<=1e-2) or (abs(md12-hmin)<=0.01*(hmax-hmin+1e-9))
        b_atom=(m12>=ATOM_MASS_THRESHOLD) and (near_max or near_min)
        sc_modes=[]; s_atom3=[]; s_atom2=[]; s_b=[]
        for sidx in range(N):
            hsc=np.asarray(YH[sidx,:,j],dtype=float)
            md3,ms3=_mode_mass(hsc,1e-3); md2,ms2=_mode_mass(hsc,1e-2)
            mn,mx=np.min(hsc),np.max(hsc)
            nmin=(abs(md2-mn)<=1e-2) or (abs(md2-mn)<=0.01*(mx-mn+1e-9)); nmax=(abs(md2-mx)<=1e-2) or (abs(md2-mx)<=0.01*(mx-mn+1e-9))
            sb=(ms2>=ATOM_MASS_THRESHOLD) and (nmin or nmax)
            scene_rows.append({'scenario_idx':sidx,'theta_idx':j,'theta':float(theta_list[j]),'scene_h_min':float(mn),'scene_h_max':float(mx),'scene_h_mean':float(np.mean(hsc)),'scene_h_std':float(np.std(hsc)),'scene_mode_eps_1e-3':float(md3),'scene_mass_eps_1e-3':float(ms3),'scene_mode_eps_1e-2':float(md2),'scene_mass_eps_1e-2':float(ms2),'scene_mode_near_min':int(nmin),'scene_mode_near_max':int(nmax),'scene_boundary_like_atom':int(sb)})
            sc_modes.append(md2); s_atom3.append(ms3>=ATOM_MASS_THRESHOLD); s_atom2.append(ms2>=ATOM_MASS_THRESHOLD); s_b.append(sb)
        arms=arms_map.get(j,float('nan'))
        sug='continuous_gmm_ok'
        if m12>=ATOM_MASS_THRESHOLD and (near_max or near_min): sug='atom_plus_continuous'
        elif m12>=POSSIBLE_ATOM_MASS_THRESHOLD and (np.isfinite(arms) and arms>=HIGH_ARMS_THRESHOLD): sug='possible_atom_plus_continuous'
        row={'theta_idx':j,'theta':float(theta_list[j]),'alpha':float(THETA_FEAT[j,0]),'beta':float(THETA_FEAT[j,1]),'h_min':float(hmin),'h_max':float(hmax),'h_mean':float(hmean),'h_std':float(hstd),'h_q01':float(q[0]),'h_q05':float(q[1]),'h_q50':float(q[2]),'h_q95':float(q[3]),'h_q99':float(q[4]),'mode_eps_1e-4':float(mode_mass[1e-4][0]),'mass_eps_1e-4':float(mode_mass[1e-4][1]),'mode_eps_1e-3':float(mode_mass[1e-3][0]),'mass_eps_1e-3':float(mode_mass[1e-3][1]),'mode_eps_1e-2':float(mode_mass[1e-2][0]),'mass_eps_1e-2':float(mode_mass[1e-2][1]),'mode_eps_5e-2':float(mode_mass[5e-2][0]),'mass_eps_5e-2':float(mode_mass[5e-2][1]),'mode_near_min':int(near_min),'mode_near_max':int(near_max),'boundary_like_atom':int(b_atom),'scene_atom_ratio_1e-3':float(np.mean(s_atom3)),'scene_atom_ratio_1e-2':float(np.mean(s_atom2)),'scene_boundary_atom_ratio_1e-2':float(np.mean(s_b)),'scene_mode_std':float(np.std(sc_modes)),'scene_mode_range':float(np.max(sc_modes)-np.min(sc_modes)),'arms_pct':float(arms) if np.isfinite(arms) else np.nan,'suggested_distribution':sug}
        rows.append(row)
        ar=diagnose_atom_active_constraints(YH,active_records,j,row['mode_eps_1e-2'],eps=1e-2)
        active_rows.append({'theta_idx':j,'theta':float(theta_list[j]),'h_mode_eps_1e-2':row['mode_eps_1e-2'],'mass_eps_1e-2':row['mass_eps_1e-2'],**ar})
    import csv as _csv
    with open(f'{save_prefix}_by_theta.csv','w',newline='',encoding='utf-8') as f:
        w=_csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(f'{save_prefix}_by_scene_theta.csv','w',newline='',encoding='utf-8') as f:
        w=_csv.DictWriter(f,fieldnames=list(scene_rows[0].keys())); w.writeheader(); w.writerows(scene_rows)
    if active_records is not None and len(active_records)==N*M*T:
        with open(f'{save_prefix}_active_patterns.csv','w',newline='',encoding='utf-8') as f:
            w=_csv.DictWriter(f,fieldnames=list(active_rows[0].keys())); w.writeheader(); w.writerows(active_rows)
    else:
        print('[atom-diagnostic] warning: active_records unavailable or length mismatch, skip active pattern csv')
    # plots
    xs=[r['theta_idx'] for r in rows]; m2=[r['mass_eps_1e-2'] for r in rows]; arms=[r['arms_pct'] for r in rows]
    colors=['#dc2626' if v>=ATOM_MASS_THRESHOLD else '#6b7280' for v in m2]
    plt.figure(figsize=(9,4),dpi=260); plt.bar(xs,m2,color=colors); plt.axhline(ATOM_MASS_THRESHOLD,color='k',ls='--',lw=1); plt.grid(alpha=0.25)
    if np.isfinite(np.nanmean(arms)):
        ax2=plt.gca().twinx(); ax2.plot(xs,arms,'-o',color='#1d4ed8',alpha=0.8,label='ARMS')
    plt.title('support atom mass by theta'); plt.xlabel('theta idx'); plt.ylabel('mass_eps_1e-2'); plt.tight_layout(); plt.savefig('support_atom_mass_by_theta.png',dpi=280)
    plt.figure(figsize=(9,4),dpi=260); plt.bar(xs,[r['scene_boundary_atom_ratio_1e-2'] for r in rows],color='#60a5fa'); plt.axhline(0.2,color='k',ls='--',lw=1); plt.grid(alpha=0.25); plt.title('scene boundary atom ratio by theta'); plt.xlabel('theta idx'); plt.ylabel('ratio'); plt.tight_layout(); plt.savefig('support_scene_atom_ratio_by_theta.png',dpi=280)
    pick=np.argsort(np.array([r['arms_pct'] if np.isfinite(r['arms_pct']) else r['mass_eps_1e-2']*100 for r in rows]))[::-1][:3]
    fig,axs=plt.subplots(3,1,figsize=(8,10),dpi=260)
    for ax,k in zip(axs,pick):
        r=rows[int(k)]; hs=YH[:,:,r['theta_idx']].reshape(-1)
        ax.hist(hs,bins=40,color='#93c5fd',alpha=0.7,density=True)
        ax2=ax.twinx(); sx=np.sort(hs); c=np.arange(1,len(sx)+1)/len(sx); ax2.plot(sx,c,color='#1d4ed8',lw=1.5)
        ax.axvline(r['mode_eps_1e-2'],color='#dc2626',ls='--'); ax.set_title(f"theta={r['theta_idx']} mass={r['mass_eps_1e-2']:.3f} arms={r['arms_pct']:.2f}")
        ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig('support_distribution_histograms_worst_theta.png',dpi=280)
    valid=[r for r in rows if np.isfinite(r['arms_pct'])]
    if valid:
        plt.figure(figsize=(6,5),dpi=260); xv=[r['mass_eps_1e-2'] for r in valid]; yv=[r['arms_pct'] for r in valid]
        plt.scatter(xv,yv,c='#0ea5e9')
        for r in valid: plt.text(r['mass_eps_1e-2'],r['arms_pct'],str(r['theta_idx']),fontsize=8)
        plt.grid(alpha=0.25); plt.xlabel('mass_eps_1e-2'); plt.ylabel('ARMS (%)'); plt.tight_layout(); plt.savefig('support_atom_vs_arms_scatter.png',dpi=280)
    strong=[r['theta_idx'] for r in rows if r['mass_eps_1e-2']>=ATOM_MASS_THRESHOLD and r['boundary_like_atom']==1]
    poss=[r['theta_idx'] for r in rows if r['mass_eps_1e-2']>=POSSIBLE_ATOM_MASS_THRESHOLD and (np.isfinite(r['arms_pct']) and r['arms_pct']>=HIGH_ARMS_THRESHOLD)]
    highlow=[r['theta_idx'] for r in rows if (np.isfinite(r['arms_pct']) and r['arms_pct']>=HIGH_ARMS_THRESHOLD and r['mass_eps_1e-2']<POSSIBLE_ATOM_MASS_THRESHOLD)]
    print('[atom-diagnostic-summary]')
    print(f'theta with strong boundary atom: {strong}')
    print(f'theta with possible atom: {poss}')
    print(f'theta with high ARMS but low atom mass: {highlow}')
    print('recommendation:')
    print('- if strong_boundary_atom directions exist and overlap with high ARMS, consider discrete-continuous mixture / Atom-GMM;')
    print('- if high ARMS but low atom mass, improve continuous distribution head or calibration instead;')
    print('- if scene_atom_ratio high but global mass low, use conditional atom with h_sat(x_mu,theta), not fixed atom.')


def analyze_cdf_error_decomposition(net,norm,XMU_eval,THETA_FEAT,theta_list,YH_eval,atom_diag_csv_path=None,active_diag_csv_path=None,taus=(0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99),cdf_grid_size=300,n_posterior_samples=30,save_prefix="cdf_error_decomp"):
    print(f"[cdf-error-decomp] XMU_eval shape={XMU_eval.shape}")
    print(f"[cdf-error-decomp] YH_eval shape={YH_eval.shape}")
    same_baseline=(XMU_eval.shape[0]==YH_eval.shape[0]) or (XMU_eval.shape[0]==1 and YH_eval.shape[0]==1)
    print(f"[cdf-error-decomp] using same eval baseline as all-theta ARMS={same_baseline}")
    if not same_baseline: raise ValueError(f"XMU_eval and YH_eval shape mismatch: {XMU_eval.shape} vs {YH_eval.shape}")
    import csv
    atom_map={}
    if atom_diag_csv_path and Path(atom_diag_csv_path).exists():
        with open(atom_diag_csv_path,'r',encoding='utf-8') as f:
            for r in csv.DictReader(f): atom_map[int(r['theta_idx'])]=r
    rec_map={}
    if active_diag_csv_path and Path(active_diag_csv_path).exists():
        with open(active_diag_csv_path,'r',encoding='utf-8') as f:
            for r in csv.DictReader(f): rec_map[int(r['theta_idx'])]=r
    rows=[]; qrows=[]
    hmean=float(norm['h_mean'][0,0]); hstd=float(norm['h_std'][0,0])
    for j,th in enumerate(theta_list):
        h_mc=np.asarray(YH_eval[:,:,j].reshape(-1),dtype=float)
        if len(h_mc)<10: continue
        hr=max(1e-9,h_mc.max()-h_mc.min()); z=np.linspace(h_mc.min()-0.05*hr,h_mc.max()+0.05*hr,cdf_grid_size)
        F_mc=np.searchsorted(np.sort(h_mc),z,side='right')/len(h_mc)
        # posterior CDF ensemble
        cdfs=[]; q50s=[]
        for _ in range(n_posterior_samples):
            c_scene=[]
            for sidx in range(XMU_eval.shape[0]):
                xmu=XMU_eval[sidx:sidx+1]
                xt=torch.tensor((xmu-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE)
                th_t=torch.tensor([THETA_FEAT[j]],dtype=torch.float32,device=DEVICE)
                with torch.no_grad():
                    _,w,mu,s=net.forward_gmm(xt,th_t,sample=True)
                zn=(z-hmean)/(hstd+1e-9)
                c_scene.append(gmm_cdf(zn,w.cpu().numpy().reshape(-1),mu.cpu().numpy().reshape(-1),s.cpu().numpy().reshape(-1)))
                q50s.append((hmean+hstd*gmm_quantile_torch(w,mu,s,[0.5]).cpu().numpy().reshape(-1)[0]))
            cdfs.append(np.mean(np.array(c_scene),axis=0))
        cdfs=np.array(cdfs); F_bp=cdfs.mean(0)
        arms=100*np.sqrt(np.mean((F_bp-F_mc)**2)); ks=np.max(np.abs(F_bp-F_mc)); sbias=np.mean(F_bp-F_mc); abias=np.mean(np.abs(F_bp-F_mc))
        q10=np.quantile(h_mc,0.10); q90=np.quantile(h_mc,0.90)
        m1=z<=q10; m2=(z>q10)&(z<q90); m3=z>=q90
        lower=100*np.sqrt(np.mean((F_bp[m1]-F_mc[m1])**2)) if m1.any() else np.nan
        middle=100*np.sqrt(np.mean((F_bp[m2]-F_mc[m2])**2)) if m2.any() else np.nan
        upper=100*np.sqrt(np.mean((F_bp[m3]-F_mc[m3])**2)) if m3.any() else np.nan
        qmc=np.quantile(h_mc,taus)
        qbp_s=[]; bmean_s=[]; bstd_s=[]
        for sidx in range(XMU_eval.shape[0]):
            xmu0=XMU_eval[sidx:sidx+1]
            xt=torch.tensor((xmu0-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE); th_t=torch.tensor([THETA_FEAT[j]],dtype=torch.float32,device=DEVICE)
            with torch.no_grad():
                _,w,mu,s=net.forward_gmm(xt,th_t,sample=False); qn=gmm_quantile_torch(w,mu,s,list(taus)).cpu().numpy().reshape(-1)
            qbp_s.append(hmean+hstd*qn)
            bp_mean_norm=(w*mu).sum(dim=1).cpu().numpy()[0]; bp_var_norm=(w*(s*s+mu*mu)).sum(dim=1).cpu().numpy()[0]-bp_mean_norm**2
            bmean_s.append(hmean+hstd*bp_mean_norm); bstd_s.append(abs(hstd)*np.sqrt(max(0.0,bp_var_norm)))
        qbp=np.mean(np.array(qbp_s),axis=0); bp_mean=float(np.mean(bmean_s)); bp_std=float(np.mean(bstd_s)); mc_mean=float(np.mean(h_mc)); mc_std=float(np.std(h_mc))
        errs=qbp-qmc
        qd={f'q{int(t*100):02d}_mc':float(qmc[i]) for i,t in enumerate(taus)}
        qd.update({f'q{int(t*100):02d}_bp':float(qbp[i]) for i,t in enumerate(taus)})
        qd.update({f'q{int(t*100):02d}_err':float(errs[i]) for i,t in enumerate(taus)})
        for i,t in enumerate(taus): qrows.append({'theta_idx':j,'theta':float(th),'tau':float(t),'q_mc':float(qmc[i]),'q_bp':float(qbp[i]),'q_err':float(errs[i]),'q_abs_err':float(abs(errs[i])),'q_rel_err':float(abs(errs[i])/(abs(qmc[i])+1e-9))})
        tail_span_mc=float(np.quantile(h_mc,0.95)-np.quantile(h_mc,0.05)); tail_span_bp=float(np.interp(0.95,taus,qbp)-np.interp(0.05,taus,qbp))
        ext_mc=float(np.quantile(h_mc,0.99)-np.quantile(h_mc,0.01)); ext_bp=float(np.interp(0.99,taus,qbp)-np.interp(0.01,taus,qbp))
        post_std=float(np.mean(np.std(cdfs,axis=0))); post_bw=float(np.mean(np.quantile(cdfs,0.975,axis=0)-np.quantile(cdfs,0.025,axis=0))); post_med_std=float(np.std(q50s))
        atom=atom_map.get(j,{})
        m_atom=float(atom.get('mass_eps_1e-2',np.nan)) if atom else np.nan
        b_atom=int(float(atom.get('boundary_like_atom',0))) if atom else 0
        scene_atom=float(atom.get('scene_boundary_atom_ratio_1e-2',np.nan)) if atom else np.nan
        sugg=atom.get('suggested_distribution','NA') if atom else 'NA'
        median_err=float(np.interp(0.50,taus,errs)); mean_err=float(bp_mean-mc_mean); std_ratio=float(bp_std/(mc_std+1e-9)); tail_low=float(np.interp(0.05,taus,errs)); tail_high=float(np.interp(0.95,taus,errs))
        tags=[]
        if b_atom and (m_atom>=0.20) and arms>=10: tags.append('boundary_atom_error')
        if abs(median_err)>=0.10*hr or abs(mean_err)>=0.10*hr: tags.append('location_shift_error')
        if abs(std_ratio-1.0)>=0.25 or abs((tail_span_bp-tail_span_mc))>=0.20*(tail_span_mc+1e-9): tags.append('scale_mismatch_error')
        if max(abs(tail_low),abs(tail_high))>=2*abs(median_err)+1e-9 and arms>=8: tags.append('tail_mismatch_error')
        if arms>=8 and len(tags)==0 and not (m_atom>=0.10): tags.append('shape_mismatch_error')
        if (m_atom>=0.10) and arms>=10: tags.append('possible_atom_error')
        if arms<5 and abs(median_err)<0.05*hr and abs(std_ratio-1.0)<0.15: tags.append('well_calibrated')
        pri='well_calibrated'
        order=['boundary_atom_error','location_shift_error','scale_mismatch_error','tail_mismatch_error','shape_mismatch_error','possible_atom_error','well_calibrated']
        for o in order:
            if o in tags: pri=o; break
        row={'theta_idx':j,'theta':float(th),'alpha':float(THETA_FEAT[j,0]),'beta':float(THETA_FEAT[j,1]),'arms_pct':float(arms),'ks_stat':float(ks),'signed_cdf_bias':float(sbias),'abs_cdf_bias':float(abias),'lower_cdf_arms':float(lower),'middle_cdf_arms':float(middle),'upper_cdf_arms':float(upper),'mc_mean':mc_mean,'bp_mean':float(bp_mean),'mean_err':float(mean_err),'mc_std':mc_std,'bp_std':float(bp_std),'std_err':float(bp_std-mc_std),'std_ratio':float(std_ratio),'tail_span_mc':tail_span_mc,'tail_span_bp':tail_span_bp,'tail_span_err':tail_span_bp-tail_span_mc,'extreme_span_mc':ext_mc,'extreme_span_bp':ext_bp,'extreme_span_err':ext_bp-ext_mc,'posterior_cdf_std_mean':post_std,'posterior_cdf_band_width_mean':post_bw,'posterior_median_std':post_med_std,'bias_to_uncertainty_ratio':float(abias/(post_std+1e-9)),'mass_eps_1e-2':m_atom,'boundary_like_atom':b_atom,'scene_boundary_atom_ratio_1e-2':scene_atom,'suggested_distribution':sugg,'primary_error_type':pri,'error_type_multi':';'.join(tags)}
        row.update(qd); rows.append(row)
    import csv as _csv
    if not rows: return
    with open(f'{save_prefix}_by_theta.csv','w',newline='',encoding='utf-8') as f: w=_csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(f'{save_prefix}_quantile_matrix.csv','w',newline='',encoding='utf-8') as f: w=_csv.DictWriter(f,fieldnames=list(qrows[0].keys())); w.writeheader(); w.writerows(qrows)
    from collections import defaultdict
    grp=defaultdict(list)
    for r in rows: grp[r['primary_error_type']].append(r)
    srows=[]
    for k,v in grp.items():
        srows.append({'primary_error_type':k,'count':len(v),'mean_arms':float(np.mean([a['arms_pct'] for a in v])),'max_arms':float(np.max([a['arms_pct'] for a in v])),'mean_abs_median_err':float(np.mean([abs(a.get('q50_err',0.0)) for a in v])),'mean_abs_tail_span_err':float(np.mean([abs(a['tail_span_err']) for a in v])),'mean_std_ratio':float(np.mean([a['std_ratio'] for a in v])),'theta_indices':';'.join([str(a['theta_idx']) for a in v])})
    with open(f'{save_prefix}_summary.csv','w',newline='',encoding='utf-8') as f: w=_csv.DictWriter(f,fieldnames=list(srows[0].keys())); w.writeheader(); w.writerows(srows)
    rec=[]
    maprec={'boundary_atom_error':'consider discrete-continuous mixture / Atom-GMM','location_shift_error':'strengthen median/mean calibration; increase q50/mean loss or scenario-conditioned bias correction','scale_mismatch_error':'strengthen variance/interval width calibration; add scale loss or CRPS-like loss','tail_mismatch_error':'strengthen tail quantile loss at 0.01/0.05/0.95/0.99','shape_mismatch_error':'increase distribution flexibility or condition on active pattern / critical region','possible_atom_error':'consider discrete-continuous mixture / Atom-GMM','well_calibrated':'no major change needed'}
    for r in rows: rec.append({'theta_idx':r['theta_idx'],'theta':r['theta'],'primary_error_type':r['primary_error_type'],'recommendation':maprec.get(r['primary_error_type'],'no major change needed')})
    with open(f'{save_prefix}_recommendation.csv','w',newline='',encoding='utf-8') as f: w=_csv.DictWriter(f,fieldnames=list(rec[0].keys())); w.writeheader(); w.writerows(rec)
    # plots
    cmap={'boundary_atom_error':'#dc2626','location_shift_error':'#2563eb','scale_mismatch_error':'#7c3aed','tail_mismatch_error':'#ea580c','shape_mismatch_error':'#059669','possible_atom_error':'#be123c','well_calibrated':'#6b7280'}
    plt.figure(figsize=(9,4),dpi=260); xs=[r['theta_idx'] for r in rows]; ys=[r['arms_pct'] for r in rows]; cs=[cmap.get(r['primary_error_type'],'#6b7280') for r in rows]; plt.bar(xs,ys,color=cs); plt.axhline(np.mean(ys),ls='--',color='k'); plt.axhline(10,ls=':',color='r'); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig('cdf_error_type_by_theta.png',dpi=280)
    taus_sorted=sorted(list(set([q['tau'] for q in qrows]))); mat=np.full((len(rows),len(taus_sorted)),np.nan)
    for qi in qrows: mat[int(qi['theta_idx']),taus_sorted.index(qi['tau'])]=qi['q_err']
    plt.figure(figsize=(8,5),dpi=260); plt.imshow(mat,aspect='auto',cmap='coolwarm',vmin=-np.nanmax(np.abs(mat)),vmax=np.nanmax(np.abs(mat))); plt.colorbar(); plt.xticks(range(len(taus_sorted)),[f'{t:.2f}' for t in taus_sorted],rotation=45); plt.yticks(range(len(rows)),[r['theta_idx'] for r in rows]); plt.tight_layout(); plt.savefig('cdf_quantile_error_heatmap.png',dpi=280)
    plt.figure(figsize=(9,4),dpi=260); xr=np.array([max(1e-9,r.get('q99_mc',0)-r.get('q01_mc',0)) for r in rows]); a=np.array([abs(r.get('q50_err',0))/x for r,x in zip(rows,xr)]); b=np.array([abs(r['std_ratio']-1.0) for r in rows]); c=np.array([abs(r['tail_span_err'])/(abs(r['tail_span_mc'])+1e-9) for r in rows]); xx=np.arange(len(rows)); w=0.25; plt.bar(xx-w,a,width=w,label='location'); plt.bar(xx,b,width=w,label='scale'); plt.bar(xx+w,c,width=w,label='tail'); plt.legend(); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig('cdf_location_scale_tail_bar.png',dpi=280)
    worst=sorted(rows,key=lambda r:r['arms_pct'],reverse=True)[:4]
    fig,axs=plt.subplots(2,2,figsize=(11,8),dpi=260)
    for ax,r in zip(axs.ravel(),worst):
        j=r['theta_idx']; h_mc=np.asarray(YH_eval[:,:,j].reshape(-1),dtype=float); hr=max(1e-9,h_mc.max()-h_mc.min()); z=np.linspace(h_mc.min()-0.05*hr,h_mc.max()+0.05*hr,300); F_mc=np.searchsorted(np.sort(h_mc),z,side='right')/len(h_mc)
        cdfs=[]
        for _ in range(n_posterior_samples):
            xmu=XMU_eval.mean(axis=0,keepdims=True); xt=torch.tensor((xmu-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE); th_t=torch.tensor([THETA_FEAT[j]],dtype=torch.float32,device=DEVICE)
            with torch.no_grad(): _,w,mu,s=net.forward_gmm(xt,th_t,sample=True)
            zn=(z-hmean)/(hstd+1e-9); cdfs.append(gmm_cdf(zn,w.cpu().numpy().reshape(-1),mu.cpu().numpy().reshape(-1),s.cpu().numpy().reshape(-1)))
        cdfs=np.array(cdfs); m=cdfs.mean(0); lo=np.quantile(cdfs,0.025,0); hi=np.quantile(cdfs,0.975,0)
        ax.fill_between(z,lo,hi,alpha=0.2,color='#99f6e4'); ax.plot(z,F_mc,'k',lw=2); ax.plot(z,m,'--',color='#ea580c',lw=2); ax2=ax.twinx(); ax2.plot(z,m-F_mc,color='#2563eb',alpha=0.5); ax.set_title(f"theta={j} ARMS={r['arms_pct']:.2f} {r['primary_error_type']}"); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig('cdf_residual_worst_theta.png',dpi=280)
    plt.figure(figsize=(6,5),dpi=260); x=[r['mass_eps_1e-2'] if np.isfinite(r['mass_eps_1e-2']) else 0 for r in rows]; y=[r['arms_pct'] for r in rows]; c=[cmap.get(r['primary_error_type'],'#6b7280') for r in rows]; plt.scatter(x,y,c=c)
    for r in rows: plt.text((r['mass_eps_1e-2'] if np.isfinite(r['mass_eps_1e-2']) else 0),r['arms_pct'],str(r['theta_idx']),fontsize=8)
    plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig('cdf_atom_vs_error_scatter.png',dpi=280)
    plt.figure(figsize=(6,5),dpi=260); x=[r['posterior_cdf_std_mean'] for r in rows]; y=[abs(r['abs_cdf_bias']) for r in rows]; plt.scatter(x,y,c=c)
    mx=max(max(x),max(y))+1e-9; plt.plot([0,mx],[0,mx],'k--'); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig('cdf_bias_vs_uncertainty_scatter.png',dpi=280)
    print('[cdf-error-summary]')
    arr=np.array([r['arms_pct'] for r in rows]); print(f"mean ARMS = {arr.mean():.4f}"); print(f"median ARMS = {np.median(arr):.4f}"); print(f"max ARMS = {arr.max():.4f}")
    w=max(rows,key=lambda r:r['arms_pct']); print(f"worst theta = {w['theta_idx']}")
    print(f"error type counts = {{k:len(v) for k,v in grp.items()}}")
    print(f"theta requiring Atom-GMM = {[r['theta_idx'] for r in rows if r['primary_error_type'] in ['boundary_atom_error','possible_atom_error']]}")
    print(f"theta requiring location calibration = {[r['theta_idx'] for r in rows if r['primary_error_type']=='location_shift_error']}")
    print(f"theta requiring scale calibration = {[r['theta_idx'] for r in rows if r['primary_error_type']=='scale_mismatch_error']}")
    print(f"theta requiring tail calibration = {[r['theta_idx'] for r in rows if r['primary_error_type']=='tail_mismatch_error']}")
    print(f"theta requiring active-pattern/critical-region conditioning = {[r['theta_idx'] for r in rows if r['primary_error_type']=='shape_mismatch_error']}")
    hi=[r for r in rows if r['arms_pct']>=10]
    if hi and sum(r['primary_error_type']=='boundary_atom_error' for r in hi)>=len(hi)/2: print('Atom-GMM/discrete-continuous mixture is likely useful.')
    elif hi and sum(r['primary_error_type'] in ['location_shift_error','scale_mismatch_error'] for r in hi)>=len(hi)/2: print('Atom-GMM alone will not solve the issue; calibration loss or bias/scale correction is needed.')
    elif hi and sum(r['primary_error_type']=='shape_mismatch_error' for r in hi)>=len(hi)/2: print('Consider richer distribution model or active-pattern conditioned mixture.')
def main():
    case=build_ieee33_case()
    if RUN_SANITY_CHECKS: flex_opf_sanity_check(case)
    print('生成 33 节点灵活域训练数据...')
    print(f'[run-mode] RUN_MODE={RUN_MODE}')
    print(f'[run-mode] NUM_SCENARIOS={NUM_SCENARIOS}, MC_PER_SCENARIO={MC_PER_SCENARIO}, N_THETA={N_THETA}, EPOCHS={EPOCHS}')
    print(f'[run-mode] total_OPF_labels={NUM_SCENARIOS*MC_PER_SCENARIO*N_THETA}')
    print(f'[run-mode] USE_H_QUANTILE_LOSS={USE_H_QUANTILE_LOSS}, LAM_H_QUANTILE={LAM_H_QUANTILE}')
    XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln=generate_flex_dataset(case,NUM_SCENARIOS,MC_PER_SCENARIO,THETA_LIST,SEED_DATA)
    summarize_active_patterns(active,alln,top_k=10)
    experiment_ready_check(case,XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active)
    print('\n=== Support distribution atom/boundary diagnostic on training dataset ===')
    diagnose_support_distribution_atoms(YH,THETA_FEAT,THETA_LIST,active_records=active,arms_csv_path=None,save_prefix='support_atom_train')
    net,norm=train_bayes_flex_gmm2(case,XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG)
    save_training_artifacts(net,norm,{
        "run_mode":RUN_MODE,"num_scenarios":NUM_SCENARIOS,"mc_per_scenario":MC_PER_SCENARIO,
        "n_theta":N_THETA,"epochs":EPOCHS,"seed_train":SEED_TRAIN,"gmm_components":N_GMM_COMPONENTS,
        "dataset_cache_mode":DATASET_CACHE_MODE,"dataset_version":DATASET_VERSION
    })
    if RUN_SANITY_CHECKS: flex_realization_sanity_check(case,net,norm,THETA_LIST)
    if RUN_FULL_OPF_EVAL:
        eval_and_plot_flex_domain(case,net,norm,THETA_LIST)
        eval_and_plot_flex_domain_posterior(case,net,norm,THETA_LIST)
        eval_and_plot_realization_cloud(case,net,norm,THETA_LIST)
        eval_and_plot_direction_cdfs(case,net,norm,THETA_LIST)
        eval_pack=eval_all_theta_cdf_arms(case,net,norm,THETA_LIST)
    else:
        print("[eval] RUN_FULL_OPF_EVAL=False, skip OPF-based evaluation plots.")
    print('\n=== Support distribution atom/boundary diagnostic with ARMS ===')
    diagnose_support_distribution_atoms(YH,THETA_FEAT,THETA_LIST,active_records=None,arms_csv_path='all_theta_cdf_arms_v1.csv',save_prefix='support_atom_eval')
    print('\n=== CDF error decomposition diagnostic ===')
    if Path('all_theta_eval_cache.npz').exists():
        d=np.load('all_theta_eval_cache.npz',allow_pickle=True)
        xmu_eval=d['XMU_eval']
        valid=[]; missing=False
        for j in range(THETA_FEAT.shape[0]):
            key=f'YH_theta_{j}'
            if key not in d:
                print(f"[cdf-error-decomp] warning: missing {key} in eval cache."); missing=True; break
            arr=np.asarray(d[key],dtype=float); arr=arr[np.isfinite(arr)]
            if len(arr)<30:
                print(f"[cdf-error-decomp] warning: {key} valid samples < 30."); missing=True; break
            valid.append(arr)
        if missing:
            print('[cdf-error-decomp] warning: eval cache unavailable, fallback to training YH.')
            XMU_eval=XMU; YH_eval=YH
        else:
            m_min=min(len(v) for v in valid); YH_eval=np.zeros((1,m_min,THETA_FEAT.shape[0]),dtype=float)
            for j in range(THETA_FEAT.shape[0]): YH_eval[0,:,j]=valid[j][:m_min]
            XMU_eval=xmu_eval
    else:
        print('[cdf-error-decomp] warning: eval cache unavailable, fallback to training YH.')
        XMU_eval=XMU; YH_eval=YH
    analyze_cdf_error_decomposition(net=net,norm=norm,XMU_eval=XMU_eval,THETA_FEAT=THETA_FEAT,theta_list=THETA_LIST,YH_eval=YH_eval,atom_diag_csv_path='support_atom_eval_by_theta.csv',active_diag_csv_path='support_atom_eval_active_patterns.csv',save_prefix='cdf_error_decomp')
    if RUN_MULTI_TEST: eval_multiple_flex_scenarios(case,net,norm,THETA_LIST)

if __name__=='__main__':
    main()
