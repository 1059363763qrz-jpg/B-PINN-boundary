# -*- coding: utf-8 -*-
"""
flex-domain v1 support-function B-PINN (experiment-ready v2)
推荐调试顺序:
1) RUN_MODE="smoke": 检查 ready-check/active_records_match/max_abs_h_res/训练无 NaN/inf/support_res_raw≈0
2) RUN_MODE="small": 检查 val_boundary_sup、val_phys_flex、val_raw_pcc_p/q、P0/Q0/t 误差是否下降，CDF 是否可输出
3) RUN_MODE="fast": 正式快速实验
# 若尚未生成缓存，请先设置 DATASET_CACHE_MODE="auto" 或 "rebuild" 跑一次；
# 后续调参固定训练集时，使用 DATASET_CACHE_MODE="load_only"；
# 若只比较神经网络训练结果，RUN_FULL_OPF_EVAL=False；
# 若需要正式出图和外部 MC/OPF 评估，再设置 RUN_FULL_OPF_EVAL=True。
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

RUN_MODE = "data800"
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
elif RUN_MODE == "data800":
    NUM_SCENARIOS=800; MC_PER_SCENARIO=30; N_THETA=12
    RUN_MULTI_TEST=False; MC_EVAL_MULTI=400; EVAL_THETA_SAMPLES=40
    USE_H_QUANTILE_LOSS=True; LAM_H_QUANTILE=0.05
elif RUN_MODE == "full":
    NUM_SCENARIOS, MC_PER_SCENARIO, N_THETA, EPOCHS = 1000, 60, 16, 700
    RUN_MULTI_TEST=True; MC_EVAL_MULTI=400; EVAL_THETA_SAMPLES=40
    USE_H_QUANTILE_LOSS=True; LAM_H_QUANTILE=0.10
else:
    raise ValueError(f"Unsupported RUN_MODE={RUN_MODE}")
THETA_LIST = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)

USE_LOC_SCALE_CALIBRATION = True
CALIB_TAUS = [0.01, 0.05, 0.50, 0.95, 0.99]
LAM_LOC_CAL_A = 0.00
LAM_SCALE_CAL_A = 0.00
LAM_TAIL_CAL_A = 0.00
LAM_MEAN_CAL_A = 0.00
LAM_STD_CAL_A = 0.00
LAM_LOC_CAL_B = 0.08
LAM_SCALE_CAL_B = 0.06
LAM_TAIL_CAL_B = 0.04
LAM_MEAN_CAL_B = 0.04
LAM_STD_CAL_B = 0.04
LAM_LOC_CAL_C = 0.18
LAM_SCALE_CAL_C = 0.14
LAM_TAIL_CAL_C = 0.08
LAM_MEAN_CAL_C = 0.08
LAM_STD_CAL_C = 0.08
USE_STAGE_D_FINE_CALIBRATION = True
LAM_LOC_CAL_D = 0.25
LAM_SCALE_CAL_D = 0.20
LAM_TAIL_CAL_D = 0.12
LAM_MEAN_CAL_D = 0.12
LAM_STD_CAL_D = 0.12
USE_STATIC_ERROR_TYPE_WEIGHTS = True
LOCATION_THETA_INIT = [7, 8, 6, 5, 3, 2, 10, 11, 9]
SCALE_THETA_INIT = [7, 8, 6, 5, 9, 2]
SHAPE_THETA_INIT = [3, 2, 10, 11, 9]
LOC_THETA_WEIGHT = 2.0
SCALE_THETA_WEIGHT = 1.8
SHAPE_THETA_WEIGHT = 1.5
DEFAULT_THETA_WEIGHT = 1.0
EXTRA_HARD_THETA_INIT = [2, 9, 3, 11]
EXTRA_HARD_THETA_WEIGHT = 3.0
MODERATE_HARD_THETA_INIT = [7, 8, 6, 5]
MODERATE_HARD_THETA_WEIGHT = 2.0
THETA_PROBLEM_WEIGHT_MAX = 5.0
USE_THETA_WISE_CALIB_NORMALIZATION = False
THETA_STD_FLOOR = 1e-3
USE_DENSE_QUANTILE_CALIBRATION = False
DENSE_Q_TAUS = np.linspace(0.01, 0.99, 33).tolist()
LAM_DENSE_Q_CAL_A = 0.00
LAM_DENSE_Q_CAL_B = 0.04
LAM_DENSE_Q_CAL_C = 0.10
LAM_DENSE_Q_CAL_D = 0.15
LAM_DENSE_Q_CAL_E = 0.30
LAM_LOC_CAL_E = 0.40
LAM_MEAN_CAL_E = 0.20
LAM_SCALE_CAL_E = 0.30
LAM_STD_CAL_E = 0.20
LAM_TAIL_CAL_E = 0.15
LAM_THETA_AFFINE_REG_E = 5e-4

CALIBRATION_USE_DETERMINISTIC_MEAN = True
USE_THETA_AFFINE_CALIBRATION = True
THETA_AFFINE_SCALE_CLAMP = 0.5
LAM_THETA_AFFINE_REG = 1e-4
USE_VAL_ARMS_DYNAMIC_REWEIGHT = True
VAL_ARMS_REWEIGHT_START_EPOCH = 80
VAL_ARMS_REWEIGHT_UPDATE_EVERY = 20
VAL_ARMS_REWEIGHT_POWER = 1.5
VAL_ARMS_REWEIGHT_MIN = 0.5
VAL_ARMS_REWEIGHT_MAX = 6.0
VAL_ARMS_REWEIGHT_SMOOTH = 0.5
VAL_ARMS_FAST_GRID = 150
FAST_TRAINING = True
QUANTILE_EVERY_N_BATCHES = 5
H_QUANTILE_START_EPOCH = 30
FAST_EPOCHS = 150
if FAST_TRAINING and RUN_MODE == "fast":
    EPOCHS = FAST_EPOCHS
BATCH_SIZE, LR = (4096 if torch.cuda.is_available() else 2048), 1e-3
STAGE_E_LR = LR * 0.05
LAM_BOUNDARY_SUP, LAM_DISPATCH_SUP, LAM_PHYS_FLEX, LAM_SUPPORT_CONSIST, LAM_T_SUP = 0.05, 0.05, 0.08, 0.01, 0.05
H_QUANTILE_TAUS = [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
N_GMM_COMPONENTS = 6
GMM_SIGMA_FLOOR = 1e-3
USE_H_CDF_LOSS = True
Z_CDF_GRID = np.array([-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5],dtype=float)
FAST_STAGE_A_EPOCHS = 60
FAST_STAGE_B_EPOCHS = 60
FAST_STAGE_C_EPOCHS = 100
FAST_STAGE_D_EPOCHS = 120
FAST_TOTAL_EPOCHS = FAST_STAGE_A_EPOCHS + FAST_STAGE_B_EPOCHS + FAST_STAGE_C_EPOCHS + FAST_STAGE_D_EPOCHS
USE_STAGE_E_AFFINE_ONLY_CALIBRATION = False
FAST_STAGE_E_EPOCHS = 80
EPOCHS = FAST_TOTAL_EPOCHS + (FAST_STAGE_E_EPOCHS if USE_STAGE_E_AFFINE_ONLY_CALIBRATION else 0)
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
RUN_SANITY_CHECKS = False
N_TEST_SCENARIOS = 20
SEED_DATA, SEED_TRAIN, SEED_EVAL = 0, 0, 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_DATASET_CACHE = True
DATASET_CACHE_MODE = "auto"
# First run for data800: set DATASET_CACHE_MODE="rebuild" (or keep auto). After cache built, use "load_only".
BUILD_DATASET_ONLY = False  # "auto" | "rebuild" | "load_only"
DATASET_CACHE_DIR = "dataset_cache"
DATASET_CACHE_TAG = "flex33_data800_800scen_30mc_12theta_seed0_v1"
DATASET_CACHE_NPZ = f"{DATASET_CACHE_DIR}/{DATASET_CACHE_TAG}.npz"
DATASET_CACHE_PICKLE = f"{DATASET_CACHE_DIR}/{DATASET_CACHE_TAG}_active.pkl"
DATASET_CACHE_META = f"{DATASET_CACHE_DIR}/{DATASET_CACHE_TAG}_meta.json"
DATASET_VERSION = "flex_support_dataset_v1"
SAVE_TRAINING_RESULT = True
TRAINING_RESULT_DIR = "training_results"
RUN_TRAINING = True
RUN_EVAL_ONLY = False
LOAD_TRAINED_MODEL = False
LOAD_MODEL_PATH = "training_results/expB_v7_extendedCD_best_seed0_model.pt"
LOAD_NORM_PATH = "training_results/expB_v7_extendedCD_best_seed0_norm.pkl"
LOAD_CONFIG_PATH = "training_results/expB_v7_extendedCD_best_seed0_config.json"
TRAINING_RUN_TAG = "v12_data800_multiscen_train"
RUN_FULL_OPF_EVAL = False
MULTI_SCENARIO_FORMAL_EVAL = True
N_FORMAL_EVAL_SCENARIOS = 10
MC_EVAL_PER_SCENARIO = 100
FORMAL_EVAL_REBUILD_CACHE = False
FORMAL_EVAL_CACHE_PATH = "all_theta_eval_cache_multiscen_v9.npz"
SAVE_FORMAL_ACTIVE_SIGNATURES = True
POLYGON_MIN_AREA = 1e-5
POLYGON_DIAG_ROWS = []
POLYGON_JITTER_EPS = 1e-8
POLYGON_DEDUP_TOL = 1e-7
POLYGON_USE_HALFSPACE_FALLBACK = True
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

def gmm_quantile(taus, w, mu, s, n_iter=80):
    taus = np.asarray(taus, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)

    lo = float(np.min(mu - 8.0 * s))
    hi = float(np.max(mu + 8.0 * s))
    out = []

    for tau in taus:
        l, r = lo, hi
        for _ in range(n_iter):
            mid = 0.5 * (l + r)
            cdf = np.sum(w * norm.cdf((mid - mu) / (s + 1e-12)))
            if cdf < tau:
                l = mid
            else:
                r = mid
        out.append(0.5 * (l + r))

    return np.array(out, dtype=float)

def infer_theta_idx_from_feat(theta_feat):
    base=np.stack([np.cos(THETA_LIST),np.sin(THETA_LIST)],axis=1).astype(np.float32)
    bt=torch.as_tensor(base,dtype=theta_feat.dtype,device=theta_feat.device)
    d=((theta_feat.unsqueeze(1)-bt.unsqueeze(0))**2).sum(dim=2)
    return torch.argmin(d,dim=1).long()

class BayesFlexGMM2SupportNet(nn.Module):
    def __init__(self,in_dim,case,hidden=160,depth=3):
        super().__init__(); self.n_gen=len(case.gen_buses); self.x_dim=in_dim
        self.layers=nn.ModuleList(); d=in_dim+2
        for _ in range(depth): self.layers.append(BayesLinear(d,hidden,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO)); d=hidden
        self.gmm_out=BayesLinear(d,3*N_GMM_COMPONENTS,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO)
        self.use_theta_affine_calibration = USE_THETA_AFFINE_CALIBRATION
        self.theta_mu_bias = nn.Parameter(torch.zeros(N_THETA))
        self.theta_log_scale = nn.Parameter(torch.zeros(N_THETA))
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
    def apply_theta_affine_calibration(self, theta_feat, mu, s):
        if not self.use_theta_affine_calibration: return mu,s
        theta_idx=infer_theta_idx_from_feat(theta_feat)
        bias=self.theta_mu_bias[theta_idx].view(-1,1)
        log_scale=torch.clamp(self.theta_log_scale[theta_idx].view(-1,1),-THETA_AFFINE_SCALE_CLAMP,THETA_AFFINE_SCALE_CLAMP)
        scale=torch.exp(log_scale)
        mu_cal=scale*mu+bias
        s_cal=torch.clamp(scale*s,min=GMM_SIGMA_FLOOR)
        return mu_cal,s_cal
    def forward_gmm(self,x_mu_norm,theta_feat,sample=True):
        h=self.encode_gmm(torch.cat([x_mu_norm,theta_feat],dim=1),sample=sample)
        w,mu,s=self.gmm_head(h,sample=sample)
        mu,s=self.apply_theta_affine_calibration(theta_feat,mu,s)
        return h,w,mu,s
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

def compute_h_loc_scale_calibration_loss(net,XMU_scen,THETA_FEAT,QH_CAL_EMP,H_MEAN_EMP,H_STD_EMP,x_mu_mean,x_mu_std,h_mean,h_std,n_scenarios_sample=32,n_thetas_sample=None,sample=True,rng=None,theta_sampling_weights=None,theta_problem_weights=None,taus=CALIB_TAUS,return_parts=False,h_theta_mean=None,h_theta_std=None,use_theta_wise_norm=USE_THETA_WISE_CALIB_NORMALIZATION):
    if (not USE_LOC_SCALE_CALIBRATION) or XMU_scen.shape[0]==0:
        z=torch.tensor(0.0,device=DEVICE); parts={"loc":z,"mean":z,"scale":z,"std":z,"tail":z}
        return parts if return_parts else (z,parts)
    if rng is None: rng=np.random.default_rng(0)
    S,T=XMU_scen.shape[0],THETA_FEAT.shape[0]
    s_idx=rng.choice(S,size=min(n_scenarios_sample,S),replace=False)
    if n_thetas_sample is None or n_thetas_sample>=T: t_idx=np.arange(T,dtype=int)
    else:
        p=None if theta_sampling_weights is None else np.asarray(theta_sampling_weights,dtype=float)/np.sum(theta_sampling_weights)
        t_idx=np.sort(rng.choice(T,size=min(n_thetas_sample,T),replace=False,p=p))
    sg,tg=np.meshgrid(s_idx,t_idx,indexing='ij'); sf,tf=sg.reshape(-1),tg.reshape(-1)
    xmu=((XMU_scen[sf]-x_mu_mean)/(x_mu_std+1e-9)).astype(np.float32); th=THETA_FEAT[tf].astype(np.float32)
    xmu_t=torch.as_tensor(xmu,dtype=torch.float32,device=DEVICE); th_t=torch.as_tensor(th,dtype=torch.float32,device=DEVICE)
    _,w,mu,s=net.forward_gmm(xmu_t,th_t,sample=sample)
    q_emp=QH_CAL_EMP[sf,tf,:]
    if use_theta_wise_norm and (h_theta_mean is not None) and (h_theta_std is not None):
        mref=torch.as_tensor(h_theta_mean[tf].reshape(-1,1),dtype=torch.float32,device=DEVICE)
        sref=torch.as_tensor(h_theta_std[tf].reshape(-1,1),dtype=torch.float32,device=DEVICE)
        hm_t=torch.as_tensor(h_mean,dtype=torch.float32,device=DEVICE); hs_t=torch.as_tensor(h_std,dtype=torch.float32,device=DEVICE)
        mu_phys=hm_t + hs_t*mu; s_phys=hs_t*s
        mu=(mu_phys-mref)/sref; s=torch.clamp(s_phys/sref,min=GMM_SIGMA_FLOOR)
        q_emp_t=torch.as_tensor((q_emp-h_theta_mean[tf].reshape(-1,1))/(h_theta_std[tf].reshape(-1,1)+1e-9),dtype=torch.float32,device=DEVICE)
        m_emp_t=torch.as_tensor((H_MEAN_EMP[sf,tf].reshape(-1,1)-h_theta_mean[tf].reshape(-1,1))/(h_theta_std[tf].reshape(-1,1)+1e-9),dtype=torch.float32,device=DEVICE)
        sd_emp_t=torch.as_tensor((H_STD_EMP[sf,tf].reshape(-1,1)/(h_theta_std[tf].reshape(-1,1)+1e-9)),dtype=torch.float32,device=DEVICE)
    else:
        q_emp_t=torch.as_tensor((q_emp-h_mean)/(h_std+1e-9),dtype=torch.float32,device=DEVICE)
        m_emp_t=torch.as_tensor(((H_MEAN_EMP[sf,tf]).reshape(-1,1)-h_mean)/(h_std+1e-9),dtype=torch.float32,device=DEVICE)
        sd_emp_t=torch.as_tensor(((H_STD_EMP[sf,tf]).reshape(-1,1)/(h_std+1e-9)),dtype=torch.float32,device=DEVICE)
    q_pred=gmm_quantile_torch(w,mu,s,taus)
    q01,q05,q50,q95,q99=[q_pred[:,i:i+1] for i in [0,1,2,3,4]]
    e01,e05,e50,e95,e99=[q_emp_t[:,i:i+1] for i in [0,1,2,3,4]]
    mu_mix=(w*mu).sum(dim=1,keepdim=True); var_mix=(w*(s*s+mu*mu)).sum(dim=1,keepdim=True)-mu_mix*mu_mix; std_mix=torch.sqrt(torch.clamp(var_mix,min=1e-12))
    loc=(q50-e50).pow(2); mean=(mu_mix-m_emp_t).pow(2); scale=((q95-q05)-(e95-e05)).pow(2); std=(std_mix-sd_emp_t).pow(2); tail=(q01-e01).pow(2)+(q05-e05).pow(2)+(q95-e95).pow(2)+(q99-e99).pow(2)
    if theta_problem_weights is not None:
        wt=torch.as_tensor(np.asarray(theta_problem_weights,dtype=np.float32)[tf].reshape(-1,1),dtype=torch.float32,device=DEVICE)
        loc,mean,scale,std,tail=[x*wt for x in [loc,mean,scale,std,tail]]
    parts={"loc":loc.mean(),"mean":mean.mean(),"scale":scale.mean(),"std":std.mean(),"tail":tail.mean()}
    total=parts["loc"]+parts["mean"]+parts["scale"]+parts["std"]+parts["tail"]
    return parts if return_parts else (total,parts)


def safe_update_theta_weights_from_arms(arms,theta_sampling_weights,smooth=VAL_ARMS_REWEIGHT_SMOOTH,power=VAL_ARMS_REWEIGHT_POWER,w_min=VAL_ARMS_REWEIGHT_MIN,w_max=VAL_ARMS_REWEIGHT_MAX):
    arms=np.asarray(arms,dtype=float); finite=np.isfinite(arms)
    if not finite.any():
        print("[val-arms-reweight] warning: no finite validation ARMS; skip update.")
        return theta_sampling_weights,False,None
    arms_safe=arms.copy(); mean_finite=float(np.mean(arms_safe[finite])); arms_safe[~finite]=mean_finite
    raw_w=(arms_safe/(mean_finite+1e-9))**power; raw_w=np.clip(raw_w,w_min,w_max); raw_w=raw_w/(raw_w.mean()+1e-9)
    new_w=smooth*theta_sampling_weights+(1.0-smooth)*raw_w; new_w=new_w/(new_w.mean()+1e-9)
    info={"arms":arms,"raw_weights":raw_w,"new_weights":new_w,"worst3":np.argsort(-arms_safe)[:3].tolist(),"mean":float(np.mean(arms_safe)),"max":float(np.max(arms_safe))}
    return new_w,True,info

def set_trainable_for_stage(net,stage):
    if stage!="E":
        for p in net.parameters(): p.requires_grad=True
        return
    for p in net.parameters(): p.requires_grad=False
    net.theta_mu_bias.requires_grad=True
    net.theta_log_scale.requires_grad=True
    n=sum(int(p.numel()) for p in net.parameters() if p.requires_grad)
    print(f"[stage-trainable] stage=E, trainable_params={n}")
    print("[stage-trainable] only theta affine parameters are trainable=True")

def compute_h_dense_quantile_calibration_loss(net,XMU_scen,THETA_FEAT,QH_DENSE_EMP,x_mu_mean,x_mu_std,h_mean,h_std,h_theta_mean=None,h_theta_std=None,n_scenarios_sample=32,n_thetas_sample=None,sample=False,rng=None,theta_sampling_weights=None,theta_problem_weights=None,taus=DENSE_Q_TAUS,use_theta_wise_norm=USE_THETA_WISE_CALIB_NORMALIZATION):
    if (not USE_DENSE_QUANTILE_CALIBRATION) or XMU_scen.shape[0]==0: return torch.tensor(0.0,device=DEVICE)
    if rng is None: rng=np.random.default_rng(0)
    S,T=XMU_scen.shape[0],THETA_FEAT.shape[0]; scen_idx=rng.choice(S,size=min(n_scenarios_sample,S),replace=False)
    if n_thetas_sample is None or n_thetas_sample>=T: theta_idx=np.arange(T,dtype=int)
    else: theta_idx=rng.choice(T,size=min(n_thetas_sample,T),replace=False,p=(None if theta_sampling_weights is None else theta_sampling_weights/theta_sampling_weights.sum()))
    sg,tg=np.meshgrid(scen_idx,theta_idx,indexing='ij'); sf,tf=sg.reshape(-1),tg.reshape(-1)
    xmu=((XMU_scen[sf]-x_mu_mean)/(x_mu_std+1e-9)).astype(np.float32); th=THETA_FEAT[tf].astype(np.float32)
    xmu_t=torch.as_tensor(xmu,dtype=torch.float32,device=DEVICE); th_t=torch.as_tensor(th,dtype=torch.float32,device=DEVICE)
    _,w,mu,s=net.forward_gmm(xmu_t,th_t,sample=sample)
    q_emp=QH_DENSE_EMP[sf,tf,:]
    if use_theta_wise_norm and h_theta_mean is not None and h_theta_std is not None:
        h_theta_mean_t=torch.as_tensor(h_theta_mean[tf].reshape(-1,1),device=DEVICE,dtype=torch.float32)
        h_theta_std_t=torch.as_tensor(h_theta_std[tf].reshape(-1,1),device=DEVICE,dtype=torch.float32)
        h_mean_t=torch.as_tensor(h_mean,device=DEVICE,dtype=torch.float32); h_std_t=torch.as_tensor(h_std,device=DEVICE,dtype=torch.float32)
        mu_phys=h_mean_t + h_std_t*mu; s_phys=h_std_t*s
        mu=(mu_phys-h_theta_mean_t)/h_theta_std_t; s=torch.clamp(s_phys/h_theta_std_t,min=GMM_SIGMA_FLOOR)
        q_emp=(q_emp-h_theta_mean[tf].reshape(-1,1))/(h_theta_std[tf].reshape(-1,1)+1e-9)
    else:
        q_emp=(q_emp-h_mean)/(h_std+1e-9)
    q_emp_t=torch.as_tensor(q_emp,dtype=torch.float32,device=DEVICE); q_pred=gmm_quantile_torch(w,mu,s,taus); d=(q_pred-q_emp_t).pow(2).mean(dim=1,keepdim=True)
    if theta_problem_weights is not None: d=d*torch.as_tensor(theta_problem_weights[tf].reshape(-1,1),dtype=torch.float32,device=DEVICE)
    return d.mean()
def train_bayes_flex_gmm2(case,XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG):
    rng=np.random.default_rng(SEED_TRAIN); n_scen=XMU.shape[0]; n_val=max(1,int(0.1*n_scen)); idx=rng.permutation(n_scen); tr,va=idx[:-n_val],idx[-n_val:]
    def split(arr): return arr[tr],arr[va]
    XMU_tr,XMU_va=split(XMU); XREAL_tr,XREAL_va=split(XREAL); YH_tr,YH_va=split(YH); YP0_tr,YP0_va=split(YP0); YQ0_tr,YQ0_va=split(YQ0); YPG_tr,YPG_va=split(YPG); YQG_tr,YQG_va=split(YQG)
    xmu_mean=XMU_tr.mean(0,keepdims=True); xmu_std=XMU_tr.std(0,keepdims=True)+1e-9; xr_mean=XREAL_tr.reshape(-1,XREAL_tr.shape[-1]).mean(0,keepdims=True); xr_std=XREAL_tr.reshape(-1,XREAL_tr.shape[-1]).std(0,keepdims=True)+1e-9
    h_mean=np.array([[YH_tr.mean()]]); h_std=np.array([[YH_tr.std()+1e-9]]); p0_mean=np.array([[YP0_tr.mean()]]); p0_std=np.array([[YP0_tr.std()+1e-9]]); q0_mean=np.array([[YQ0_tr.mean()]]); q0_std=np.array([[YQ0_tr.std()+1e-9]])
    t_train = -THETA_FEAT[None,None,:,1]*YP0_tr + THETA_FEAT[None,None,:,0]*YQ0_tr; t_mean=np.array([[t_train.mean()]]); t_std=np.array([[t_train.std()+1e-9]])
    QH_EMP_TR=precompute_h_empirical_quantiles(YH_tr,H_QUANTILE_TAUS); QH_EMP_VAL=precompute_h_empirical_quantiles(YH_va,H_QUANTILE_TAUS)
    QH_CAL_EMP_TR=precompute_h_empirical_quantiles(YH_tr,CALIB_TAUS); QH_CAL_EMP_VAL=precompute_h_empirical_quantiles(YH_va,CALIB_TAUS)
    QH_DENSE_EMP_TR=precompute_h_empirical_quantiles(YH_tr,DENSE_Q_TAUS); QH_DENSE_EMP_VAL=precompute_h_empirical_quantiles(YH_va,DENSE_Q_TAUS)
    H_MEAN_EMP_TR=YH_tr.mean(axis=1); H_STD_EMP_TR=YH_tr.std(axis=1)+1e-9; H_MEAN_EMP_VAL=YH_va.mean(axis=1); H_STD_EMP_VAL=YH_va.std(axis=1)+1e-9
    H_THETA_MEAN_EMP_TR=YH_tr.mean(axis=(0,1)); H_THETA_STD_EMP_TR=YH_tr.std(axis=(0,1))+THETA_STD_FLOOR
    H_THETA_MEAN_EMP_VAL=YH_va.mean(axis=(0,1)); H_THETA_STD_EMP_VAL=YH_va.std(axis=(0,1))+THETA_STD_FLOOR
    print(f"[dense-q] QH_DENSE_EMP_TR shape={QH_DENSE_EMP_TR.shape}"); print(f"[dense-q] QH_DENSE_EMP_VAL shape={QH_DENSE_EMP_VAL.shape}")
    print(f"[theta-wise-norm] H_THETA_MEAN_EMP_TR={np.round(H_THETA_MEAN_EMP_TR,4).tolist()}")
    print(f"[theta-wise-norm] H_THETA_STD_EMP_TR={np.round(H_THETA_STD_EMP_TR,4).tolist()}")
    trf=flatten_flex_dataset(XMU_tr,XREAL_tr,THETA_FEAT,YH_tr,YP0_tr,YQ0_tr,YPG_tr,YQG_tr); vaf=flatten_flex_dataset(XMU_va,XREAL_va,THETA_FEAT,YH_va,YP0_va,YQ0_va,YPG_va,YQG_va)
    xmu_f,xr_f,th_f,yh_f,yp0_f,yq0_f,ypg_f,yqg_f,yt_f=trf; xmu_v,xr_v,th_v,yh_v,yp0_v,yq0_v,ypg_v,yqg_v,yt_v=vaf
    to=lambda a:torch.tensor(a,dtype=torch.float32,device=DEVICE)
    xmu_t,xr_t,th_t,yh_t,yp0_t,yq0_t,ypg_t,yqg_t,yt_t=map(to,[ (xmu_f-xmu_mean)/xmu_std, (xr_f-xr_mean)/xr_std, th_f,yh_f,yp0_f,yq0_f,ypg_f,yqg_f,yt_f ])
    xr_raw_t=to(xr_f); xmu_tv,xr_tv,th_tv,yh_tv,yp0_tv,yq0_tv,ypg_tv,yqg_tv,yt_tv=map(to,[ (xmu_v-xmu_mean)/xmu_std, (xr_v-xr_mean)/xr_std, th_v,yh_v,yp0_v,yq0_v,ypg_v,yqg_v,yt_v ]); xr_raw_v=to(xr_v)
    net=BayesFlexGMM2SupportNet(XMU.shape[1],case).to(DEVICE); n=xmu_t.shape[0]; nb=(n+BATCH_SIZE-1)//BATCH_SIZE
    hm,hs,p0m,p0s,q0m,q0s,tm,ts=to(h_mean),to(h_std),to(p0_mean),to(p0_std),to(q0_mean),to(q0_std),to(t_mean),to(t_std)
    T=THETA_FEAT.shape[0]; theta_sampling_weights=np.ones(T,dtype=float); theta_problem_weights=np.ones(T,dtype=float)*DEFAULT_THETA_WEIGHT
    for j in LOCATION_THETA_INIT:
        if 0<=j<T: theta_problem_weights[j]=max(theta_problem_weights[j],LOC_THETA_WEIGHT)
    for j in SCALE_THETA_INIT:
        if 0<=j<T: theta_problem_weights[j]=max(theta_problem_weights[j],SCALE_THETA_WEIGHT)
    for j in SHAPE_THETA_INIT:
        if 0<=j<T: theta_problem_weights[j]=max(theta_problem_weights[j],SHAPE_THETA_WEIGHT)
    for j in EXTRA_HARD_THETA_INIT:
        if 0<=j<T: theta_problem_weights[j]=max(theta_problem_weights[j],EXTRA_HARD_THETA_WEIGHT)
    for j in MODERATE_HARD_THETA_INIT:
        if 0<=j<T: theta_problem_weights[j]=max(theta_problem_weights[j],MODERATE_HARD_THETA_WEIGHT)
    theta_problem_weights=np.clip(theta_problem_weights,0.0,THETA_PROBLEM_WEIGHT_MAX)
    print("[theta-problem-weights]")
    for jj in range(T):
        bt="default"
        if jj in LOCATION_THETA_INIT: bt="location"
        if jj in SCALE_THETA_INIT: bt="scale"
        if jj in SHAPE_THETA_INIT: bt="shape"
        if jj in MODERATE_HARD_THETA_INIT: bt="moderate_hard"
        if jj in EXTRA_HARD_THETA_INIT: bt="extra_hard"
        print(f"{jj},{bt},{float(theta_problem_weights[jj]):.3f}")
    theta_problem_weights/=theta_problem_weights.mean()+1e-9; combined_theta_weights=theta_problem_weights.copy()
    best_state=None; best_score=np.inf; no_improve=0; patience=25; prev_stage=None; opt=None; val_arms_mean_recent=100.0; val_arms_max_recent=100.0
    for ep in range(EPOCHS):
        if ep+1<=FAST_STAGE_A_EPOCHS: stage='A'
        elif ep+1<=FAST_STAGE_A_EPOCHS+FAST_STAGE_B_EPOCHS: stage='B'
        elif ep+1<=FAST_STAGE_A_EPOCHS+FAST_STAGE_B_EPOCHS+FAST_STAGE_C_EPOCHS: stage='C'
        elif (not USE_STAGE_E_AFFINE_ONLY_CALIBRATION) or ep+1<=FAST_STAGE_A_EPOCHS+FAST_STAGE_B_EPOCHS+FAST_STAGE_C_EPOCHS+FAST_STAGE_D_EPOCHS: stage='D'
        else: stage='E'
        if stage!=prev_stage:
            set_trainable_for_stage(net,stage)
            cur_lr=LR if stage=='A' else (LR*0.3 if stage=='B' else (LR*0.15 if stage=='C' else (LR*0.08 if stage=='D' else STAGE_E_LR)))
            opt=torch.optim.Adam(filter(lambda p:p.requires_grad,net.parameters()),lr=cur_lr)
            prev_stage=stage
        cur_q_scen=8 if stage=='A' else (32 if stage=='B' else min(64,max(1,XMU_tr.shape[0]))); cur_q_theta=4 if stage=='A' else None
        cur_lam_hq=0.03 if stage=='A' else (0.15 if stage=='B' else (0.20 if stage in ['C','D'] else 0.0)); cur_lam_hcdf=0.0 if stage=='A' else (0.15 if stage=='B' else (0.25 if stage=='C' else (0.30 if stage=='D' else 0.0)))
        cur_lam_dense_q={'A':LAM_DENSE_Q_CAL_A,'B':LAM_DENSE_Q_CAL_B,'C':LAM_DENSE_Q_CAL_C,'D':LAM_DENSE_Q_CAL_D,'E':LAM_DENSE_Q_CAL_E}[stage]
        cur_lam_loc={'A':LAM_LOC_CAL_A,'B':LAM_LOC_CAL_B,'C':LAM_LOC_CAL_C,'D':LAM_LOC_CAL_D,'E':LAM_LOC_CAL_E}[stage]
        cur_lam_mean={'A':LAM_MEAN_CAL_A,'B':LAM_MEAN_CAL_B,'C':LAM_MEAN_CAL_C,'D':LAM_MEAN_CAL_D,'E':LAM_MEAN_CAL_E}[stage]
        cur_lam_scale={'A':LAM_SCALE_CAL_A,'B':LAM_SCALE_CAL_B,'C':LAM_SCALE_CAL_C,'D':LAM_SCALE_CAL_D,'E':LAM_SCALE_CAL_E}[stage]
        cur_lam_std={'A':LAM_STD_CAL_A,'B':LAM_STD_CAL_B,'C':LAM_STD_CAL_C,'D':LAM_STD_CAL_D,'E':LAM_STD_CAL_E}[stage]
        cur_lam_tail={'A':LAM_TAIL_CAL_A,'B':LAM_TAIL_CAL_B,'C':LAM_TAIL_CAL_C,'D':LAM_TAIL_CAL_D,'E':LAM_TAIL_CAL_E}[stage]
        cur_lam_boundary=0.0 if stage=='E' else (0.4*LAM_BOUNDARY_SUP if stage=='D' else LAM_BOUNDARY_SUP)
        cur_lam_dispatch=0.0 if stage=='E' else (0.4*LAM_DISPATCH_SUP if stage=='D' else LAM_DISPATCH_SUP)
        cur_lam_phys=0.0 if stage=='E' else (0.4*LAM_PHYS_FLEX if stage=='D' else LAM_PHYS_FLEX)
        perm=rng.permutation(n)
        for b in range(nb):
            ii=perm[b*BATCH_SIZE:min((b+1)*BATCH_SIZE,n)]
            xmu,xr,xr_raw,th,yh,yp0,yq0,ypg,yqg,yt=xmu_t[ii],xr_t[ii],xr_raw_t[ii],th_t[ii],yh_t[ii],yp0_t[ii],yq0_t[ii],ypg_t[ii],yqg_t[ii],yt_t[ii]
            henc,w,mu,s=net.forward_gmm(xmu,th,sample=True); nll=(-gmm_log_prob((yh-hm)/(hs+1e-9),w,mu,s)).mean()
            p0h,q0h,pgh,qgh,thh=net.recover_boundary_dispatch_from_h_theta(henc,xr,th,yh,hm,hs,tm,ts,sample=True)
            bsup=((p0h-yp0)/p0s).pow(2).mean()+((q0h-yq0)/q0s).pow(2).mean(); dsup=((pgh-ypg)/(net.pg_max_t-net.pg_min_t+1e-6)).pow(2).mean(); phys=physics_loss_flex(case,xr_raw,p0h,q0h,pgh,qgh,p_scale=p0s,q_scale=q0s); scons=(((th[:,0:1]*p0h+th[:,1:2]*q0h-yh)/(hs+1e-9))**2).mean()
            hq=torch.tensor(0.0,device=DEVICE); hcdf=torch.tensor(0.0,device=DEVICE)
            h_loc_cal=torch.tensor(0.0,device=DEVICE); h_mean_cal=torch.tensor(0.0,device=DEVICE); h_scale_cal=torch.tensor(0.0,device=DEVICE); h_std_cal=torch.tensor(0.0,device=DEVICE); h_tail_cal=torch.tensor(0.0,device=DEVICE)
            h_dense_q_cal=torch.tensor(0.0,device=DEVICE)
            _,parts=compute_h_loc_scale_calibration_loss(net,XMU_tr,THETA_FEAT,QH_CAL_EMP_TR,H_MEAN_EMP_TR,H_STD_EMP_TR,xmu_mean,xmu_std,h_mean,h_std,n_scenarios_sample=cur_q_scen,n_thetas_sample=cur_q_theta,sample=False,rng=np.random.default_rng(SEED_TRAIN+300000*ep+b),theta_sampling_weights=theta_sampling_weights,theta_problem_weights=combined_theta_weights,taus=CALIB_TAUS,return_parts=False,h_theta_mean=H_THETA_MEAN_EMP_TR,h_theta_std=H_THETA_STD_EMP_TR,use_theta_wise_norm=USE_THETA_WISE_CALIB_NORMALIZATION)
            h_loc_cal,h_mean_cal,h_scale_cal,h_std_cal,h_tail_cal=parts['loc'],parts['mean'],parts['scale'],parts['std'],parts['tail']
            if cur_lam_dense_q>0:
                h_dense_q_cal=compute_h_dense_quantile_calibration_loss(net,XMU_tr,THETA_FEAT,QH_DENSE_EMP_TR,xmu_mean,xmu_std,h_mean,h_std,h_theta_mean=H_THETA_MEAN_EMP_TR,h_theta_std=H_THETA_STD_EMP_TR,n_scenarios_sample=cur_q_scen,n_thetas_sample=cur_q_theta,sample=False,rng=np.random.default_rng(SEED_TRAIN+400000*ep+b),theta_sampling_weights=theta_sampling_weights,theta_problem_weights=combined_theta_weights,taus=DENSE_Q_TAUS,use_theta_wise_norm=USE_THETA_WISE_CALIB_NORMALIZATION)
            theta_affine_reg=(net.theta_mu_bias.pow(2).mean()+net.theta_log_scale.pow(2).mean())
            if stage=='E':
                loss=LAM_LOC_CAL_E*h_loc_cal+LAM_MEAN_CAL_E*h_mean_cal+LAM_SCALE_CAL_E*h_scale_cal+LAM_STD_CAL_E*h_std_cal+LAM_TAIL_CAL_E*h_tail_cal+LAM_DENSE_Q_CAL_E*h_dense_q_cal+LAM_THETA_AFFINE_REG_E*theta_affine_reg
            else:
                loss=nll+cur_lam_boundary*bsup+cur_lam_dispatch*dsup+cur_lam_phys*phys+LAM_SUPPORT_CONSIST*scons+cur_lam_loc*h_loc_cal+cur_lam_mean*h_mean_cal+cur_lam_scale*h_scale_cal+cur_lam_std*h_std_cal+cur_lam_tail*h_tail_cal+cur_lam_dense_q*h_dense_q_cal
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            _,calv=compute_h_loc_scale_calibration_loss(net,XMU_va,THETA_FEAT,QH_CAL_EMP_VAL,H_MEAN_EMP_VAL,H_STD_EMP_VAL,xmu_mean,xmu_std,h_mean,h_std,n_scenarios_sample=min(cur_q_scen,max(1,XMU_va.shape[0])),n_thetas_sample=cur_q_theta,sample=False,rng=np.random.default_rng(SEED_TRAIN+8888),theta_problem_weights=combined_theta_weights,taus=CALIB_TAUS,return_parts=False,h_theta_mean=H_THETA_MEAN_EMP_VAL,h_theta_std=H_THETA_STD_EMP_VAL,use_theta_wise_norm=USE_THETA_WISE_CALIB_NORMALIZATION)
            val_loc_cal,val_scale_cal,val_tail_cal=calv['loc'],calv['scale'],calv['tail']
            val_dense_q_cal=compute_h_dense_quantile_calibration_loss(net,XMU_va,THETA_FEAT,QH_DENSE_EMP_VAL,xmu_mean,xmu_std,h_mean,h_std,h_theta_mean=H_THETA_MEAN_EMP_VAL,h_theta_std=H_THETA_STD_EMP_VAL,n_scenarios_sample=min(cur_q_scen,max(1,XMU_va.shape[0])),sample=False,theta_problem_weights=combined_theta_weights,taus=DENSE_Q_TAUS,use_theta_wise_norm=USE_THETA_WISE_CALIB_NORMALIZATION)
            bsv=torch.tensor(0.0,device=DEVICE); phv=torch.tensor(0.0,device=DEVICE); hqv=torch.tensor(0.0,device=DEVICE); hcdfv=torch.tensor(0.0,device=DEVICE)
        if USE_VAL_ARMS_DYNAMIC_REWEIGHT and (ep+1)>=VAL_ARMS_REWEIGHT_START_EPOCH and ((ep+1)%VAL_ARMS_REWEIGHT_UPDATE_EVERY==0) and stage in ['C','D','E']:
            arms=compute_cached_val_arms_by_theta_fast(net,{"x_mu_mean":xmu_mean,"x_mu_std":xmu_std,"h_mean":h_mean,"h_std":h_std},XMU_va,THETA_FEAT,THETA_LIST,YH_va,n_grid=VAL_ARMS_FAST_GRID,max_scenarios=30,sample=False)
            theta_sampling_weights,updated,info=safe_update_theta_weights_from_arms(arms,theta_sampling_weights)
            if updated:
                combined_theta_weights=theta_problem_weights*theta_sampling_weights; combined_theta_weights/=combined_theta_weights.mean()+1e-9
                val_arms_mean_recent=info['mean']; val_arms_max_recent=info['max']
        vscore=1.0*val_arms_mean_recent/100+0.6*val_arms_max_recent/100+float(hcdfv)+0.5*float(hqv)+float(val_loc_cal)+float(val_scale_cal)+0.5*float(val_tail_cal)+float(val_dense_q_cal)+0.2*float(bsv)+0.2*float(phv)
        if vscore<best_score: best_score=vscore; best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; no_improve=0
        else: no_improve+=1
        if no_improve>=patience: break
    if best_state is not None: net.load_state_dict(best_state)
    norm={'x_mu_mean':xmu_mean,'x_mu_std':xmu_std,'x_real_mean':xr_mean,'x_real_std':xr_std,'h_mean':h_mean,'h_std':h_std,'p0_mean':p0_mean,'p0_std':p0_std,'q0_mean':q0_mean,'q0_std':q0_std,'t_mean':t_mean,'t_std':t_std}
    eval_cached_validation_cdf_arms(net,norm,XMU_va,THETA_FEAT,THETA_LIST,YH_va,save_prefix="cached_val_cdf_arms_v8")
    return net,norm,{"XMU_val":XMU_va,"YH_val":YH_va,"THETA_FEAT":THETA_FEAT,"theta_list":THETA_LIST,"cached_val_arms_csv":"cached_val_cdf_arms_v8.csv"}

def compute_cached_val_arms_by_theta_fast(net,norm,XMU_val,THETA_FEAT,theta_list,YH_val,n_grid=VAL_ARMS_FAST_GRID,max_scenarios=30,sample=False):
    hm,hs=float(norm['h_mean'][0,0]),float(norm['h_std'][0,0]); T=len(theta_list); idx=np.arange(XMU_val.shape[0])
    if len(idx)>max_scenarios: idx=np.random.default_rng(SEED_TRAIN).choice(idx,size=max_scenarios,replace=False)
    out=np.zeros(T,dtype=float)
    for j,th in enumerate(theta_list):
        vals=[]
        for i in idx:
            ys=np.sort(YH_val[i,:,j]); z=np.linspace(ys.min()-0.2,ys.max()+0.2,n_grid); cdf_mc=np.searchsorted(ys,z,side='right')/len(ys)
            xt=torch.as_tensor((XMU_val[i:i+1]-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE); tt=torch.as_tensor(THETA_FEAT[j:j+1],dtype=torch.float32,device=DEVICE)
            with torch.no_grad(): _,w,mu,s=net.forward_gmm(xt,tt,sample=sample)
            vals.append(100*math.sqrt(np.mean((gmm_cdf((z-hm)/(hs+1e-9),w.cpu().numpy().reshape(-1),mu.cpu().numpy().reshape(-1),s.cpu().numpy().reshape(-1))-cdf_mc)**2)))
        out[j]=float(np.mean(vals)) if vals else 0.0
    return out

def eval_cached_validation_cdf_arms(net,norm,XMU_val,THETA_FEAT,theta_list,YH_val,save_prefix="cached_val_cdf_arms_v8",n_grid=300,n_posterior_samples=0):
    hm,hs=float(norm['h_mean'][0,0]),float(norm['h_std'][0,0]); rows=[]
    for j,th in enumerate(theta_list):
        arms_l=[]; ks_l=[]; q05_l=[]; q50_l=[]; q95_l=[]; mean_l=[]; std_l=[]; sp_l=[]; sr_l=[]
        for i in range(XMU_val.shape[0]):
            ys=np.sort(YH_val[i,:,j]); z=np.linspace(ys.min()-0.2,ys.max()+0.2,n_grid); cdf_mc=np.searchsorted(ys,z,side='right')/len(ys)
            xt=torch.as_tensor((XMU_val[i:i+1]-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE); th_t=torch.as_tensor(THETA_FEAT[j:j+1],dtype=torch.float32,device=DEVICE)
            with torch.no_grad(): _,w,mu,s=net.forward_gmm(xt,th_t,sample=False)
            wp,mp,sp=w.cpu().numpy().reshape(-1),mu.cpu().numpy().reshape(-1),s.cpu().numpy().reshape(-1)
            cdf_bp=gmm_cdf((z-hm)/(hs+1e-9),wp,mp,sp); arms_l.append(100*math.sqrt(np.mean((cdf_bp-cdf_mc)**2))); ks_l.append(np.max(np.abs(cdf_bp-cdf_mc)))
            q50_bp=float(gmm_quantile(np.array([0.5]),wp,mp,sp)[0]*hs+hm); q05_bp=float(gmm_quantile(np.array([0.05]),wp,mp,sp)[0]*hs+hm); q95_bp=float(gmm_quantile(np.array([0.95]),wp,mp,sp)[0]*hs+hm)
            q50_mc=float(np.quantile(ys,0.5)); q05_mc=float(np.quantile(ys,0.05)); q95_mc=float(np.quantile(ys,0.95)); q05_l.append(q05_bp-q05_mc); q50_l.append(q50_bp-q50_mc); q95_l.append(q95_bp-q95_mc); sp_l.append((q95_bp-q05_bp)-(q95_mc-q05_mc))
            mu_mix=np.sum(wp*mp); var_mix=np.sum(wp*(sp*sp+mp*mp))-mu_mix*mu_mix; mean_bp=mu_mix*hs+hm; std_bp=np.sqrt(max(var_mix,1e-12))*hs; mean_l.append(mean_bp-float(np.mean(ys))); std_l.append(std_bp-float(np.std(ys))); std_mc=np.std(ys); sr_l.append(std_bp/(std_mc+1e-9))
        rows.append({"theta_idx":j,"theta":float(th),"arms_pct":float(np.mean(arms_l)),"ks_stat":float(np.mean(ks_l)),"q05_err":float(np.mean(q05_l)),"q50_err":float(np.mean(q50_l)),"q95_err":float(np.mean(q95_l)),"span95_err":float(np.mean(sp_l)),"mean_err":float(np.mean(mean_l)),"std_err":float(np.mean(std_l)),"std_ratio":float(np.mean(sr_l)),"primary_hint":"location" if abs(float(np.mean(q50_l)))>abs(float(np.mean(sp_l))) else "scale"})
    with open(f"{save_prefix}.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["theta_idx","theta","arms_pct","ks_stat","q50_err","span95_err","std_ratio","q05_err","q95_err","mean_err","std_err","primary_hint"]); w.writeheader(); w.writerows(rows)
    vals=np.array([r['arms_pct'] for r in rows]); worst=np.argsort(-vals)[:3].tolist(); print(f"[cached-val-arms-v8] mean={vals.mean():.4f}"); print(f"[cached-val-arms-v8] median={np.median(vals):.4f}"); print(f"[cached-val-arms-v8] max={vals.max():.4f}"); print(f"[cached-val-arms-v8] worst3={worst}")
    colors=['#6b7280']*len(rows)
    for wi in worst: colors[wi]='#dc2626'
    plt.figure(figsize=(9,4.2),dpi=220); plt.bar([r['theta_idx'] for r in rows],[r['arms_pct'] for r in rows],color=colors); plt.axhline(vals.mean(),color='#1d4ed8',ls='--',lw=1.5,label='mean ARMS'); plt.xlabel('theta_idx'); plt.ylabel('ARMS (%)'); plt.title('Cached validation CDF ARMS of support-function distribution'); plt.legend(); plt.tight_layout(); plt.savefig(f"{save_prefix}_bar_v8.png",dpi=260); plt.close()
    return rows

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

def eval_all_theta_cdf_arms_multiscenario(case,net,norm,theta_list,n_eval_scenarios=10,mc_eval_per_scenario=400,seed=SEED_EVAL,cache_path="all_theta_eval_cache_multiscen_v9.npz",rebuild_cache=True,save_prefix="all_theta_multiscen_v12"):
    if (not rebuild_cache) and Path(cache_path).exists():
        d=np.load(cache_path,allow_pickle=True); XMU_eval,YH_eval=d["XMU_eval"],d["YH_eval"]
    else:
        rng=np.random.default_rng(seed); S,T=n_eval_scenarios,len(theta_list)
        XMU_eval=[]; YH_eval=np.full((S,mc_eval_per_scenario,T),np.nan,dtype=float)
        for s in range(S):
            pdm,qdm,prm,qrm=sample_scenario_means(case,rng); XMU_eval.append(make_feature_vector(case,pdm,prm))
            for m in range(mc_eval_per_scenario):
                pd=pdm.copy(); pr=prm.copy()
                pd[case.pv_buses]*=(1+rng.uniform(-qdm,qdm,size=len(case.pv_buses))); pr[case.pv_buses]*=(1+rng.uniform(-qrm,qrm,size=len(case.pv_buses)))
                for j,th in enumerate(theta_list):
                    sol=solve_flex_support_gurobi_33bus(case,pd,pr,float(np.cos(th)),float(np.sin(th)))
                    if sol is not None: YH_eval[s,m,j]=sol['h']
        XMU_eval=np.array(XMU_eval,dtype=float); np.savez(cache_path,XMU_eval=XMU_eval,YH_eval=YH_eval)
    S,T=YH_eval.shape[0],YH_eval.shape[2]
    arms_matrix=np.full((S,T),np.nan)
    rows=[]
    for sidx in range(S):
        for j,th in enumerate(theta_list):
            m=compute_pair_cdf_metrics_unified(net,norm,XMU_eval[sidx:sidx+1],THETA_FEAT[j:j+1],YH_eval[sidx,:,j],n_grid=300,posterior_samples=EVAL_THETA_SAMPLES,sample_mode="posterior_mean",z_margin_abs=0.2)
            arms_matrix[sidx,j]=m['arms_pct']
            rows.append({'scenario_idx':sidx,'theta_idx':j,'theta':float(th),'arms_pct':float(m['arms_pct']),'ks_stat':float(m['ks_stat'])})
    import pandas as pd
    df=pd.DataFrame(rows); df.to_csv(f'{save_prefix}_by_scenario_theta.csv',index=False)
    by_sc=df.groupby('scenario_idx',as_index=False).agg(mean_arms=('arms_pct','mean'),max_arms=('arms_pct','max'),q90_arms=('arms_pct',lambda x:np.quantile(x,0.9)),worst_theta_idx=('arms_pct',lambda x:int(df.loc[x.index[np.argmax(x.values)],'theta_idx']))); by_sc.to_csv(f'{save_prefix}_by_scenario_summary.csv',index=False)
    by_th=df.groupby(['theta_idx','theta'],as_index=False).agg(mean_arms=('arms_pct','mean'),median_arms=('arms_pct','median'),max_arms=('arms_pct','max'),q90_arms=('arms_pct',lambda x:np.quantile(x,0.9))); by_th.to_csv(f'{save_prefix}_by_theta_summary.csv',index=False)
    print(f"[formal-multiscen] mean={np.nanmean(arms_matrix):.4f} max={np.nanmax(arms_matrix):.4f}")
    return {'XMU_eval':XMU_eval,'YH_eval':YH_eval,'arms_matrix':arms_matrix}


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
        if p is not None and len(p)>2: plt.plot(np.r_[p[:,0],p[0,0]],np.r_[p[:,1],p[0,1]],color='#14b8a6',alpha=0.15,lw=0.8)
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
    if poly_mc is not None and len(poly_mc)>2: plt.plot(np.r_[poly_mc[:,0],poly_mc[0,0]],np.r_[poly_mc[:,1],poly_mc[0,1]],color='#1d4ed8',lw=2.6,label='MC median domain')
    if poly_det is not None and len(poly_det)>2: plt.plot(np.r_[poly_det[:,0],poly_det[0,0]],np.r_[poly_det[:,1],poly_det[0,1]],'--',color='#c2410c',lw=2.6,label='deterministic B-PINN median domain')
    if polys:
        hs_mean=np.mean([np.array([p[i,0]*np.cos(theta_list[i])+p[i,1]*np.sin(theta_list[i]) for i in range(min(len(theta_list),len(p)))]) if len(p)>=len(theta_list) else np.zeros(len(theta_list)) for p in polys],axis=0)
        poly_pm=support_values_to_polygon(theta_list,hs_mean)
        if poly_pm is not None and len(poly_pm)>2: plt.plot(np.r_[poly_pm[:,0],poly_pm[0,0]],np.r_[poly_pm[:,1],poly_pm[0,1]],color='#0f766e',lw=3,label='posterior mean median domain')
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
        if p is not None and len(p)>2: plt.plot(np.r_[p[:,0],p[0,0]],np.r_[p[:,1],p[0,1]],color='#94a3b8',alpha=0.18,lw=0.8)
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
    if poly is None or len(poly)<3: return np.nan
    x=poly[:,0]; y=poly[:,1]
    a=0.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))
    return np.nan if (not np.isfinite(a) or a<POLYGON_MIN_AREA) else a

def support_values_to_polygon(theta_list,h_values,eps=1e-9,do_convex_cleanup=True,diag_meta=None):
    h_values=np.asarray(h_values,dtype=float)

    def _diag(reason,n_points=0,area=np.nan,valid=False):
        meta=diag_meta or {}
        POLYGON_DIAG_ROWS.append({
            'scenario_idx':int(meta.get('scenario_idx',-1)),
            'domain_type':str(meta.get('domain_type','unknown')),
            'quantile':str(meta.get('quantile','unknown')),
            'method':'support_values_to_polygon',
            'n_points':int(n_points),
            'area':float(area) if np.isfinite(area) else np.nan,
            'valid':bool(valid),
            'reason':str(reason)
        })
    if h_values.ndim!=1 or len(h_values)!=len(theta_list): _diag("shape_mismatch"); return None
    if len(theta_list)<3 or (not np.all(np.isfinite(h_values))) or np.std(h_values)<POLYGON_JITTER_EPS: _diag("degenerate_support"); return None
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
    if len(pts)<3: _diag("insufficient_intersections",n_points=len(pts)); return None
    pts=np.array(pts,dtype=float)
    pts=pts[np.all(np.isfinite(pts),axis=1)]
    pts=pts[(np.abs(pts[:,0])<=POLYGON_COORD_MAX)&(np.abs(pts[:,1])<=POLYGON_COORD_MAX)]
    if len(pts)<3: _diag("insufficient_intersections",n_points=len(pts)); return None
    pts=np.unique(np.round(pts,10),axis=0)
    if do_convex_cleanup and len(pts)>=3:
        try:
            from scipy.spatial import ConvexHull
            hull=ConvexHull(pts); pts=pts[hull.vertices]
        except Exception:
            ang=np.arctan2(pts[:,1],pts[:,0]); pts=pts[np.argsort(ang)]
    area=polygon_area(pts)
    if (not np.isfinite(area)) or area>1e4:
        print(f"[polygon] warning: failed to reconstruct valid polygon; skip this domain.")
        _diag("invalid_area",n_points=len(pts),area=area); return None
    _diag("ok",n_points=len(pts),area=area,valid=True)
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
    if meta.get("pg_min") != [float(x) for x in case.pg_min.tolist()]: return False
    if meta.get("pg_max") != [float(x) for x in case.pg_max.tolist()]: return False
    if meta.get("qg_min") != [float(x) for x in case.qg_min.tolist()]: return False
    if meta.get("qg_max") != [float(x) for x in case.qg_max.tolist()]: return False
    if float(meta.get("vmin",-1)) != float(case.vmin): return False
    if float(meta.get("vmax",-1)) != float(case.vmax): return False
    if float(meta.get("fmax_p_first",-1)) != float(case.fmax_p[0]): return False
    if float(meta.get("fmax_q_first",-1)) != float(case.fmax_q[0]): return False
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
    if not (np.all(np.isfinite(XMU)) and np.all(np.isfinite(XREAL)) and np.all(np.isfinite(YH)) and np.all(np.isfinite(YP0)) and np.all(np.isfinite(YQ0)) and np.all(np.isfinite(YPG)) and np.all(np.isfinite(YQG))):
        raise RuntimeError("cached dataset has non-finite values")
    if THETA_FEAT.shape != (N_THETA,2): raise RuntimeError("cached THETA_FEAT shape mismatch")
    if (YH.shape[1],YH.shape[2]) != (MC_PER_SCENARIO,N_THETA): raise RuntimeError("cached YH shape mismatch with MC_PER_SCENARIO/N_THETA")
    if len(active) != YH.shape[0]*MC_PER_SCENARIO*N_THETA: raise RuntimeError("cached active_records count mismatch")
    print(f"[dataset-cache] effective scenarios N={YH.shape[0]}")
    return XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln

def get_or_build_flex_dataset_cache(case):
    Path(DATASET_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    meta_now=build_dataset_cache_meta(case,THETA_LIST)
    exists=Path(DATASET_CACHE_NPZ).exists() and Path(DATASET_CACHE_PICKLE).exists() and Path(DATASET_CACHE_META).exists()
    if DATASET_CACHE_MODE=="load_only" and (not exists):
        raise RuntimeError("data800 cache not found. Please set DATASET_CACHE_MODE=\"rebuild\" or \"auto\" for first run.")
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
            print("[dataset-cache] loaded cached dataset; skip OPF label generation.")
            return load_flex_dataset_cache(True)
    else:
        if DATASET_CACHE_MODE=="load_only":
            raise RuntimeError("dataset cache unavailable or mismatched under load_only mode")
    print("[dataset-cache] building dataset with OPF labels...")
    XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln = generate_flex_dataset(
        case,NUM_SCENARIOS,MC_PER_SCENARIO,THETA_LIST,SEED_DATA
    )
    save_flex_dataset_cache(XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln,meta_now)
    return XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln

def save_training_artifacts(net,norm,config_dict):
    if not SAVE_TRAINING_RESULT: return
    Path(TRAINING_RESULT_DIR).mkdir(parents=True,exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    base=f"{TRAINING_RESULT_DIR}/{TRAINING_RUN_TAG}_{ts}"
    mp=f"{base}_model.pt"; npth=f"{base}_norm.pkl"; cp=f"{base}_config.json"
    torch.save(net.state_dict(),mp)
    with open(npth,"wb") as f: pickle.dump(norm,f)
    with open(cp,"w",encoding="utf-8") as f: json.dump(config_dict,f,ensure_ascii=False,indent=2)
    print(f"[train-save] model saved to {mp}")
    print(f"[train-save] norm saved to {npth}")
    print(f"[train-save] config saved to {cp}")

def load_trained_flex_model(case,model_path,norm_path):
    mp, npth = Path(model_path), Path(norm_path)
    if (not mp.exists()) or (not npth.exists()):
        raise FileNotFoundError(f"missing checkpoint files: model={model_path}, norm={norm_path}. Please run training first.")
    with open(npth,"rb") as f: norm=pickle.load(f)
    in_dim=int(norm['x_mu_mean'].shape[1])
    net=BayesFlexGMM2SupportNet(in_dim,case).to(DEVICE)
    net.load_state_dict(torch.load(mp,map_location=DEVICE))
    net.eval()
    print(f"[eval-only] loaded model from {model_path}")
    print(f"[eval-only] loaded norm from {norm_path}")
    print("[eval-only] RUN_TRAINING=False, skip training.")
    return net,norm


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
    arms_arr=np.asarray(arms,dtype=float); finite=np.isfinite(arms_arr)
    if finite.any():
        ax2=plt.gca().twinx(); ax2.plot(xs,arms,'-o',color='#1d4ed8',alpha=0.8,label='ARMS')
    else:
        print("[atom-diagnostic] no finite ARMS overlay; skip.")
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
                th_t=torch.as_tensor(THETA_FEAT[j:j+1],dtype=torch.float32,device=DEVICE)
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
            xt=torch.tensor((xmu0-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE); th_t=torch.as_tensor(THETA_FEAT[j:j+1],dtype=torch.float32,device=DEVICE)
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
            xmu=XMU_eval.mean(axis=0,keepdims=True); xt=torch.tensor((xmu-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE); th_t=torch.as_tensor(THETA_FEAT[j:j+1],dtype=torch.float32,device=DEVICE)
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
    cnt_dict = {k: len(v) for k, v in grp.items()}
    print(f"error type counts = {cnt_dict}")
    print(f"theta requiring Atom-GMM = {[r['theta_idx'] for r in rows if r['primary_error_type'] in ['boundary_atom_error','possible_atom_error']]}")
    print(f"theta requiring location calibration = {[r['theta_idx'] for r in rows if r['primary_error_type']=='location_shift_error']}")
    print(f"theta requiring scale calibration = {[r['theta_idx'] for r in rows if r['primary_error_type']=='scale_mismatch_error']}")
    print(f"theta requiring tail calibration = {[r['theta_idx'] for r in rows if r['primary_error_type']=='tail_mismatch_error']}")
    print(f"theta requiring active-pattern/critical-region conditioning = {[r['theta_idx'] for r in rows if r['primary_error_type']=='shape_mismatch_error']}")
    hi=[r for r in rows if r['arms_pct']>=10]
    if hi and sum(r['primary_error_type']=='boundary_atom_error' for r in hi)>=len(hi)/2: print('Atom-GMM/discrete-continuous mixture is likely useful.')
    elif hi and sum(r['primary_error_type'] in ['location_shift_error','scale_mismatch_error'] for r in hi)>=len(hi)/2: print('Atom-GMM alone will not solve the issue; calibration loss or bias/scale correction is needed.')
    elif hi and sum(r['primary_error_type']=='shape_mismatch_error' for r in hi)>=len(hi)/2: print('Consider richer distribution model or active-pattern conditioned mixture.')


def _safe_corr(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float); m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3: return np.nan
    return float(np.corrcoef(a[m],b[m])[0,1])


def compute_pair_cdf_metrics_unified(net,norm,xmu,theta_feat,h_mc,n_grid=300,posterior_samples=None,sample_mode="posterior_mean",z_margin_abs=0.2):
    h_mc=np.asarray(h_mc,dtype=float); h_mc=h_mc[np.isfinite(h_mc)]
    hm=float(norm['h_mean'][0,0]); hs=float(norm['h_std'][0,0])
    z=np.linspace(h_mc.min()-z_margin_abs,h_mc.max()+z_margin_abs,n_grid)
    F_mc=np.searchsorted(np.sort(h_mc),z,side='right')/len(h_mc)
    xt=torch.tensor((xmu-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE)
    th_t=torch.tensor(theta_feat,dtype=torch.float32,device=DEVICE)
    z_norm=(z-hm)/(hs+1e-9)
    if posterior_samples is None: posterior_samples=EVAL_THETA_SAMPLES
    cdfs=[]
    if sample_mode=='deterministic':
        with torch.no_grad(): _,w,mu,s=net.forward_gmm(xt,th_t,sample=False)
        cdfs=[gmm_cdf(z_norm,w.cpu().numpy().reshape(-1),mu.cpu().numpy().reshape(-1),s.cpu().numpy().reshape(-1))]
        w0,mu0,s0=w,mu,s
    elif sample_mode=='single_sample':
        with torch.no_grad(): _,w,mu,s=net.forward_gmm(xt,th_t,sample=True)
        cdfs=[gmm_cdf(z_norm,w.cpu().numpy().reshape(-1),mu.cpu().numpy().reshape(-1),s.cpu().numpy().reshape(-1))]
        with torch.no_grad(): _,w0,mu0,s0=net.forward_gmm(xt,th_t,sample=False)
    else:
        for _ in range(int(posterior_samples)):
            with torch.no_grad(): _,w,mu,s=net.forward_gmm(xt,th_t,sample=True)
            cdfs.append(gmm_cdf(z_norm,w.cpu().numpy().reshape(-1),mu.cpu().numpy().reshape(-1),s.cpu().numpy().reshape(-1)))
        with torch.no_grad(): _,w0,mu0,s0=net.forward_gmm(xt,th_t,sample=False)
    F_bp=np.mean(np.asarray(cdfs),axis=0)
    arms=100*np.sqrt(np.mean((F_bp-F_mc)**2)); ks=np.max(np.abs(F_bp-F_mc)); sb=np.mean(F_bp-F_mc); ab=np.mean(np.abs(F_bp-F_mc))
    qmc=np.quantile(h_mc,[0.05,0.5,0.95]); qn=gmm_quantile_torch(w0,mu0,s0,[0.05,0.5,0.95]).cpu().numpy().reshape(-1); qbp=hm+hs*qn
    q05,q50,q95=(qbp-qmc).tolist(); span95=(qbp[2]-qbp[0])-(qmc[2]-qmc[0])
    mu_mix=float((w0*mu0).sum(dim=1).cpu().numpy()[0]); var_mix=float((w0*(s0*s0+mu0*mu0)).sum(dim=1).cpu().numpy()[0]-mu_mix**2)
    bp_mean=float(hm+hs*mu_mix); bp_std=float(abs(hs)*np.sqrt(max(0.0,var_mix))); mc_mean=float(h_mc.mean()); mc_std=float(h_mc.std())
    q10,q90=np.quantile(h_mc,[0.1,0.9]); m1=z<=q10; m2=(z>q10)&(z<q90); m3=z>=q90
    lower=100*np.sqrt(np.mean((F_bp[m1]-F_mc[m1])**2)) if m1.any() else np.nan
    middle=100*np.sqrt(np.mean((F_bp[m2]-F_mc[m2])**2)) if m2.any() else np.nan
    upper=100*np.sqrt(np.mean((F_bp[m3]-F_mc[m3])**2)) if m3.any() else np.nan
    return dict(arms_pct=float(arms),ks_stat=float(ks),signed_cdf_bias=float(sb),abs_cdf_bias=float(ab),lower_cdf_arms=float(lower),middle_cdf_arms=float(middle),upper_cdf_arms=float(upper),q05_err=float(q05),q50_err=float(q50),q95_err=float(q95),span95_err=float(span95),mean_err=float(bp_mean-mc_mean),std_err=float(bp_std-mc_std),std_ratio=float(bp_std/(mc_std+1e-9)),F_mc=F_mc,F_bp=F_bp,z=z,qbp=qbp,qmc=qmc,bp_mean=bp_mean,bp_std=bp_std,mc_mean=mc_mean,mc_std=mc_std)

def analyze_cdf_error_decomposition_multiscenario(net,norm,XMU_eval,THETA_FEAT,theta_list,YH_eval,save_prefix="cdf_error_decomp_multiscen_v10",cdf_grid_size=300,n_posterior_samples=EVAL_THETA_SAMPLES):
    rows=[]
    for sidx in range(YH_eval.shape[0]):
        for j,th in enumerate(theta_list):
            h_mc=np.asarray(YH_eval[sidx,:,j],dtype=float)
            m=compute_pair_cdf_metrics_unified(net,norm,XMU_eval[sidx:sidx+1],THETA_FEAT[j:j+1],h_mc,n_grid=cdf_grid_size,posterior_samples=EVAL_THETA_SAMPLES,sample_mode="posterior_mean",z_margin_abs=0.2)
            hr=max(1e-9,m['qmc'][2]-m['qmc'][0])
            tags=[]
            if abs(m['q50_err'])>0.1*hr: tags.append('location_shift_error')
            if abs(m['std_ratio']-1)>0.25 or abs(m['span95_err'])>0.15*hr: tags.append('scale_mismatch_error')
            if m['arms_pct']>=8 and not tags: tags.append('shape_mismatch_error')
            if not tags: tags=['well_calibrated']
            rows.append(dict(scenario_idx=sidx,theta_idx=j,theta=float(th),arms_pct=float(m['arms_pct']),ks_stat=float(m['ks_stat']),signed_cdf_bias=float(m['signed_cdf_bias']),abs_cdf_bias=float(m['abs_cdf_bias']),lower_cdf_arms=float(m['lower_cdf_arms']),middle_cdf_arms=float(m['middle_cdf_arms']),upper_cdf_arms=float(m['upper_cdf_arms']),q05_err=float(m['q05_err']),q50_err=float(m['q50_err']),q95_err=float(m['q95_err']),span95_err=float(m['span95_err']),mean_err=float(m['mean_err']),std_err=float(m['std_err']),std_ratio=float(m['std_ratio']),primary_error_type=tags[0],error_type_multi=';'.join(tags)))
    import pandas as pd
    df=pd.DataFrame(rows); df.to_csv(f'{save_prefix}_by_pair.csv',index=False)
    by_theta=df.groupby(['theta_idx','theta'],as_index=False).agg(mean_arms=('arms_pct','mean'),median_arms=('arms_pct','median'),max_arms=('arms_pct','max'),q90_arms=('arms_pct',lambda x:np.quantile(x,0.9)),mean_q50_err=('q50_err','mean'),mean_span95_err=('span95_err','mean'),mean_std_ratio=('std_ratio','mean'))
    by_theta.to_csv(f'{save_prefix}_by_theta.csv',index=False)
    by_sc=df.groupby('scenario_idx',as_index=False).agg(mean_arms=('arms_pct','mean'),max_arms=('arms_pct','max'),q90_arms=('arms_pct',lambda x:np.quantile(x,0.9)),worst_theta_idx=('arms_pct',lambda x:int(df.loc[x.index[np.argmax(x.values)],'theta_idx'])))
    by_sc.to_csv(f'{save_prefix}_by_scenario.csv',index=False)
    w=df.iloc[df['arms_pct'].idxmax()]
    summ=pd.DataFrame([{'overall_mean_arms':df['arms_pct'].mean(),'overall_median_arms':df['arms_pct'].median(),'overall_max_arms':df['arms_pct'].max(),'overall_q90_arms':df['arms_pct'].quantile(0.9),'num_pairs_over_10pct':int((df['arms_pct']>10).sum()),'num_pairs_over_15pct':int((df['arms_pct']>15).sum()),'num_pairs_over_20pct':int((df['arms_pct']>20).sum()),'num_pairs_over_30pct':int((df['arms_pct']>30).sum()),'num_pairs_over_40pct':int((df['arms_pct']>40).sum()),'worst_scenario':int(w.scenario_idx),'worst_theta':int(w.theta_idx),'worst_pair_arms':float(w.arms_pct)}])
    summ.to_csv(f'{save_prefix}_summary.csv',index=False)
    print('[cdf-error-multiscen-summary]'); print(f"overall mean ARMS = {summ.loc[0,'overall_mean_arms']:.4f}"); print(f"overall max ARMS = {summ.loc[0,'overall_max_arms']:.4f}"); print(f"worst scenario = {int(w.scenario_idx)}"); print(f"worst theta = {int(w.theta_idx)}")
    return df,by_theta,by_sc,summ

def diagnose_scenario_coverage_shift(XMU_train,XMU_eval,scenario_summary_csv,norm,save_path='scenario_generalization_diagnostics_v12.csv'):
    import pandas as pd
    tr=(XMU_train-norm['x_mu_mean'])/norm['x_mu_std']; ev=(XMU_eval-norm['x_mu_mean'])/norm['x_mu_std']
    d=np.linalg.norm(ev[:,None,:]-tr[None,:,:],axis=2)
    nd=d.min(axis=1)
    center=np.linalg.norm(ev-tr.mean(axis=0,keepdims=True),axis=1)
    # adaptive radii from train NN distances
    dt=np.linalg.norm(tr[:,None,:]-tr[None,:,:],axis=2)
    np.fill_diagonal(dt,np.inf)
    train_nn=np.min(dt,axis=1)
    r10=float(np.quantile(train_nn,0.10)); r25=float(np.quantile(train_nn,0.25)); r50=float(np.quantile(train_nn,0.50))
    pct=np.array([100*np.mean(train_nn<=x) for x in nd])
    nq10=(d<=r10).sum(axis=1); nq25=(d<=r25).sum(axis=1); nq50=(d<=r50).sum(axis=1)
    sc=pd.read_csv(scenario_summary_csv)
    out=sc[['scenario_idx','mean_arms','max_arms','q90_arms','worst_theta_idx']].copy()
    out['nearest_train_dist']=nd; out['center_dist']=center; out['nearest_dist_percentile']=pct
    out['radius_q10']=r10; out['radius_q25']=r25; out['radius_q50']=r50
    out['num_train_neighbors_q10']=nq10; out['num_train_neighbors_q25']=nq25; out['num_train_neighbors_q50']=nq50
    out.to_csv(save_path,index=False)
    print('[scenario-coverage]'); print(f"corr(mean_arms, nearest_train_dist)={_safe_corr(out.mean_arms,out.nearest_train_dist):.4f}"); print(f"corr(max_arms, nearest_train_dist)={_safe_corr(out.max_arms,out.nearest_train_dist):.4f}"); print(f"corr(mean_arms, center_dist)={_safe_corr(out.mean_arms,out.center_dist):.4f}"); print(f"corr(max_arms, center_dist)={_safe_corr(out.max_arms,out.center_dist):.4f}")

def generate_worst_theta_diagnostics(pair_csv,theta_csv,save_path='worst_theta_diagnostics_v12.csv'):
    import pandas as pd
    df=pd.read_csv(pair_csv); th=pd.read_csv(theta_csv)
    dom=df.groupby('theta_idx')['primary_error_type'].agg(lambda x:x.value_counts().index[0]).reset_index(name='dominant_error_type')
    agg=df.groupby('theta_idx',as_index=False).agg(mean_q50_err=('q50_err','mean'),mean_span95_err=('span95_err','mean'),mean_std_ratio=('std_ratio','mean'))
    out=th.merge(dom,on='theta_idx',how='left').merge(agg,on='theta_idx',how='left')
    over10=df.groupby('theta_idx')['arms_pct'].apply(lambda x:int((x>10).sum())); over15=df.groupby('theta_idx')['arms_pct'].apply(lambda x:int((x>15).sum())); over20=df.groupby('theta_idx')['arms_pct'].apply(lambda x:int((x>20).sum()))
    out['num_scenarios_over_10pct']=out.theta_idx.map(over10).fillna(0).astype(int)
    out['num_scenarios_over_15pct']=out.theta_idx.map(over15).fillna(0).astype(int)
    out['num_scenarios_over_20pct']=out.theta_idx.map(over20).fillna(0).astype(int)
    def act(r):
        if r.dominant_error_type=='location_shift_error': return 'scenario-conditioned bias / active-region-conditioned mean correction'
        if r.dominant_error_type=='scale_mismatch_error': return 'scenario-conditioned scale calibration / CRPS-like scale loss'
        return 'theta-specific GMM head or active-pattern-conditioned mixture'
    out['suggested_action']=out.apply(act,axis=1)
    out.to_csv(save_path,index=False)
    print('[worst-theta-diagnostics]')

def plot_worst_pair_cdfs(pair_csv,net,norm,XMU_eval,THETA_FEAT,theta_list,YH_eval):
    import pandas as pd
    hm=float(norm['h_mean'][0,0]); hs=float(norm['h_std'][0,0])
    df=pd.read_csv(pair_csv).sort_values('arms_pct',ascending=False)
    def one(ax,row):
        sidx=int(row.scenario_idx); j=int(row.theta_idx)
        h_mc=np.asarray(YH_eval[sidx,:,j],dtype=float)
        hr=max(1e-9,h_mc.max()-h_mc.min())
        z=np.linspace(h_mc.min()-0.05*hr,h_mc.max()+0.05*hr,300)
        F_mc=np.searchsorted(np.sort(h_mc),z,side='right')/len(h_mc)
        xt=torch.tensor((XMU_eval[sidx:sidx+1]-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE)
        th_t=torch.tensor(THETA_FEAT[j:j+1],dtype=torch.float32,device=DEVICE)
        with torch.no_grad(): _,w,mu,s=net.forward_gmm(xt,th_t,sample=False)
        F_bp=gmm_cdf((z-hm)/(hs+1e-9),w.cpu().numpy().reshape(-1),mu.cpu().numpy().reshape(-1),s.cpu().numpy().reshape(-1))
        ax.plot(z,F_mc,'k',label='MC'); ax.plot(z,F_bp,'r--',label='B-PINN')
        ax.set_title(f"sc={sidx} th={j} ARMS={row.arms_pct:.2f} q50={row.q50_err:.3f} span95={row.span95_err:.3f} stdR={row.std_ratio:.2f}")
    fig,ax=plt.subplots(1,1,figsize=(7,5),dpi=240); one(ax,df.iloc[0]); ax.legend(); ax.grid(alpha=0.2); plt.tight_layout(); plt.savefig('all_theta_multiscen_v10_worst_pair_cdf.png',dpi=260)
    fig,axs=plt.subplots(2,3,figsize=(14,8),dpi=220)
    for ax,(_,r) in zip(axs.ravel(),df.head(6).iterrows()): one(ax,r); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig('all_theta_multiscen_v10_worst6_cdf_grid.png',dpi=260)

def write_polygon_diag_csv(path='polygon_reconstruction_diagnostics_v10.csv'):
    import pandas as pd
    cols=['scenario_idx','domain_type','quantile','method','n_points','area','valid','reason']
    pd.DataFrame(POLYGON_DIAG_ROWS if POLYGON_DIAG_ROWS else [{c:(np.nan if c in ['area'] else -1 if c=='scenario_idx' else False if c=='valid' else 'none') for c in cols}]).to_csv(path,index=False)


def check_multiscen_decomp_consistency(decomp_pair_csv="cdf_error_decomp_multiscen_v10_by_pair.csv",formal_pair_csv="all_theta_multiscen_v9_by_scenario_theta.csv",tolerance_mean=0.5,tolerance_max=2.0):
    import pandas as pd
    if (not Path(decomp_pair_csv).exists()) or (not Path(formal_pair_csv).exists()):
        print('[consistency-check] skipped (missing csv).')
        return np.nan,np.nan,np.nan
    d1=pd.read_csv(decomp_pair_csv); d2=pd.read_csv(formal_pair_csv)
    m=d2[['scenario_idx','theta_idx','arms_pct']].rename(columns={'arms_pct':'arms_formal'}).merge(d1[['scenario_idx','theta_idx','arms_pct']].rename(columns={'arms_pct':'arms_decomp'}),on=['scenario_idx','theta_idx'],how='inner')
    m['arms_diff']=m['arms_decomp']-m['arms_formal']
    m[['scenario_idx','theta_idx','arms_formal','arms_decomp','arms_diff']].to_csv('cdf_error_decomp_consistency_check_v10.csv',index=False)
    mean_abs=float(np.mean(np.abs(m['arms_diff']))) if len(m) else np.nan
    max_abs=float(np.max(np.abs(m['arms_diff']))) if len(m) else np.nan
    corr=float(np.corrcoef(m['arms_formal'],m['arms_decomp'])[0,1]) if len(m)>=3 else np.nan
    print('[consistency-check]'); print(f'mean abs ARMS diff = {mean_abs:.4f}'); print(f'max abs ARMS diff = {max_abs:.4f}'); print(f'corr = {corr:.4f}')
    if (np.isfinite(mean_abs) and mean_abs>tolerance_mean) or (np.isfinite(max_abs) and max_abs>tolerance_max) or (np.isfinite(corr) and corr<0.98):
        print('[consistency-check-warning] v10 decomposition does not match formal all-theta ARMS; check normalization/grid.')
    return mean_abs,max_abs,corr



def compare_v12_against_previous(v12_pair_csv='cdf_error_decomp_multiscen_v12_by_pair.csv'):
    import pandas as pd
    v10_src='all_theta_multiscen_v9_by_scenario_theta.csv' if Path('all_theta_multiscen_v9_by_scenario_theta.csv').exists() else 'cdf_error_decomp_multiscen_v10_by_pair.csv'
    v11_src='cdf_error_decomp_multiscen_v11_by_pair.csv'
    if (not Path(v12_pair_csv).exists()) or (not Path(v10_src).exists()) or (not Path(v11_src).exists()):
        print('[v12-compare] skipped (missing csv).'); return
    v10=pd.read_csv(v10_src); v11=pd.read_csv(v11_src); v12=pd.read_csv(v12_pair_csv)
    c10='arms_pct' if 'arms_pct' in v10.columns else 'arms_v10'
    m=v10[['scenario_idx','theta_idx',c10]].rename(columns={c10:'arms_v10'}).merge(v11[['scenario_idx','theta_idx','arms_pct']].rename(columns={'arms_pct':'arms_v11'}),on=['scenario_idx','theta_idx']).merge(v12[['scenario_idx','theta_idx','arms_pct']].rename(columns={'arms_pct':'arms_v12'}),on=['scenario_idx','theta_idx'])
    m['delta_v12_vs_v10']=m['arms_v12']-m['arms_v10']; m['delta_v12_vs_v11']=m['arms_v12']-m['arms_v11']
    m['improved_vs_v10']=(m['delta_v12_vs_v10']<0).astype(int); m['improved_vs_v11']=(m['delta_v12_vs_v11']<0).astype(int)
    m.to_csv('v12_vs_v10_v11_multiscen_comparison.csv',index=False)
    q90=lambda x: float(np.quantile(x,0.9))
    print('[v12-compare]')
    print(f"mean ARMS v10 = {m['arms_v10'].mean():.4f}")
    print(f"mean ARMS v11 = {m['arms_v11'].mean():.4f}")
    print(f"mean ARMS v12 = {m['arms_v12'].mean():.4f}")
    print(f"max ARMS v10 = {m['arms_v10'].max():.4f}")
    print(f"max ARMS v11 = {m['arms_v11'].max():.4f}")
    print(f"max ARMS v12 = {m['arms_v12'].max():.4f}")
    print(f"q90 ARMS v10 = {q90(m['arms_v10']):.4f}")
    print(f"q90 ARMS v11 = {q90(m['arms_v11']):.4f}")
    print(f"q90 ARMS v12 = {q90(m['arms_v12']):.4f}")
    print(f"num improved vs v10 = {int((m['delta_v12_vs_v10']<0).sum())}")
    print(f"num improved vs v11 = {int((m['delta_v12_vs_v11']<0).sum())}")
    print(f"num worsened vs v11 = {int((m['delta_v12_vs_v11']>0).sum())}")


def finalize_v12_checkpoint(net,norm,cached_stats,external_stats):
    Path(TRAINING_RESULT_DIR).mkdir(parents=True,exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    base=f"{TRAINING_RESULT_DIR}/{TRAINING_RUN_TAG}_{ts}"
    model_path=f"{base}_model.pt"; norm_path=f"{base}_norm.pkl"; config_path=f"{base}_config.json"
    torch.save(net.state_dict(),model_path)
    with open(norm_path,'wb') as f: pickle.dump(norm,f)
    cfg={'timestamp':ts,'run_mode':RUN_MODE,'dataset_cache_tag':DATASET_CACHE_TAG,'num_scenarios':NUM_SCENARIOS,'mc_per_scenario':MC_PER_SCENARIO,'n_theta':N_THETA,'epochs':EPOCHS}
    with open(config_path,'w',encoding='utf-8') as f: json.dump(cfg,f,ensure_ascii=False,indent=2)
    selection_score=float(external_stats['external_mean_arms']+0.30*external_stats['external_q90_arms']+0.20*cached_stats['cached_val_mean_arms'])
    metrics={**cfg,**cached_stats,**external_stats,'selection_score':selection_score,'model_path':model_path,'norm_path':norm_path,'config_path':config_path}
    with open(f"{TRAINING_RESULT_DIR}/v12_data800_multiscen_train_metrics.json",'w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
    best_metrics_path=Path(TRAINING_RESULT_DIR)/'v12_data800_multiscen_train_best_metrics.json'
    best_score=float('inf')
    if best_metrics_path.exists():
        try: best_score=float(json.loads(best_metrics_path.read_text(encoding='utf-8')).get('selection_score',float('inf')))
        except Exception: best_score=float('inf')
    update_best=(selection_score<best_score)
    if update_best:
        shutil.copy2(model_path,f"{TRAINING_RESULT_DIR}/v12_data800_multiscen_train_best_model.pt")
        shutil.copy2(norm_path,f"{TRAINING_RESULT_DIR}/v12_data800_multiscen_train_best_norm.pkl")
        shutil.copy2(config_path,f"{TRAINING_RESULT_DIR}/v12_data800_multiscen_train_best_config.json")
        with open(best_metrics_path,'w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
    print('[v12-checkpoint]')
    print(f"cached_val_mean={cached_stats['cached_val_mean_arms']:.4f}")
    print(f"external_mean={external_stats['external_mean_arms']:.4f}")
    print(f"external_q90={external_stats['external_q90_arms']:.4f}")
    print(f"selection_score={selection_score:.4f}")
    print(f"best_score={best_score:.4f}")
    print(f"update_best={update_best}")

def main():
    case=build_ieee33_case()
    if RUN_SANITY_CHECKS: flex_opf_sanity_check(case)
    print('生成 33 节点灵活域训练数据...')
    print('[v12-run-guide]')
    print('Step 1: if data800 cache does not exist, set DATASET_CACHE_MODE="rebuild" or "auto".')
    print('Step 2: after cache is built, use DATASET_CACHE_MODE="load_only" for repeated training.')
    print('Step 3: external formal OPF is not rebuilt; FORMAL_EVAL_REBUILD_CACHE=False.')
    print('Step 4: this version tests whether increasing independent source-load mean scenarios improves external generalization.')
    print(f'[run-mode] RUN_MODE={RUN_MODE}')
    print(f'[run-mode] NUM_SCENARIOS={NUM_SCENARIOS}, MC_PER_SCENARIO={MC_PER_SCENARIO}, N_THETA={N_THETA}, EPOCHS={EPOCHS}')
    print(f'[run-mode] total_OPF_labels={NUM_SCENARIOS*MC_PER_SCENARIO*N_THETA}')
    print(f'[run-mode] USE_H_QUANTILE_LOSS={USE_H_QUANTILE_LOSS}, LAM_H_QUANTILE={LAM_H_QUANTILE}')
    if USE_DATASET_CACHE:
        XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln = get_or_build_flex_dataset_cache(case)
    else:
        XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln = generate_flex_dataset(
            case,NUM_SCENARIOS,MC_PER_SCENARIO,THETA_LIST,SEED_DATA
        )
    summarize_active_patterns(active,alln,top_k=10)
    if BUILD_DATASET_ONLY:
        print("[dataset-cache] build only finished. Please set DATASET_CACHE_MODE=\"load_only\" and BUILD_DATASET_ONLY=False for training.")
        return
    experiment_ready_check(case,XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active)
    print('\n=== Support distribution atom/boundary diagnostic on training dataset ===')
    diagnose_support_distribution_atoms(YH,THETA_FEAT,THETA_LIST,active_records=active,arms_csv_path=None,save_prefix='support_atom_train')
    if RUN_TRAINING:
        net,norm,train_info=train_bayes_flex_gmm2(case,XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG)
    else:
        if not LOAD_TRAINED_MODEL: raise RuntimeError("RUN_TRAINING=False but LOAD_TRAINED_MODEL=False; no model source.")
        net,norm=load_trained_flex_model(case,LOAD_MODEL_PATH,LOAD_NORM_PATH); train_info={}
        if RUN_EVAL_ONLY:
            cv=eval_cached_validation_cdf_arms(net,norm,XMU,THETA_FEAT,THETA_LIST,YH,save_prefix="cached_val_cdf_arms_v12")
            vals=np.array([r["arms_pct"] for r in cv],dtype=float)
            print(f"[eval-only-cached-val] mean={np.nanmean(vals):.4f} median={np.nanmedian(vals):.4f} max={np.nanmax(vals):.4f} worst3={np.argsort(-vals)[:3].tolist()}")
            Path(TRAINING_RESULT_DIR).mkdir(parents=True,exist_ok=True)
            metrics={'cached_val_mean_arms':float(np.nanmean(vals)),'cached_val_max_arms':float(np.nanmax(vals)),'worst3':np.argsort(-vals)[:3].tolist(),'timestamp':datetime.now().isoformat(),'model_path':LOAD_MODEL_PATH if not RUN_TRAINING else 'trained_in_run'}
            with open('training_results/v12_data800_multiscen_train_metrics.json','w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
            if np.nanmean(vals)>15 or np.nanmax(vals)>25:
                print("[eval-only-warning] loaded model cached validation ARMS is high; formal multiscen evaluation may be poor.")
    if RUN_SANITY_CHECKS: flex_realization_sanity_check(case,net,norm,THETA_LIST)
    if RUN_FULL_OPF_EVAL:
        eval_and_plot_flex_domain(case,net,norm,THETA_LIST)
        eval_and_plot_flex_domain_posterior(case,net,norm,THETA_LIST)
        eval_and_plot_realization_cloud(case,net,norm,THETA_LIST)
        eval_and_plot_direction_cdfs(case,net,norm,THETA_LIST)
        if MULTI_SCENARIO_FORMAL_EVAL:
            eval_pack=eval_all_theta_cdf_arms_multiscenario(case,net,norm,THETA_LIST,n_eval_scenarios=N_FORMAL_EVAL_SCENARIOS,mc_eval_per_scenario=MC_EVAL_PER_SCENARIO,cache_path=FORMAL_EVAL_CACHE_PATH,rebuild_cache=FORMAL_EVAL_REBUILD_CACHE,save_prefix="all_theta_multiscen_v12")
            formal_active_rebuilt = FORMAL_EVAL_REBUILD_CACHE
        else:
            eval_pack=eval_all_theta_cdf_arms(case,net,norm,THETA_LIST)
    else:
        print("[eval] RUN_FULL_OPF_EVAL=False, skip OPF-based evaluation plots.")
    print('\n=== Support distribution atom/boundary diagnostic with ARMS ===')
    arms_path='all_theta_cdf_arms_v1.csv' if Path('all_theta_cdf_arms_v1.csv').exists() else None
    diagnose_support_distribution_atoms(YH,THETA_FEAT,THETA_LIST,active_records=None,arms_csv_path=arms_path,save_prefix='support_atom_eval')
    cached_cv=eval_cached_validation_cdf_arms(net,norm,XMU,THETA_FEAT,THETA_LIST,YH,save_prefix='cached_val_cdf_arms_v12')
    cv_vals=np.array([r['arms_pct'] for r in cached_cv],dtype=float)
    cached_stats={'cached_val_mean_arms':float(np.nanmean(cv_vals)),'cached_val_max_arms':float(np.nanmax(cv_vals)),'cached_val_worst3':np.argsort(-cv_vals)[:3].tolist()}
    print(f"[cached-val-v12] mean={cached_stats['cached_val_mean_arms']:.4f} median={float(np.nanmedian(cv_vals)):.4f} max={cached_stats['cached_val_max_arms']:.4f} worst3={cached_stats['cached_val_worst3']}")
    print('\n=== CDF error decomposition diagnostic ===')
    if MULTI_SCENARIO_FORMAL_EVAL and Path(FORMAL_EVAL_CACHE_PATH).exists():
        d=np.load(FORMAL_EVAL_CACHE_PATH,allow_pickle=True)
        XMU_eval=d['XMU_eval']; YH_eval=d['YH_eval']
    elif Path('all_theta_eval_cache.npz').exists():
        if not RUN_FULL_OPF_EVAL:
            print("[cdf-error-decomp] using existing all_theta_eval_cache.npz; it may come from a previous evaluation run.")
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
    if MULTI_SCENARIO_FORMAL_EVAL and Path(FORMAL_EVAL_CACHE_PATH).exists():
        d=np.load(FORMAL_EVAL_CACHE_PATH,allow_pickle=True); XMU_eval=d['XMU_eval']; YH_eval=d['YH_eval']
        pair_df,_,_,_=analyze_cdf_error_decomposition_multiscenario(net,norm,XMU_eval,THETA_FEAT,THETA_LIST,YH_eval,save_prefix='cdf_error_decomp_multiscen_v12',cdf_grid_size=300,n_posterior_samples=EVAL_THETA_SAMPLES)
        scenario_src='all_theta_multiscen_v9_by_scenario_summary.csv' if Path('all_theta_multiscen_v9_by_scenario_summary.csv').exists() else 'cdf_error_decomp_multiscen_v12_by_scenario.csv'
        diagnose_scenario_coverage_shift(XMU,XMU_eval,scenario_src,norm,save_path='scenario_generalization_diagnostics_v12.csv')
        theta_src='all_theta_multiscen_v9_by_theta_summary.csv' if Path('all_theta_multiscen_v9_by_theta_summary.csv').exists() else 'cdf_error_decomp_multiscen_v12_by_theta.csv'
        generate_worst_theta_diagnostics('cdf_error_decomp_multiscen_v12_by_pair.csv',theta_src,save_path='worst_theta_diagnostics_v12.csv')
        check_multiscen_decomp_consistency('cdf_error_decomp_multiscen_v12_by_pair.csv','all_theta_multiscen_v9_by_scenario_theta.csv')
        plot_worst_pair_cdfs('cdf_error_decomp_multiscen_v12_by_pair.csv',net,norm,XMU_eval,THETA_FEAT,THETA_LIST,YH_eval)
        if Path('formal_active_pattern_records_v10.csv').exists() or FORMAL_EVAL_REBUILD_CACHE:
            print('[active-pattern-novelty] placeholder: records ready for analysis.')
        else:
            print('[active-pattern-novelty] formal active records unavailable; set FORMAL_EVAL_REBUILD_CACHE=True to generate.')
        ext_df=pd.read_csv('cdf_error_decomp_multiscen_v12_by_pair.csv')
        wrow=ext_df.loc[ext_df['arms_pct'].idxmax()]
        external_stats={'external_mean_arms':float(ext_df['arms_pct'].mean()),'external_max_arms':float(ext_df['arms_pct'].max()),'external_q90_arms':float(np.quantile(ext_df['arms_pct'],0.9)),'external_worst_theta':int(wrow['theta_idx']),'external_worst_scenario':int(wrow['scenario_idx'])}
        print('[formal-multiscen-v12]')
        print(f"overall mean={external_stats['external_mean_arms']:.4f}")
        print(f"overall median={float(np.median(ext_df['arms_pct'])):.4f}")
        print(f"overall max={external_stats['external_max_arms']:.4f}")
        print(f"overall q90={external_stats['external_q90_arms']:.4f}")
        print(f"worst scenario={external_stats['external_worst_scenario']}")
        print(f"worst theta={external_stats['external_worst_theta']}")
        if RUN_TRAINING:
            finalize_v12_checkpoint(net,norm,cached_stats,external_stats)
    else:
        analyze_cdf_error_decomposition(net=net,norm=norm,XMU_eval=XMU_eval,THETA_FEAT=THETA_FEAT,theta_list=THETA_LIST,YH_eval=YH_eval,atom_diag_csv_path='support_atom_eval_by_theta.csv',active_diag_csv_path='support_atom_eval_active_patterns.csv',save_prefix='cdf_error_decomp')
    write_polygon_diag_csv()
    if RUN_MULTI_TEST: eval_multiple_flex_scenarios(case,net,norm,THETA_LIST)

if __name__=='__main__':
    main()
