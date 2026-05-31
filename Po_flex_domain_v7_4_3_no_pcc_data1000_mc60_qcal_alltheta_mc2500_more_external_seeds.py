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
import shutil
from datetime import datetime
from typing import List, Dict
import numpy as np
import pandas as pd
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
STABLE_HARD_THETA_INIT = []
STABLE_HARD_THETA_WEIGHT = 1.5
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
VIS_VERSION = "v15_theta_independent"
PLOT_PREFIX = "v15_theta_independent"
POSSIBLE_ATOM_MASS_THRESHOLD = 0.10
HIGH_ARMS_THRESHOLD = 10.0
STABILIZE_OPF_DISPATCH, SUPPORT_EPS, W_QG_STAB, W_PG_STAB, W_PQ0_STAB = True, 1e-5, 1.0, 0.01, 1e-5
PRIOR_SIGMA, INIT_RHO, BETA_KL_MAX, KL_WARMUP_EPOCHS = 1.0, -5.0, 1.0, 500
RUN_SANITY_CHECKS = False
N_TEST_SCENARIOS = 20
SEED_DATA, SEED_TRAIN, SEED_EVAL = 0, 0, 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_DATASET_CACHE = True
DATASET_CACHE_MODE = "load_only"
# First run for data800: set DATASET_CACHE_MODE="rebuild" (or keep auto). After cache built, use "load_only".
BUILD_DATASET_ONLY = False  # "auto" | "rebuild" | "load_only"
DATASET_CACHE_DIR = "dataset_cache_no_pcc_branch_limit_data1000_mc60"
DATASET_CACHE_TAG = "flex33_data1000_1000scen_60mc_12theta_seed0_no_pcc_branch_limit_v1"
DATASET_CACHE_NPZ = f"{DATASET_CACHE_DIR}/{DATASET_CACHE_TAG}.npz"
DATASET_CACHE_PICKLE = f"{DATASET_CACHE_DIR}/{DATASET_CACHE_TAG}_active.pkl"
DATASET_CACHE_META = f"{DATASET_CACHE_DIR}/{DATASET_CACHE_TAG}_meta.json"
LINE_LIMIT_MODE = "internal_only_no_pcc"
USE_EXPLICIT_PCC_LIMITS = False
PCC_P_MIN = -20.0
PCC_P_MAX = 20.0
PCC_Q_MIN = -20.0
PCC_Q_MAX = 20.0
INTERNAL_FMAX_P = 5.0
INTERNAL_FMAX_Q = 5.0
RUN_V7_NO_PCC_PIPELINE = True
BUILD_NEW_DATASET = False
RUN_BRANCH_LIMIT_SANITY = False
RUN_ALL_THETA_TRAIN = False
RUN_ALL_THETA_EVAL = False
RUN_FLEX_SYNTHESIS = False
SMOKE_TEST = False
RUN_TARGETED_DIAGNOSTICS = False
TARGETED_DIAG_DIR = "diagnostics_no_pcc_data1000_mc60_qcal_only"
DIAG_TARGET_PAIRS = [(4,1),(5,1),(4,2),(2,2),(7,2)]
DIAG_MC_EVAL = 2500
DIAG_K_LIST = [20,50]
DIAG_QUANTILES = [0.01,0.05,0.25,0.50,0.75,0.95,0.99]
DIAG_PROGRESS_EVERY = 100
DIAG_CDF_GRID_N = 800

# v7.4 quantile-calibration-only ablation.
# Original v7style single-theta loss:
#   loss_old = nll + LAM_T_SUP*t_sup + LAM_DISPATCH_SUP*disp + LAM_PHYS_FLEX*phys + beta_kl*kl
# where nll is the GMM negative log likelihood on h_theta, t_sup supervises
# t=-beta*P0+alpha*Q0, disp supervises Pg/Qg, phys is quantile physics on recovered
# dispatch, and kl is the Bayesian-layer KL. h_theta remains the only probabilistic
# target; t is an auxiliary recovery variable, not a separate probability model.
RUN_QCAL_ALLTHETA_MORE_SEEDS_EVAL_ONLY = True
RUN_QUANTILE_CALIBRATION_TRAIN = False
EXISTING_QCAL_THETA_LIST = [2,4,5,7]
NEED_TRAIN_THETA_LIST = [0,1,3,6,8,9,10,11]
HARD_THETA_LIST = []
TRAIN_THETA_LIST = []
ALL_THETA_LIST = list(range(N_THETA))
SKIP_QCAL_TRAIN_IF_MODEL_EXISTS = True
USE_SCENARIO_QUANTILE_CALIBRATION = True
SCENARIO_Q_TAUS = [0.05,0.25,0.50,0.75,0.95]
SCENARIO_Q_WEIGHTS = [1.0,1.0,3.0,1.0,1.0]
LAM_SCEN_Q_A = 0.00
LAM_SCEN_Q_B = 0.05
LAM_SCEN_Q_C = 0.12
LAM_SCEN_Q_D = 0.20
USE_SCENARIO_MOMENT_CALIBRATION = True
LAM_SCEN_MEAN = 0.08
LAM_SCEN_STD = 0.04
SCEN_Q_START_EPOCH_B = 60
SCEN_Q_START_EPOCH_C = 140
SCEN_Q_START_EPOCH_D = 240
SCENARIO_CALIB_BATCH_MAX_SCENARIOS = 64
DATASET_VERSION = "flex_support_dataset_v1"
SAVE_TRAINING_RESULT = True
TRAINING_RESULT_DIR = "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_qcal_only"
RUN_TRAINING = True
RUN_EVAL_ONLY = False
LOAD_TRAINED_MODEL = False
LOAD_MODEL_PATH = "training_results/expB_v7_extendedCD_best_seed0_model.pt"
LOAD_NORM_PATH = "training_results/expB_v7_extendedCD_best_seed0_norm.pkl"
LOAD_CONFIG_PATH = "training_results/expB_v7_extendedCD_best_seed0_config.json"
TRAINING_RUN_TAG = "v15_theta_independent_v8_style"
RUN_FULL_OPF_EVAL = False
RUN_FLEX_DOMAIN_PLOTS = False
RUN_P0_STYLE_EXTERNAL_EVAL = True
P0_STYLE_EVAL_THETA_LIST = [2]
# 若只想用 v3 已训练好的 theta02 模型做 P0-style external evaluation，
# 请设置：
# RUN_TRAINING = False
# RUN_EVAL_ONLY = True
# RUN_P0_STYLE_EXTERNAL_EVAL = True
# P0_STYLE_EVAL_THETA_LIST = [2]
# P0_STYLE_SINGLE_SCENARIO_MC = 20  # smoke 测试
# P0_STYLE_MULTI_SCENARIO_EVAL = False
#
# 正式单场景外部验证再改为：
# P0_STYLE_SINGLE_SCENARIO_MC = 2500
P0_STYLE_SINGLE_SCENARIO_MC = 2500
P0_STYLE_MULTI_SCENARIO_EVAL = False
P0_STYLE_N_TEST_SCENARIOS = 20
P0_STYLE_MULTI_SCENARIO_MC = 800
P0_STYLE_EVAL_SEED = SEED_EVAL
P0_STYLE_EXTERNAL_EVAL_DIR = TRAINING_RESULT_DIR
THETA_TRAIN_MODE = "subset"
TRAIN_THETA_LIST = list(range(N_THETA))
DEBUG_THETA_LIST = [6]
RUN_COMBINE_THETA_FLEX_DOMAIN = False
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
    topo_order:List[int]; rev_topo_order:List[int]; in_branches:List[List[int]]; out_branches:List[List[int]]; pcc_branch_mask:np.ndarray; internal_branch_mask:np.ndarray; pcc_branch_indices:np.ndarray; line_limit_mode:str; use_explicit_pcc_limits:bool; pcc_p_min:float; pcc_p_max:float; pcc_q_min:float; pcc_q_max:float

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
    fmax_p=np.full(nl,INTERNAL_FMAX_P); fmax_q=np.full(nl,INTERNAL_FMAX_Q)
    pcc_branch_indices=np.array(outb[0],dtype=int)
    pcc_branch_mask=np.zeros(nl,dtype=bool); pcc_branch_mask[pcc_branch_indices]=True
    internal_branch_mask=~pcc_branch_mask
    return GridCase(nb,0,fb,tb,r,x,0.95,1.05,fmax_p,fmax_q,pd,qd,pv_b,pv_max,0.98,gen_b,pg_min,pg_max,qg_min,qg_max,c,pb,pbu,to,rt,inb,outb,pcc_branch_mask,internal_branch_mask,pcc_branch_indices,LINE_LIMIT_MODE,USE_EXPLICIT_PCC_LIMITS,PCC_P_MIN,PCC_P_MAX,PCC_Q_MIN,PCC_Q_MAX)

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
            is_pcc_branch = bool(case.pcc_branch_mask[l])
            if case.line_limit_mode == "internal_only_no_pcc" and is_pcc_branch:
                continue
            m.addConstr(P[l]<=case.fmax_p[l]); m.addConstr(P[l]>=-case.fmax_p[l]); m.addConstr(Q[l]<=case.fmax_q[l]); m.addConstr(Q[l]>=-case.fmax_q[l])
        root_out=case.out_branches[case.root]
        m.addConstr(gp.quicksum(P[l] for l in root_out)==P0); m.addConstr(gp.quicksum(Q[l] for l in root_out)==Q0)
        if case.use_explicit_pcc_limits:
            m.addConstr(P0 >= case.pcc_p_min); m.addConstr(P0 <= case.pcc_p_max)
            m.addConstr(Q0 >= case.pcc_q_min); m.addConstr(Q0 <= case.pcc_q_max)
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
    p0 = float(sol.get('P0', np.nan)); q0 = float(sol.get('Q0', np.nan))
    for g in range(len(case.gen_buses)): names += [f'Pg_min_g{g}',f'Pg_max_g{g}',f'Qg_min_g{g}',f'Qg_max_g{g}']; bits += [int(pg[g]<=case.pg_min[g]+tol),int(pg[g]>=case.pg_max[g]-tol),int(qg[g]<=case.qg_min[g]+tol),int(qg[g]>=case.qg_max[g]-tol)]
    for i in range(case.n_bus): names += [f'Vmin_bus{i+1:02d}',f'Vmax_bus{i+1:02d}']; bits += [int(v[i]<=case.vmin**2+tol),int(v[i]>=case.vmax**2-tol)]
    for l in range(case.from_bus.size):
        is_pcc_branch = bool(case.pcc_branch_mask[l])
        if case.line_limit_mode == "internal_only_no_pcc" and is_pcc_branch:
            continue
        names += [f'Pline_pos_l{l:02d}',f'Pline_neg_l{l:02d}',f'Qline_pos_l{l:02d}',f'Qline_neg_l{l:02d}']; bits += [int(p[l]>=case.fmax_p[l]-tol),int(p[l]<=-case.fmax_p[l]+tol),int(q[l]>=case.fmax_q[l]-tol),int(q[l]<=-case.fmax_q[l]+tol)]
    if case.use_explicit_pcc_limits:
        names += ['PCC_P_min','PCC_P_max','PCC_Q_min','PCC_Q_max']
        bits += [int(p0 <= case.pcc_p_min + tol), int(p0 >= case.pcc_p_max - tol), int(q0 <= case.pcc_q_min + tol), int(q0 >= case.pcc_q_max - tol)]
    sig=tuple(bits); act=[n for n,b in zip(names,bits) if b==1]
    if case.line_limit_mode == "internal_only_no_pcc" and any('Qline_pos_l00' == n for n in act):
        print('[warning] Qline_pos_l00 still appears although PCC branch line limit should be disabled.', flush=True)
    return sig,act,names

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
    B,Kmix=w.shape
    if torch.is_tensor(taus):
        tt = taus.detach().to(device=w.device, dtype=w.dtype).view(1, -1).repeat(B, 1)
    else:
        tt = torch.as_tensor(taus, device=w.device, dtype=w.dtype).view(1, -1).repeat(B, 1)
    Ktau=tt.shape[1]
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
    if case.line_limit_mode == "internal_only_no_pcc":
        mask=torch.as_tensor(case.internal_branch_mask, device=x_real_raw.device, dtype=torch.bool)
        P_lim=P[:,mask]; Q_lim=Q[:,mask]; fp_lim=fp[:,mask]; fq_lim=fq[:,mask]
    else:
        P_lim=P; Q_lim=Q; fp_lim=fp; fq_lim=fq
    lp=(relu(P_lim.abs()-fp_lim)**2).mean(); lq=(relu(Q_lim.abs()-fq_lim)**2).mean(); pgmn=torch.tensor(case.pg_min,device=x_real_raw.device).view(1,-1); pgmx=torch.tensor(case.pg_max,device=x_real_raw.device).view(1,-1); qgmn=torch.tensor(case.qg_min,device=x_real_raw.device).view(1,-1); qgmx=torch.tensor(case.qg_max,device=x_real_raw.device).view(1,-1)
    lpg=((relu(pgmn-pg)**2)+(relu(pg-pgmx)**2)).mean(); lqg=((relu(qgmn-qg)**2)+(relu(qg-qgmx)**2)).mean(); pv=(relu(pr[:,torch.tensor(case.pv_buses,device=x_real_raw.device)]-torch.tensor(case.pv_pmax,device=x_real_raw.device).view(1,-1))**2).mean()
    pcc_lim_loss = torch.tensor(0.0, device=x_real_raw.device, dtype=x_real_raw.dtype)
    if case.use_explicit_pcc_limits:
        pcc_lim_loss = (
            torch.relu(torch.as_tensor(case.pcc_p_min, device=x_real_raw.device, dtype=x_real_raw.dtype) - p0_hat).pow(2).mean()
            + torch.relu(p0_hat - torch.as_tensor(case.pcc_p_max, device=x_real_raw.device, dtype=x_real_raw.dtype)).pow(2).mean()
            + torch.relu(torch.as_tensor(case.pcc_q_min, device=x_real_raw.device, dtype=x_real_raw.dtype) - q0_hat).pow(2).mean()
            + torch.relu(q0_hat - torch.as_tensor(case.pcc_q_max, device=x_real_raw.device, dtype=x_real_raw.dtype)).pow(2).mean()
        )
    kcl=[]
    for i in range(case.n_bus):
        if i==case.root: continue
        kcl.append((P[:,case.in_branches[i]].sum(dim=1,keepdim=True)-P[:,case.out_branches[i]].sum(dim=1,keepdim=True)+pinj[:,i:i+1]).pow(2))
        kcl.append((Q[:,case.in_branches[i]].sum(dim=1,keepdim=True)-Q[:,case.out_branches[i]].sum(dim=1,keepdim=True)+qinj[:,i:i+1]).pow(2))
    lkcl=torch.cat(kcl,dim=1).mean(); loss=pccp+pccq+gp+gq+v+lp+lq+lpg+lqg+pv+0.1*lkcl+pcc_lim_loss
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
    for j in STABLE_HARD_THETA_INIT:
        if 0<=j<T: theta_problem_weights[j]=max(theta_problem_weights[j],STABLE_HARD_THETA_WEIGHT)
    theta_problem_weights=np.clip(theta_problem_weights,0.0,THETA_PROBLEM_WEIGHT_MAX)
    print("[theta-problem-weights-v14]")
    print("theta_idx, final_weight, reason")
    for jj in range(T):
        bt="default"
        if jj in LOCATION_THETA_INIT: bt="location"
        if jj in SCALE_THETA_INIT: bt="scale"
        if jj in SHAPE_THETA_INIT: bt="shape"
        if jj in MODERATE_HARD_THETA_INIT: bt="moderate_hard"
        if jj in EXTRA_HARD_THETA_INIT: bt="extra_hard"
        print(f"{jj},{float(theta_problem_weights[jj]):.3f},{bt}")
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
    eval_cached_validation_cdf_arms(net,norm,XMU_va,THETA_FEAT,THETA_LIST,YH_va,save_prefix="cached_val_cdf_arms_v14")
    return net,norm,{"XMU_val":XMU_va,"YH_val":YH_va,"THETA_FEAT":THETA_FEAT,"theta_list":THETA_LIST,"cached_val_arms_csv":"cached_val_cdf_arms_v14.csv"}

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

def eval_cached_validation_cdf_arms(net,norm,XMU_val,THETA_FEAT,theta_list,YH_val,save_prefix="cached_val_cdf_arms_v14",n_grid=300,n_posterior_samples=0):
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
    vals=np.array([r['arms_pct'] for r in rows]); worst=np.argsort(-vals)[:3].tolist(); print(f"[cached-val-arms-v14] mean={vals.mean():.4f}"); print(f"[cached-val-arms-v14] median={np.median(vals):.4f}"); print(f"[cached-val-arms-v14] max={vals.max():.4f}"); print(f"[cached-val-arms-v14] worst3={worst}")
    colors=['#6b7280']*len(rows)
    for wi in worst: colors[wi]='#dc2626'
    plt.figure(figsize=(9,4.2),dpi=220); plt.bar([r['theta_idx'] for r in rows],[r['arms_pct'] for r in rows],color=colors); plt.axhline(vals.mean(),color='#1d4ed8',ls='--',lw=1.5,label='mean ARMS'); plt.xlabel('theta_idx'); plt.ylabel('ARMS (%)'); plt.title('Cached validation CDF ARMS of support-function distribution'); plt.legend(); plt.tight_layout(); plt.savefig(f"{save_prefix}_bar.png",dpi=260); plt.close()
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
    plt.tight_layout(); plt.savefig('FlexDomain_CDF_selected_directions_v14.png',dpi=260)

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
    plt.grid(alpha=0.25,color='0.7'); plt.xlabel('P0 (MW)',fontsize=12); plt.ylabel('Q0 (Mvar)',fontsize=12); plt.title('MC vs B-PINN quantile flexibility domains'); plt.legend(loc='center left',bbox_to_anchor=(1.02,0.5),fontsize=10); plt.axis('equal'); plt.margins(0.08); plt.tight_layout(); plt.savefig('FlexDomain_single_scenario_MC_vs_BPINN_v14_strict.png',dpi=280)
    if len(poly_mc95)>2:
        plt.figure(figsize=(9,7),dpi=260); x,y=close(poly_mc95); plt.fill(x,y,color='#bfdbfe',alpha=0.18,label='MC 95% quantile domain')
        if len(poly_mc05)>2: x,y=close(poly_mc05); plt.fill(x,y,color='#60a5fa',alpha=0.30,label='MC 5% quantile domain')
        if len(poly_mc50)>2: x,y=close(poly_mc50); plt.plot(x,y,color='#1d4ed8',lw=3,label='MC median domain')
        plt.grid(alpha=0.25); plt.title('MC quantile flexibility domains under source-load uncertainty'); plt.xlabel('P0 (MW)',fontsize=12); plt.ylabel('Q0 (Mvar)',fontsize=12); plt.axis('equal'); plt.legend(); plt.tight_layout(); plt.savefig('FlexDomain_MC_quantile_domains_v14.png',dpi=280)
    if len(poly_bn95)>2:
        plt.figure(figsize=(9,7),dpi=260); x,y=close(poly_bn95); plt.fill(x,y,color='#fed7aa',alpha=0.18,hatch='//',edgecolor='#fb923c',label='B-PINN 95% quantile domain')
        if len(poly_bn05)>2: x,y=close(poly_bn05); plt.fill(x,y,color='#fdba74',alpha=0.30,hatch='//',edgecolor='#f97316',label='B-PINN 5% quantile domain')
        if len(poly_bn50)>2: x,y=close(poly_bn50); plt.plot(x,y,'--',color='#c2410c',lw=3,label='B-PINN median domain')
        plt.grid(alpha=0.25); plt.title('B-PINN predicted quantile flexibility domains'); plt.xlabel('P0 (MW)',fontsize=12); plt.ylabel('Q0 (Mvar)',fontsize=12); plt.axis('equal'); plt.legend(); plt.tight_layout(); plt.savefig('FlexDomain_BPINN_quantile_domains_v14.png',dpi=280)

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
    plt.figure(figsize=(9,4.2),dpi=260); plt.bar([r[0] for r in rows],[r[2] for r in rows],color=colors); plt.axhline(arr.mean(),color='#1d4ed8',ls='--',lw=1.5,label='mean'); plt.axhline(arr.max(),color='#ea580c',ls=':',lw=1.8,label='max'); plt.xlabel('theta idx',fontsize=12); plt.ylabel('ARMS (%)',fontsize=12); plt.title('All-theta CDF ARMS of support-function distribution'); plt.legend(); plt.tight_layout(); plt.savefig('FlexDomain_all_theta_ARMS_bar_v14.png',dpi=280)
    return {'rows':rows,'xmu':xmu,'cache':save_cache_path}

def eval_all_theta_cdf_arms_multiscenario(case,net,norm,theta_list,n_eval_scenarios=10,mc_eval_per_scenario=400,seed=SEED_EVAL,cache_path="all_theta_eval_cache_multiscen_v9.npz",rebuild_cache=True,save_prefix="all_theta_multiscen_v14"):
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
    plt.title('B-PINN posterior uncertainty of median flexibility domain'); plt.xlabel('P0 (MW)'); plt.ylabel('Q0 (Mvar)'); plt.grid(alpha=0.25); plt.legend(); plt.axis('equal'); plt.tight_layout(); plt.savefig('FlexDomain_probability_overlay_v14_strict.png',dpi=280)

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
    plt.title('Realization cloud of flexibility domains under source-load uncertainty'); plt.xlabel('P0 (MW)'); plt.ylabel('Q0 (Mvar)'); plt.grid(alpha=0.25); plt.legend(['MC realization domains','MC median domain','B-PINN median domain']); plt.axis('equal'); plt.tight_layout(); plt.savefig('FlexDomain_realization_cloud_v14_strict.png',dpi=280)



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
        raise FileNotFoundError("No no_pcc_branch_limit dataset cache found. Set DATASET_CACHE_MODE='rebuild' or BUILD_NEW_DATASET=True.")
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
    plt.title('support atom mass by theta'); plt.xlabel('theta idx'); plt.ylabel('mass_eps_1e-2'); plt.tight_layout(); plt.savefig('support_atom_mass_by_theta_v14.png',dpi=280)
    plt.figure(figsize=(9,4),dpi=260); plt.bar(xs,[r['scene_boundary_atom_ratio_1e-2'] for r in rows],color='#60a5fa'); plt.axhline(0.2,color='k',ls='--',lw=1); plt.grid(alpha=0.25); plt.title('scene boundary atom ratio by theta'); plt.xlabel('theta idx'); plt.ylabel('ratio'); plt.tight_layout(); plt.savefig('support_scene_atom_ratio_by_theta_v14.png',dpi=280)
    pick=np.argsort(np.array([r['arms_pct'] if np.isfinite(r['arms_pct']) else r['mass_eps_1e-2']*100 for r in rows]))[::-1][:3]
    fig,axs=plt.subplots(3,1,figsize=(8,10),dpi=260)
    for ax,k in zip(axs,pick):
        r=rows[int(k)]; hs=YH[:,:,r['theta_idx']].reshape(-1)
        ax.hist(hs,bins=40,color='#93c5fd',alpha=0.7,density=True)
        ax2=ax.twinx(); sx=np.sort(hs); c=np.arange(1,len(sx)+1)/len(sx); ax2.plot(sx,c,color='#1d4ed8',lw=1.5)
        ax.axvline(r['mode_eps_1e-2'],color='#dc2626',ls='--'); ax.set_title(f"theta={r['theta_idx']} mass={r['mass_eps_1e-2']:.3f} arms={r['arms_pct']:.2f}")
        ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig('support_distribution_histograms_worst_theta_v14.png',dpi=280)
    valid=[r for r in rows if np.isfinite(r['arms_pct'])]
    if valid:
        plt.figure(figsize=(6,5),dpi=260); xv=[r['mass_eps_1e-2'] for r in valid]; yv=[r['arms_pct'] for r in valid]
        plt.scatter(xv,yv,c='#0ea5e9')
        for r in valid: plt.text(r['mass_eps_1e-2'],r['arms_pct'],str(r['theta_idx']),fontsize=8)
        plt.grid(alpha=0.25); plt.xlabel('mass_eps_1e-2'); plt.ylabel('ARMS (%)'); plt.tight_layout(); plt.savefig('support_atom_vs_arms_scatter_v14.png',dpi=280)
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
    plt.figure(figsize=(9,4),dpi=260); xs=[r['theta_idx'] for r in rows]; ys=[r['arms_pct'] for r in rows]; cs=[cmap.get(r['primary_error_type'],'#6b7280') for r in rows]; plt.bar(xs,ys,color=cs); plt.axhline(np.mean(ys),ls='--',color='k'); plt.axhline(10,ls=':',color='r'); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig('cdf_error_type_by_theta_v14.png',dpi=280)
    taus_sorted=sorted(list(set([q['tau'] for q in qrows]))); mat=np.full((len(rows),len(taus_sorted)),np.nan)
    for qi in qrows: mat[int(qi['theta_idx']),taus_sorted.index(qi['tau'])]=qi['q_err']
    plt.figure(figsize=(8,5),dpi=260); plt.imshow(mat,aspect='auto',cmap='coolwarm',vmin=-np.nanmax(np.abs(mat)),vmax=np.nanmax(np.abs(mat))); plt.colorbar(); plt.xticks(range(len(taus_sorted)),[f'{t:.2f}' for t in taus_sorted],rotation=45); plt.yticks(range(len(rows)),[r['theta_idx'] for r in rows]); plt.tight_layout(); plt.savefig('cdf_quantile_error_heatmap_v14.png',dpi=280)
    plt.figure(figsize=(9,4),dpi=260); xr=np.array([max(1e-9,r.get('q99_mc',0)-r.get('q01_mc',0)) for r in rows]); a=np.array([abs(r.get('q50_err',0))/x for r,x in zip(rows,xr)]); b=np.array([abs(r['std_ratio']-1.0) for r in rows]); c=np.array([abs(r['tail_span_err'])/(abs(r['tail_span_mc'])+1e-9) for r in rows]); xx=np.arange(len(rows)); w=0.25; plt.bar(xx-w,a,width=w,label='location'); plt.bar(xx,b,width=w,label='scale'); plt.bar(xx+w,c,width=w,label='tail'); plt.legend(); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig('cdf_location_scale_tail_bar_v14.png',dpi=280)
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
    plt.tight_layout(); plt.savefig('cdf_residual_worst_theta_v14.png',dpi=280)
    plt.figure(figsize=(6,5),dpi=260); x=[r['mass_eps_1e-2'] if np.isfinite(r['mass_eps_1e-2']) else 0 for r in rows]; y=[r['arms_pct'] for r in rows]; c=[cmap.get(r['primary_error_type'],'#6b7280') for r in rows]; plt.scatter(x,y,c=c)
    for r in rows: plt.text((r['mass_eps_1e-2'] if np.isfinite(r['mass_eps_1e-2']) else 0),r['arms_pct'],str(r['theta_idx']),fontsize=8)
    plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig('cdf_atom_vs_error_scatter_v14.png',dpi=280)
    plt.figure(figsize=(6,5),dpi=260); x=[r['posterior_cdf_std_mean'] for r in rows]; y=[abs(r['abs_cdf_bias']) for r in rows]; plt.scatter(x,y,c=c)
    mx=max(max(x),max(y))+1e-9; plt.plot([0,mx],[0,mx],'k--'); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig('cdf_bias_vs_uncertainty_scatter_v14.png',dpi=280)
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

def analyze_cdf_error_decomposition_multiscenario(net,norm,XMU_eval,THETA_FEAT,theta_list,YH_eval,save_prefix="cdf_error_decomp_multiscen_v14",cdf_grid_size=300,n_posterior_samples=EVAL_THETA_SAMPLES):
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

def diagnose_scenario_coverage_shift(XMU_train,XMU_eval,scenario_summary_csv,norm,save_path='scenario_generalization_diagnostics_v14.csv'):
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

def generate_worst_theta_diagnostics(pair_csv,theta_csv,save_path='worst_theta_diagnostics_v14.csv'):
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

def plot_worst_pair_cdfs(pair_csv,net,norm,XMU_eval,THETA_FEAT,theta_list,YH_eval,save_prefix="all_theta_multiscen_v14"):

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
    fig,ax=plt.subplots(1,1,figsize=(7,5),dpi=240); one(ax,df.iloc[0]); ax.legend(); ax.grid(alpha=0.2); plt.tight_layout(); plt.savefig(f'{save_prefix}_worst_pair_cdf.png',dpi=260)
    fig,axs=plt.subplots(2,3,figsize=(14,8),dpi=220)
    for ax,(_,r) in zip(axs.ravel(),df.head(6).iterrows()): one(ax,r); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(f'{save_prefix}_worst6_cdf_grid.png',dpi=260)

def plot_worst6_flex_domains_from_pair_csv(pair_csv,case,net,norm,theta_list):
    if (not Path(pair_csv).exists()) or (not Path(FORMAL_EVAL_CACHE_PATH).exists()):
        print("[warning] skip FlexDomain_worst6_pairs_v14_strict.png (missing pair csv or formal cache).")
        return
    df=pd.read_csv(pair_csv).sort_values('arms_pct',ascending=False).head(6)
    if len(df)==0: return
    XMU_eval=np.load(FORMAL_EVAL_CACHE_PATH,allow_pickle=True)['XMU_eval']
    fig,axs=plt.subplots(2,3,figsize=(14,8),dpi=220)
    for ax,(_,r) in zip(axs.ravel(),df.iterrows()):
        sidx=int(r['scenario_idx']); tidx=int(r['theta_idx']); arms=float(r['arms_pct']); xmu=XMU_eval[sidx]
        hs_mc=[]; hs_bp=[]
        for th in theta_list:
            hs_mc.append(support_point_mc_from_xmu(case,xmu,float(th),mc_eval=80,seed=SEED_EVAL+sidx))
            xt=torch.tensor((xmu.reshape(1,-1)-norm['x_mu_mean'])/norm['x_mu_std'],dtype=torch.float32,device=DEVICE)
            th_t=torch.tensor([[np.cos(th),np.sin(th)]],dtype=torch.float32,device=DEVICE)
            with torch.no_grad():
                _,w,mu,s=net.forward_gmm(xt,th_t,sample=False)
                qn=gmm_quantile_torch(w,mu,s,[0.5]).cpu().numpy().reshape(-1)[0]
            hs_bp.append(float(norm['h_mean'][0,0])+float(norm['h_std'][0,0])*qn)
        pm=support_values_to_polygon(theta_list,np.array(hs_mc)); pb=support_values_to_polygon(theta_list,np.array(hs_bp))
        if pm is not None and len(pm)>2: ax.plot(np.r_[pm[:,0],pm[0,0]],np.r_[pm[:,1],pm[0,1]],c='#0f766e',lw=1.8,label='OPF/MC')
        if pb is not None and len(pb)>2: ax.plot(np.r_[pb[:,0],pb[0,0]],np.r_[pb[:,1],pb[0,1]],c='#7c3aed',lw=1.6,label='B-PINN q50')
        ax.set_title(f"scenario={sidx}, worst theta={tidx}, ARMS={arms:.2f}%"); ax.grid(alpha=0.2); ax.axis('equal')
    axs.ravel()[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig('FlexDomain_worst6_pairs_v14_strict.png',dpi=260)

def write_polygon_diag_csv(path='polygon_reconstruction_diagnostics_v10.csv'):
    import pandas as pd
    cols=['scenario_idx','domain_type','quantile','method','n_points','area','valid','reason']
    pd.DataFrame(POLYGON_DIAG_ROWS if POLYGON_DIAG_ROWS else [{c:(np.nan if c in ['area'] else -1 if c=='scenario_idx' else False if c=='valid' else 'none') for c in cols}]).to_csv(path,index=False)


def check_multiscen_decomp_consistency(decomp_pair_csv="cdf_error_decomp_multiscen_v14_by_pair.csv",formal_pair_csv="all_theta_multiscen_v9_by_scenario_theta.csv",tolerance_mean=0.5,tolerance_max=2.0):
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



def compare_v14_against_v12_v13(v14_pair_csv='cdf_error_decomp_multiscen_v14_by_pair.csv'):
    if (not Path(v14_pair_csv).exists()) or (not Path('cdf_error_decomp_multiscen_v12_by_pair.csv').exists()) or (not Path('cdf_error_decomp_multiscen_v13_by_pair.csv').exists()):
        print('[v14-compare] skipped (missing csv).'); return
    v12=pd.read_csv('cdf_error_decomp_multiscen_v12_by_pair.csv')
    v13=pd.read_csv('cdf_error_decomp_multiscen_v13_by_pair.csv')
    v14=pd.read_csv(v14_pair_csv)
    m=v12[['scenario_idx','theta_idx','arms_pct']].rename(columns={'arms_pct':'arms_v12'}).merge(
        v13[['scenario_idx','theta_idx','arms_pct']].rename(columns={'arms_pct':'arms_v13'}),on=['scenario_idx','theta_idx'],how='inner'
    ).merge(
        v14[['scenario_idx','theta_idx','arms_pct']].rename(columns={'arms_pct':'arms_v14'}),on=['scenario_idx','theta_idx'],how='inner'
    )
    m['delta_v14_vs_v12']=m['arms_v14']-m['arms_v12']
    m['delta_v14_vs_v13']=m['arms_v14']-m['arms_v13']
    m['improved_vs_v12']=(m['delta_v14_vs_v12']<0).astype(int)
    m['improved_vs_v13']=(m['delta_v14_vs_v13']<0).astype(int)
    m.to_csv('v14_strict_vs_v12_v13_multiscen_comparison.csv',index=False)
    q90=lambda x: float(np.quantile(x,0.9))
    print('[v14-compare]')
    print(f"mean ARMS v12 = {m['arms_v12'].mean():.4f}")
    print(f"mean ARMS v13 = {m['arms_v13'].mean():.4f}")
    print(f"mean ARMS v14 = {m['arms_v14'].mean():.4f}")
    print(f"max ARMS v12 = {m['arms_v12'].max():.4f}")
    print(f"max ARMS v13 = {m['arms_v13'].max():.4f}")
    print(f"max ARMS v14 = {m['arms_v14'].max():.4f}")
    print(f"q90 ARMS v12 = {q90(m['arms_v12']):.4f}")
    print(f"q90 ARMS v13 = {q90(m['arms_v13']):.4f}")
    print(f"q90 ARMS v14 = {q90(m['arms_v14']):.4f}")
    print(f"num improved vs v12 = {int((m['delta_v14_vs_v12']<0).sum())}")
    print(f"num worsened vs v12 = {int((m['delta_v14_vs_v12']>0).sum())}")
    print(f"num improved vs v13 = {int((m['delta_v14_vs_v13']<0).sum())}")
    print(f"num worsened vs v13 = {int((m['delta_v14_vs_v13']>0).sum())}")
    t2=m[m['theta_idx']==2]
    t7=m[m['theta_idx']==7]
    if len(t2)>0:
        print(f"theta=2 mean/max v12/v13/v14 = {t2['arms_v12'].mean():.4f}/{t2['arms_v13'].mean():.4f}/{t2['arms_v14'].mean():.4f} ; {t2['arms_v12'].max():.4f}/{t2['arms_v13'].max():.4f}/{t2['arms_v14'].max():.4f}")
    if len(t7)>0:
        print(f"theta=7 mean/max v12/v13/v14 = {t7['arms_v12'].mean():.4f}/{t7['arms_v13'].mean():.4f}/{t7['arms_v14'].mean():.4f} ; {t7['arms_v12'].max():.4f}/{t7['arms_v13'].max():.4f}/{t7['arms_v14'].max():.4f}")
    w6v12=np.sort(m['arms_v12'].values)[-6:]; w6v13=np.sort(m['arms_v13'].values)[-6:]; w6v14=np.sort(m['arms_v14'].values)[-6:]
    print(f"worst6 mean v12/v13/v14 = {np.mean(w6v12):.4f}/{np.mean(w6v13):.4f}/{np.mean(w6v14):.4f}")

def finalize_v14_checkpoint(net,norm,cached_stats,external_stats):
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
    with open(f"{TRAINING_RESULT_DIR}/v14_rollback_v12_strategy_keep_visuals_metrics.json",'w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
    best_metrics_path=Path(TRAINING_RESULT_DIR)/'v14_rollback_v12_strategy_keep_visuals_best_metrics.json'
    best_score=float('inf')
    if best_metrics_path.exists():
        try: best_score=float(json.loads(best_metrics_path.read_text(encoding='utf-8')).get('selection_score',float('inf')))
        except Exception: best_score=float('inf')
    update_best=(selection_score<best_score)
    if update_best:
        shutil.copy2(model_path,f"{TRAINING_RESULT_DIR}/v14_rollback_v12_strategy_keep_visuals_best_model.pt")
        shutil.copy2(norm_path,f"{TRAINING_RESULT_DIR}/v14_rollback_v12_strategy_keep_visuals_best_norm.pkl")
        shutil.copy2(config_path,f"{TRAINING_RESULT_DIR}/v14_rollback_v12_strategy_keep_visuals_best_config.json")
        with open(best_metrics_path,'w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
    print('[v14-checkpoint]')
    print(f"cached_val_mean={cached_stats['cached_val_mean_arms']:.4f}")
    print(f"external_mean={external_stats['external_mean_arms']:.4f}")
    print(f"external_q90={external_stats['external_q90_arms']:.4f}")
    print(f"external_max={external_stats['external_max_arms']:.4f}")
    print(f"selection_score={selection_score:.4f}")
    print(f"best_score={best_score:.4f}")
    print(f"update_best={update_best}")


def write_visualization_manifest_v14_strict():
    rows=[
['cached_val_cdf_arms_v14.csv','cached validation','eval_cached_validation_cdf_arms','每个theta验证ARMS表','after training','v14'],
['cached_val_cdf_arms_v14_bar.png','cached validation','eval_cached_validation_cdf_arms','每个theta验证ARMS柱状图','after training','v14'],
['FlexDomain_CDF_selected_directions_v14.png','flexdomain','eval_and_plot_flex_domain','方向CDF对比','if enabled','v14'],
['FlexDomain_overlay_quantile_comparison_v14.png','flexdomain','eval_and_plot_flex_domain','分位域叠加对比','if enabled','v14'],
['FlexDomain_MC_quantile_domains_v14.png','flexdomain','eval_and_plot_flex_domain','MC分位域','if enabled','v14'],
['FlexDomain_BPINN_quantile_domains_v14.png','flexdomain','eval_and_plot_flex_domain','BPINN分位域','if enabled','v14'],
['FlexDomain_all_theta_ARMS_bar_v14.png','flexdomain','eval_all_theta_cdf_arms_multiscenario','全theta ARMS柱状图','after formal eval','v14'],
['FlexDomain_probability_overlay_v14_strict.png','flexibility domain','eval_and_plot_flex_domain_posterior','概率灵活域叠加','if RUN_FLEX_DOMAIN_PLOTS','v14_strict'],
['FlexDomain_realization_cloud_v14_strict.png','flexibility domain','eval_and_plot_realization_cloud','灵活域实现云图','if RUN_FLEX_DOMAIN_PLOTS','v14_strict'],
['FlexDomain_single_scenario_MC_vs_BPINN_v14_strict.png','flexibility domain','eval_and_plot_flex_domain','单场景真实/预测灵活域对比','if RUN_FLEX_DOMAIN_PLOTS','v14_strict'],
['FlexDomain_worst6_pairs_v14_strict.png','flexibility domain','plot_worst6_flex_domains_from_pair_csv','worst6场景完整灵活域对比','if pair csv exists','v14_strict'],
['support_atom_mass_by_theta_v14.png','atom diagnostic','diagnose_support_distribution_atoms','atom质量','always','v14'],
['support_scene_atom_ratio_by_theta_v14.png','atom diagnostic','diagnose_support_distribution_atoms','场景atom比例','always','v14'],
['support_distribution_histograms_worst_theta_v14.png','atom diagnostic','diagnose_support_distribution_atoms','最差theta分布直方图','always','v14'],
['support_atom_vs_arms_scatter_v14.png','atom diagnostic','diagnose_support_distribution_atoms','atom与ARMS散点','always','v14'],
['cdf_error_type_by_theta_v14.png','cdf decomp','analyze_cdf_error_decomposition','error type柱状图','after decomp','v14'],
['cdf_quantile_error_heatmap_v14.png','cdf decomp','analyze_cdf_error_decomposition','分位误差热力图','after decomp','v14'],
['cdf_location_scale_tail_bar_v14.png','cdf decomp','analyze_cdf_error_decomposition','loc/scale/tail条形图','after decomp','v14'],
['cdf_residual_worst_theta_v14.png','cdf decomp','analyze_cdf_error_decomposition','最差theta残差','after decomp','v14'],
['cdf_atom_vs_error_scatter_v14.png','cdf decomp','analyze_cdf_error_decomposition','atom vs error','after decomp','v14'],
['cdf_bias_vs_uncertainty_scatter_v14.png','cdf decomp','analyze_cdf_error_decomposition','bias vs uncertainty','after decomp','v14'],
['all_theta_multiscen_v14_worst_pair_cdf.png','worst pair CDF','plot_worst_pair_cdfs','最差pair CDF','after formal cache eval','v14'],
['all_theta_multiscen_v14_worst6_cdf_grid.png','worst pair CDF','plot_worst_pair_cdfs','最差6个CDF','after formal cache eval','v14'],
['cdf_error_decomp_multiscen_v14_by_pair.csv','cdf decomp','analyze_cdf_error_decomposition_multiscenario','pair级误差表','after formal cache eval','v14'],
['cdf_error_decomp_multiscen_v14_by_theta.csv','cdf decomp','analyze_cdf_error_decomposition_multiscenario','theta汇总','after formal cache eval','v14'],
['cdf_error_decomp_multiscen_v14_by_scenario.csv','cdf decomp','analyze_cdf_error_decomposition_multiscenario','scenario汇总','after formal cache eval','v14'],
['cdf_error_decomp_multiscen_v14_summary.csv','cdf decomp','analyze_cdf_error_decomposition_multiscenario','总体汇总','after formal cache eval','v14'],
['worst_theta_diagnostics_v14.csv','diagnostic','generate_worst_theta_diagnostics','worst theta诊断','after formal cache eval','v14'],
['scenario_generalization_diagnostics_v14.csv','diagnostic','diagnose_scenario_coverage_shift','场景覆盖诊断','after formal cache eval','v14'],
['v14_strict_vs_v12_v13_multiscen_comparison.csv','comparison','compare_v14_against_v12_v13','v14_strict对比v12/v13外部多场景对比','after formal cache eval','v14_strict']
]
    import csv
    with open('visualization_manifest_v14_strict.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['filename','category','function_name','purpose','generated_when','version_status']); w.writerows(rows)
    print('[visualization-manifest-v14-strict] saved visualization_manifest_v14_strict.csv')

def main():
    case=build_ieee33_case()
    if RUN_SANITY_CHECKS: flex_opf_sanity_check(case)
    print('生成 33 节点灵活域训练数据...')
    print('[v14-run-guide]')
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
    diagnose_support_distribution_atoms(YH,THETA_FEAT,THETA_LIST,active_records=active,arms_csv_path=None,save_prefix='support_atom_train_v14')
    if RUN_TRAINING:
        net,norm,train_info=train_bayes_flex_gmm2(case,XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG)
    else:
        if not LOAD_TRAINED_MODEL: raise RuntimeError("RUN_TRAINING=False but LOAD_TRAINED_MODEL=False; no model source.")
        net,norm=load_trained_flex_model(case,LOAD_MODEL_PATH,LOAD_NORM_PATH); train_info={}
        if RUN_EVAL_ONLY:
            cv=eval_cached_validation_cdf_arms(net,norm,XMU,THETA_FEAT,THETA_LIST,YH,save_prefix="cached_val_cdf_arms_v14")
            vals=np.array([r["arms_pct"] for r in cv],dtype=float)
            print(f"[eval-only-cached-val] mean={np.nanmean(vals):.4f} median={np.nanmedian(vals):.4f} max={np.nanmax(vals):.4f} worst3={np.argsort(-vals)[:3].tolist()}")
            Path(TRAINING_RESULT_DIR).mkdir(parents=True,exist_ok=True)
            metrics={'cached_val_mean_arms':float(np.nanmean(vals)),'cached_val_max_arms':float(np.nanmax(vals)),'worst3':np.argsort(-vals)[:3].tolist(),'timestamp':datetime.now().isoformat(),'model_path':LOAD_MODEL_PATH if not RUN_TRAINING else 'trained_in_run'}
            with open('training_results/v14_rollback_v12_strategy_keep_visuals_metrics.json','w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
            if np.nanmean(vals)>15 or np.nanmax(vals)>25:
                print("[eval-only-warning] loaded model cached validation ARMS is high; formal multiscen evaluation may be poor.")
    if RUN_SANITY_CHECKS: flex_realization_sanity_check(case,net,norm,THETA_LIST)
    if RUN_FULL_OPF_EVAL or RUN_FLEX_DOMAIN_PLOTS:
        eval_and_plot_flex_domain(case,net,norm,THETA_LIST)
        eval_and_plot_flex_domain_posterior(case,net,norm,THETA_LIST)
        eval_and_plot_realization_cloud(case,net,norm,THETA_LIST)
    if RUN_FULL_OPF_EVAL:
        eval_and_plot_direction_cdfs(case,net,norm,THETA_LIST)
        if MULTI_SCENARIO_FORMAL_EVAL:
            eval_pack=eval_all_theta_cdf_arms_multiscenario(case,net,norm,THETA_LIST,n_eval_scenarios=N_FORMAL_EVAL_SCENARIOS,mc_eval_per_scenario=MC_EVAL_PER_SCENARIO,cache_path=FORMAL_EVAL_CACHE_PATH,rebuild_cache=FORMAL_EVAL_REBUILD_CACHE,save_prefix="all_theta_multiscen_v14")
            formal_active_rebuilt = FORMAL_EVAL_REBUILD_CACHE
        else:
            eval_pack=eval_all_theta_cdf_arms(case,net,norm,THETA_LIST)
    else:
        print("[eval] RUN_FULL_OPF_EVAL=False, skip OPF-based evaluation plots.")
    print('\n=== Support distribution atom/boundary diagnostic with ARMS ===')
    arms_path='all_theta_cdf_arms_v1.csv' if Path('all_theta_cdf_arms_v1.csv').exists() else None
    diagnose_support_distribution_atoms(YH,THETA_FEAT,THETA_LIST,active_records=None,arms_csv_path=arms_path,save_prefix='support_atom_eval_v14')
    cached_cv=eval_cached_validation_cdf_arms(net,norm,XMU,THETA_FEAT,THETA_LIST,YH,save_prefix='cached_val_cdf_arms_v14')
    cv_vals=np.array([r['arms_pct'] for r in cached_cv],dtype=float)
    cached_stats={'cached_val_mean_arms':float(np.nanmean(cv_vals)),'cached_val_max_arms':float(np.nanmax(cv_vals)),'cached_val_worst3':np.argsort(-cv_vals)[:3].tolist()}
    print(f"[cached-val-v14] mean={cached_stats['cached_val_mean_arms']:.4f} median={float(np.nanmedian(cv_vals)):.4f} max={cached_stats['cached_val_max_arms']:.4f} worst3={cached_stats['cached_val_worst3']}")
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
        pair_df,_,_,_=analyze_cdf_error_decomposition_multiscenario(net,norm,XMU_eval,THETA_FEAT,THETA_LIST,YH_eval,save_prefix='cdf_error_decomp_multiscen_v14',cdf_grid_size=300,n_posterior_samples=EVAL_THETA_SAMPLES)
        scenario_src='cdf_error_decomp_multiscen_v14_by_scenario.csv' if Path('cdf_error_decomp_multiscen_v14_by_scenario.csv').exists() else ('all_theta_multiscen_v9_by_scenario_summary.csv' if Path('all_theta_multiscen_v9_by_scenario_summary.csv').exists() else 'cdf_error_decomp_multiscen_v14_by_scenario.csv')
        diagnose_scenario_coverage_shift(XMU,XMU_eval,scenario_src,norm,save_path='scenario_generalization_diagnostics_v14.csv')
        theta_src='cdf_error_decomp_multiscen_v14_by_theta.csv' if Path('cdf_error_decomp_multiscen_v14_by_theta.csv').exists() else ('all_theta_multiscen_v9_by_theta_summary.csv' if Path('all_theta_multiscen_v9_by_theta_summary.csv').exists() else 'cdf_error_decomp_multiscen_v14_by_theta.csv')
        generate_worst_theta_diagnostics('cdf_error_decomp_multiscen_v14_by_pair.csv',theta_src,save_path='worst_theta_diagnostics_v14.csv')
        plot_worst_pair_cdfs('cdf_error_decomp_multiscen_v14_by_pair.csv',net,norm,XMU_eval,THETA_FEAT,THETA_LIST,YH_eval)
        if RUN_FLEX_DOMAIN_PLOTS:
            plot_worst6_flex_domains_from_pair_csv('cdf_error_decomp_multiscen_v14_by_pair.csv',case,net,norm,THETA_LIST)
        compare_v14_against_v12_v13('cdf_error_decomp_multiscen_v14_by_pair.csv')
        if Path('formal_active_pattern_records_v10.csv').exists() or FORMAL_EVAL_REBUILD_CACHE:
            print('[active-pattern-novelty] placeholder: records ready for analysis.')
        else:
            print('[active-pattern-novelty] formal active records unavailable; set FORMAL_EVAL_REBUILD_CACHE=True to generate.')
        ext_df=pd.read_csv('cdf_error_decomp_multiscen_v14_by_pair.csv')
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
            finalize_v14_checkpoint(net,norm,cached_stats,external_stats)
    else:
        analyze_cdf_error_decomposition(net=net,norm=norm,XMU_eval=XMU_eval,THETA_FEAT=THETA_FEAT,theta_list=THETA_LIST,YH_eval=YH_eval,atom_diag_csv_path='support_atom_eval_by_theta.csv',active_diag_csv_path='support_atom_eval_active_patterns.csv',save_prefix='cdf_error_decomp_v14')
    write_polygon_diag_csv()
    write_visualization_manifest_v14_strict()
    if RUN_MULTI_TEST: eval_multiple_flex_scenarios(case,net,norm,THETA_LIST)


# ===== v15 theta-independent experiment =====
PRINT_TRAIN_PROGRESS = True
LOG_FIRST_N_EPOCHS = 5
LOG_EVERY_EPOCH = 10
LOG_LAST_N_EPOCHS = 3
LOG_EVERY_N_BATCHES = 20
PRINT_BATCH_PROGRESS = False
PRINT_PROGRESS_FLUSH = True
RUN_FULL_OPF_EVAL = False
RUN_FLEX_DOMAIN_PLOTS = False
RUN_P0_STYLE_EXTERNAL_EVAL = True
P0_STYLE_EVAL_THETA_LIST = [2]
P0_STYLE_SINGLE_SCENARIO_MC = 2500
P0_STYLE_MULTI_SCENARIO_EVAL = False
P0_STYLE_N_TEST_SCENARIOS = 20
P0_STYLE_MULTI_SCENARIO_MC = 800
P0_STYLE_EVAL_SEED = SEED_EVAL
P0_STYLE_EXTERNAL_EVAL_DIR = TRAINING_RESULT_DIR

def should_print_epoch(ep, epochs):
    ep1 = ep + 1
    return (
        ep1 <= LOG_FIRST_N_EPOCHS
        or ep1 % LOG_EVERY_EPOCH == 0
        or ep1 > epochs - LOG_LAST_N_EPOCHS
    )

def flatten_single_theta_dataset(XMU, XREAL, YH, YP0, YQ0, YPG, YQG, theta_idx, theta_list):
    theta = float(theta_list[theta_idx])
    alpha, beta = float(np.cos(theta)), float(np.sin(theta))
    yh = YH[:, :, theta_idx]
    yp0 = YP0[:, :, theta_idx]
    yq0 = YQ0[:, :, theta_idx]
    ypg = YPG[:, :, theta_idx, :]
    yqg = YQG[:, :, theta_idx, :]
    yt = -beta * yp0 + alpha * yq0
    n_scen, mc = yh.shape
    return dict(xmu_flat=np.repeat(XMU, mc, axis=0), xreal_flat=XREAL.reshape(n_scen*mc, -1), yh_flat=yh.reshape(-1,1), yp0_flat=yp0.reshape(-1,1), yq0_flat=yq0.reshape(-1,1), yt_flat=yt.reshape(-1,1), ypg_flat=ypg.reshape(n_scen*mc, -1), yqg_flat=yqg.reshape(n_scen*mc, -1), XMU_scen=XMU, YH_theta_scen=yh, alpha=alpha, beta=beta, theta_idx=theta_idx, theta=theta)

class BayesSingleThetaGMM2SupportNet(nn.Module):
    def __init__(self, in_dim, case, hidden=160, depth=3, prior_sigma=1.0, init_rho=-5.0):
        super().__init__()
        self.n_gen = len(case.gen_buses)
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(depth):
            self.layers.append(BayesLinear(d, hidden, prior_sigma=prior_sigma, init_rho=init_rho))
            d = hidden
        self.gmm_out = BayesLinear(hidden, 3 * N_GMM_COMPONENTS, prior_sigma=prior_sigma, init_rho=init_rho)
        self.rec_out = BayesLinear(hidden + in_dim + 1, 1 + 2*self.n_gen, prior_sigma=prior_sigma, init_rho=init_rho)
        self.act = nn.ReLU()
        self.register_buffer('pg_min_t', torch.tensor(case.pg_min, dtype=torch.float32).view(1,-1))
        self.register_buffer('pg_max_t', torch.tensor(case.pg_max, dtype=torch.float32).view(1,-1))
        self.register_buffer('qg_min_t', torch.tensor(case.qg_min, dtype=torch.float32).view(1,-1))
        self.register_buffer('qg_max_t', torch.tensor(case.qg_max, dtype=torch.float32).view(1,-1))

    def encode_gmm_single(self, x_mu_norm, sample=True):
        h = x_mu_norm
        for layer in self.layers:
            h = self.act(layer(h, sample=sample))
        return h

    def forward_gmm_single(self, x_mu_norm, sample=True):
        h = self.encode_gmm_single(x_mu_norm, sample=sample)
        out = self.gmm_out(h, sample=sample)
        K = N_GMM_COMPONENTS
        w = torch.softmax(out[:, :K], dim=1)
        mu = out[:, K:2*K]
        sigma = torch.nn.functional.softplus(out[:, 2*K:3*K]) + GMM_SIGMA_FLOOR
        return h, w, mu, sigma

    def recover_boundary_dispatch_from_h_fixed_theta(self, hidden_feature, x_real_norm, h_label, h_mean, h_std, t_mean, t_std, alpha, beta, sample=True):
        h_norm = (h_label - h_mean) / (h_std + 1e-9)
        out = self.rec_out(torch.cat([hidden_feature, x_real_norm, h_norm], dim=1), sample=sample)
        t_norm = out[:, 0:1]
        raw_pg = out[:, 1:1+self.n_gen]
        raw_qg = out[:, 1+self.n_gen:1+2*self.n_gen]
        t_hat = t_mean + t_std * t_norm
        pg_hat = self.pg_min_t + torch.sigmoid(raw_pg) * (self.pg_max_t - self.pg_min_t)
        qg_hat = self.qg_min_t + torch.sigmoid(raw_qg) * (self.qg_max_t - self.qg_min_t)
        p0_hat = alpha * h_label - beta * t_hat
        q0_hat = beta * h_label + alpha * t_hat
        return p0_hat, q0_hat, pg_hat, qg_hat, t_hat

    def kl_divergence(self):
        return sum(layer.kl_divergence() for layer in self.layers) + self.gmm_out.kl_divergence() + self.rec_out.kl_divergence()

def evaluate_single_theta_model(net, norm, XMU_scen, YH_theta_scen, theta_idx, theta_list, save_cdf_path=None):
    theta = float(theta_list[theta_idx]); alpha, beta = float(np.cos(theta)), float(np.sin(theta))
    hm, hs = float(norm['h_mean'][0,0]), float(norm['h_std'][0,0])
    errs = {q:[] for q in [0.05,0.25,0.50,0.75,0.95]}; arms=[]; me=[]; se=[]; payload=None
    for i in range(XMU_scen.shape[0]):
        ys = np.asarray(YH_theta_scen[i], dtype=float); ys = ys[np.isfinite(ys)]
        if ys.size < 4: continue
        grid = np.linspace(ys.min()-0.2, ys.max()+0.2, 250)
        emp_cdf = np.searchsorted(np.sort(ys), grid, side='right') / ys.size
        xt = torch.tensor((XMU_scen[i:i+1]-norm['x_mu_mean'])/norm['x_mu_std'], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            _, w, mu, sigma = net.forward_gmm_single(xt, sample=False)
        w = w.cpu().numpy().reshape(-1); mu = mu.cpu().numpy().reshape(-1); sigma = sigma.cpu().numpy().reshape(-1)
        pred_cdf = gmm_cdf((grid-hm)/(hs+1e-9), w, mu, sigma)
        arms.append(100*np.sqrt(np.mean((pred_cdf-emp_cdf)**2)))
        for q in errs:
            q_pred_norm = float(np.asarray(gmm_quantile(q, w, mu, sigma)).reshape(-1)[0])
            q_pred_phys = hm + hs * q_pred_norm
            q_emp_phys = float(np.quantile(ys, q))
            errs[q].append(q_pred_phys - q_emp_phys)
        pred_mean = hm + hs * float(np.sum(w*mu))
        pred_std = hs * float(np.sqrt(max(0.0, np.sum(w*(sigma**2 + mu**2)) - (np.sum(w*mu)**2))))
        me.append(pred_mean - float(np.mean(ys))); se.append(pred_std - float(np.std(ys)))
        if payload is None: payload=(grid, emp_cdf, pred_cdf)
    if save_cdf_path and payload is not None:
        Path(save_cdf_path).parent.mkdir(parents=True, exist_ok=True)
        grid, emp_cdf, pred_cdf = payload
        plt.figure(figsize=(7,5), dpi=180)
        plt.plot(grid, emp_cdf, 'k-', lw=2, label='Empirical CDF')
        plt.plot(grid, pred_cdf, 'r--', lw=2, label='Predicted GMM CDF')
        plt.title(f'CDF theta_idx={theta_idx:02d}, theta={theta:.4f}')
        plt.grid(alpha=0.2); plt.legend(); plt.tight_layout(); plt.savefig(save_cdf_path, dpi=220); plt.close()
    return dict(theta_idx=theta_idx, theta=theta, alpha=alpha, beta=beta, cdf_arms=float(np.mean(arms)) if arms else np.nan, q05_error=float(np.mean(errs[0.05])) if errs[0.05] else np.nan, q25_error=float(np.mean(errs[0.25])) if errs[0.25] else np.nan, q50_error=float(np.mean(errs[0.50])) if errs[0.50] else np.nan, q75_error=float(np.mean(errs[0.75])) if errs[0.75] else np.nan, q95_error=float(np.mean(errs[0.95])) if errs[0.95] else np.nan, q50_rmse=float(np.sqrt(np.mean(np.square(errs[0.50])))) if errs[0.50] else np.nan, mean_error=float(np.mean(me)) if me else np.nan, std_error=float(np.mean(se)) if se else np.nan, n_eval_scenarios=int(len(arms)))

def train_single_theta_model(case, XMU, XREAL, YH, YP0, YQ0, YPG, YQG, theta_idx, theta_list):
    Path(TRAINING_RESULT_DIR).mkdir(parents=True, exist_ok=True)
    d = flatten_single_theta_dataset(XMU, XREAL, YH, YP0, YQ0, YPG, YQG, theta_idx, theta_list)
    rng=np.random.default_rng(SEED_TRAIN); n_scen=XMU.shape[0]; mc=YH.shape[1]
    perm=rng.permutation(n_scen); n_val=max(1,int(round(0.1*n_scen))); va=perm[-n_val:]; tr=perm[:-n_val]
    scen_flat=np.repeat(np.arange(n_scen), mc); tr_mask=np.isin(scen_flat,tr); va_mask=np.isin(scen_flat,va)
    xmu_tr, xr_tr, yh_tr, yt_tr = d['xmu_flat'][tr_mask], d['xreal_flat'][tr_mask], d['yh_flat'][tr_mask], d['yt_flat'][tr_mask]
    yp0_tr, yq0_tr, ypg_tr, yqg_tr = d['yp0_flat'][tr_mask], d['yq0_flat'][tr_mask], d['ypg_flat'][tr_mask], d['yqg_flat'][tr_mask]
    norm={'x_mu_mean':xmu_tr.mean(0,keepdims=True),'x_mu_std':xmu_tr.std(0,keepdims=True)+1e-9,'x_real_mean':xr_tr.mean(0,keepdims=True),'x_real_std':xr_tr.std(0,keepdims=True)+1e-9,'h_mean':np.array([[yh_tr.mean()]]),'h_std':np.array([[yh_tr.std()+1e-9]]),'t_mean':np.array([[yt_tr.mean()]]),'t_std':np.array([[yt_tr.std()+1e-9]]),'theta_idx':theta_idx,'theta':d['theta'],'alpha':d['alpha'],'beta':d['beta']}
    to=lambda a: torch.tensor(a,dtype=torch.float32,device=DEVICE)
    xmu_n=to((xmu_tr-norm['x_mu_mean'])/norm['x_mu_std']); xr_n=to((xr_tr-norm['x_real_mean'])/norm['x_real_std']); xr_raw=to(xr_tr)
    yh_t, yt_t, ypg_t, yqg_t=to(yh_tr),to(yt_tr),to(ypg_tr),to(yqg_tr)
    hm,hs,tm,ts=to(norm['h_mean']),to(norm['h_std']),to(norm['t_mean']),to(norm['t_std'])
    net=BayesSingleThetaGMM2SupportNet(XMU.shape[1],case,hidden=160,depth=3,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO).to(DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=LR)
    print('[v15-theta-train-start]', flush=True); print(f'theta_idx = {theta_idx}', flush=True); print(f'theta = {d["theta"]}', flush=True); print(f'alpha = {d["alpha"]}', flush=True); print(f'beta = {d["beta"]}', flush=True); print(f'epochs = {EPOCHS}', flush=True); print(f'train samples = {xmu_n.shape[0]}', flush=True); print(f'val samples = {int(va_mask.sum())}', flush=True); print(f'batch size = {BATCH_SIZE}', flush=True); print(f'device = {DEVICE}', flush=True); print(f'DATASET_CACHE_MODE = {DATASET_CACHE_MODE}', flush=True); print('training target = h_theta', flush=True); print('v8-style realization-conditioned physics = True', flush=True)
    rows=[]; n=xmu_n.shape[0]; nb=(n+BATCH_SIZE-1)//BATCH_SIZE; t0=datetime.now(); best_cdf=np.inf; best_ep=0; last_stat=None
    for ep in range(EPOCHS):
        ep0=datetime.now(); p=np.random.default_rng(SEED_TRAIN+ep).permutation(n); beta_kl=min(1.0,(ep+1)/max(1,KL_WARMUP_EPOCHS))*BETA_KL_MAX
        stat={k:0.0 for k in ['total','nll','boundary','dispatch','t','phys','support','kl']}
        for b in range(nb):
            ii=p[b*BATCH_SIZE:min((b+1)*BATCH_SIZE,n)]
            feat,w,mu,sigma=net.forward_gmm_single(xmu_n[ii],sample=True)
            h_norm=(yh_t[ii]-hm)/(hs+1e-9)
            nll_loss=(-gmm_log_prob(h_norm,w,mu,sigma)).mean()
            p0_hat,q0_hat,pg_hat,qg_hat,t_hat=net.recover_boundary_dispatch_from_h_fixed_theta(feat,xr_n[ii],yh_t[ii],hm,hs,tm,ts,d['alpha'],d['beta'],sample=True)
            h_pred_mean_phys=hm+hs*torch.sum(w*mu,dim=1,keepdim=True)
            boundary_sup_loss=((h_pred_mean_phys-yh_t[ii])**2).mean()
            dispatch_sup_loss=((pg_hat-ypg_t[ii])**2).mean()+((qg_hat-yqg_t[ii])**2).mean()
            t_sup_loss=((t_hat-yt_t[ii])**2).mean()
            phys_loss=physics_loss_flex(case,xr_raw[ii],p0_hat,q0_hat,pg_hat,qg_hat)
            support_consist_loss=((d['alpha']*p0_hat + d['beta']*q0_hat - yh_t[ii])**2).mean()
            kl_loss=net.kl_divergence()/max(1,n)
            loss=nll_loss+LAM_BOUNDARY_SUP*boundary_sup_loss+LAM_DISPATCH_SUP*dispatch_sup_loss+LAM_T_SUP*t_sup_loss+LAM_PHYS_FLEX*phys_loss+LAM_SUPPORT_CONSIST*support_consist_loss+beta_kl*kl_loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
            for k,v in zip(stat.keys(),[loss,nll_loss,boundary_sup_loss,dispatch_sup_loss,t_sup_loss,phys_loss,support_consist_loss,kl_loss]): stat[k]+=float(v.detach().cpu())
            if PRINT_BATCH_PROGRESS and (b+1)%LOG_EVERY_N_BATCHES==0:
                print(f"[v15-theta-batch] theta_idx={theta_idx:02d} epoch {ep+1:03d}/{EPOCHS:03d} batch {b+1:03d}/{nb:03d} total={float(loss):.6f} nll={float(nll_loss):.6f} phys={float(phys_loss):.6f}", flush=True)
        for k in stat: stat[k]/=max(1,nb)
        val=evaluate_single_theta_model(net,norm,d['XMU_scen'][va],d['YH_theta_scen'][va],theta_idx,theta_list,save_cdf_path=(f"{TRAINING_RESULT_DIR}/CDF_theta_independent_theta{theta_idx:02d}_v15.png" if ep==EPOCHS-1 else None))
        elapsed=(datetime.now()-t0).total_seconds(); ep_sec=(datetime.now()-ep0).total_seconds(); eta=(EPOCHS-ep-1)*ep_sec
        if np.isfinite(val['cdf_arms']) and val['cdf_arms'] < best_cdf:
            best_cdf = float(val['cdf_arms']); best_ep = ep+1
        warn_flag = (not np.isfinite(stat['total'])) or (not np.isfinite(val['cdf_arms']))
        if should_print_epoch(ep, EPOCHS) or warn_flag:
            pct = 100.0 * (ep+1) / max(1, EPOCHS)
            bar_n = 20; fill = int(round(bar_n * (ep+1) / max(1, EPOCHS)))
            bar = '[' + '#' * fill + '-' * (bar_n - fill) + ']'
            if warn_flag:
                print(f"[v15-theta-warning] theta_idx={theta_idx:02d} epoch={ep+1} non-finite metric detected", flush=True)
            print(f"[v15-theta-train] theta_idx={theta_idx:02d} epoch {ep+1:03d}/{EPOCHS:03d} {pct:5.1f}% {bar} stage=A lr={opt.param_groups[0]['lr']:.6g} beta_kl={beta_kl:.6f}", flush=True)
            print(f"  loss: total={stat['total']:.6f} nll={stat['nll']:.6f} boundary={stat['boundary']:.6f} dispatch={stat['dispatch']:.6f} t={stat['t']:.6f} phys={stat['phys']:.6f} support={stat['support']:.6f} kl_raw={stat['kl']:.6f} kl_weighted={(beta_kl*stat['kl']):.6f}", flush=True)
            print(f"  val: cdf_arms={val['cdf_arms']:.6f} q50_rmse={val['q50_rmse']:.6f} mean_error={val['mean_error']:.6f} std_error={val['std_error']:.6f} best_cdf_arms={best_cdf:.6f} best_epoch={best_ep}", flush=True)
        rows.append({'epoch':ep+1,'stage':'A','lr':opt.param_groups[0]['lr'],'epoch_time_sec':ep_sec,'elapsed_sec':elapsed,'eta_sec':eta,'total_loss':stat['total'],'nll_loss':stat['nll'],'boundary_sup_loss':stat['boundary'],'dispatch_sup_loss':stat['dispatch'],'t_sup_loss':stat['t'],'phys_loss':stat['phys'],'support_consist_loss':stat['support'],'kl_loss':stat['kl'],'beta_kl':beta_kl,'kl_raw_loss':stat['kl'],'kl_weighted_loss':beta_kl*stat['kl'],'best_val_cdf_arms_so_far':best_cdf,'best_epoch_so_far':best_ep,'val_cdf_arms':val['cdf_arms'],'val_q50_rmse':val['q50_rmse'],'val_mean_error':val['mean_error'],'val_std_error':val['std_error']})
        last_stat = (stat.copy(), beta_kl)
    Path(TRAINING_RESULT_DIR).mkdir(parents=True,exist_ok=True)
    tag=f"theta_independent_theta{theta_idx:02d}"
    model_path=f"{TRAINING_RESULT_DIR}/{tag}_model.pt"; norm_path=f"{TRAINING_RESULT_DIR}/{tag}_norm.pkl"; cfg_path=f"{TRAINING_RESULT_DIR}/{tag}_config.json"; log_path=f"{TRAINING_RESULT_DIR}/{tag}_training_log.csv"; val_path=f"{TRAINING_RESULT_DIR}/{tag}_val_metrics.csv"
    torch.save(net.state_dict(),model_path)
    with open(norm_path,'wb') as f: pickle.dump(norm,f)
    with open(cfg_path,'w',encoding='utf-8') as f: json.dump({'theta_idx':theta_idx,'theta':d['theta'],'alpha':d['alpha'],'beta':d['beta'],'RUN_MODE':RUN_MODE,'NUM_SCENARIOS':NUM_SCENARIOS,'MC_PER_SCENARIO':MC_PER_SCENARIO,'N_THETA':N_THETA,'EPOCHS':EPOCHS,'DATASET_CACHE_TAG':DATASET_CACHE_TAG,'TRAINING_RUN_TAG':TRAINING_RUN_TAG,'model_type':'v8_style_theta_independent_support_bpinN'},f,ensure_ascii=False,indent=2)
    pd.DataFrame(rows).to_csv(log_path,index=False)
    final_val=evaluate_single_theta_model(net,norm,d['XMU_scen'][va],d['YH_theta_scen'][va],theta_idx,theta_list,save_cdf_path=f"{TRAINING_RESULT_DIR}/CDF_theta_independent_theta{theta_idx:02d}_v15.png")
    pd.DataFrame([final_val]).to_csv(val_path,index=False)
    final_stat, final_beta = last_stat if last_stat is not None else ({'total':np.nan,'nll':np.nan,'phys':np.nan,'kl':np.nan}, np.nan)
    cdf_path = f"{TRAINING_RESULT_DIR}/CDF_theta_independent_theta{theta_idx:02d}_v15.png"
    print('[v15-theta-train-finished]', flush=True); print(f'theta_idx = {theta_idx}', flush=True); print(f'total epochs = {EPOCHS}', flush=True); print(f'best val cdf_arms = {best_cdf}', flush=True); print(f'best epoch = {best_ep}', flush=True); print(f'final val cdf_arms = {final_val["cdf_arms"]}', flush=True); print(f'final q50_rmse = {final_val["q50_rmse"]}', flush=True); print(f'final total loss = {final_stat["total"]}', flush=True); print(f'final nll loss = {final_stat["nll"]}', flush=True); print(f'final phys loss = {final_stat["phys"]}', flush=True); print(f'final raw kl loss = {final_stat["kl"]}', flush=True); print(f'final weighted kl loss = {final_beta*final_stat["kl"]}', flush=True); print(f'saved model path = {model_path}', flush=True); print(f'saved norm path = {norm_path}', flush=True); print(f'saved config path = {cfg_path}', flush=True); print(f'saved training log path = {log_path}', flush=True); print(f'saved val metrics path = {val_path}', flush=True); print(f'saved cdf path = {cdf_path}', flush=True)
    return final_val

def train_theta_independent_models(case, XMU, XREAL, THETA_FEAT, YH, YP0, YQ0, YPG, YQG, theta_list):
    theta_list_to_train = DEBUG_THETA_LIST if THETA_TRAIN_MODE=='subset' else TRAIN_THETA_LIST
    out=[]
    for k,j in enumerate(theta_list_to_train, start=1):
        out.append(train_single_theta_model(case,XMU,XREAL,YH,YP0,YQ0,YPG,YQG,j,theta_list))
        print(f"[v15-all-theta-progress] finished {k}/{len(theta_list_to_train)} theta models", flush=True)
    df=pd.DataFrame(out); Path(TRAINING_RESULT_DIR).mkdir(parents=True,exist_ok=True)
    df.to_csv(f"{TRAINING_RESULT_DIR}/theta_independent_all_theta_metrics_v15.csv",index=False)
    return df

def combine_theta_independent_flex_domain(case, XMU, YH, theta_list):
    model_paths=[Path(f"{TRAINING_RESULT_DIR}/theta_independent_theta{j:02d}_model.pt") for j in range(N_THETA)]
    norm_paths=[Path(f"{TRAINING_RESULT_DIR}/theta_independent_theta{j:02d}_norm.pkl") for j in range(N_THETA)]
    if any((not m.exists()) or (not n.exists()) for m,n in zip(model_paths,norm_paths)):
        print('[v15-combine] warning: missing theta models, skip combine.', flush=True); return None
    i=0; xmu_one=XMU[i]
    hq={0.05:[],0.50:[],0.95:[]}
    for j in range(N_THETA):
        norm=pickle.load(open(norm_paths[j],'rb'))
        net=BayesSingleThetaGMM2SupportNet(len(xmu_one),case,hidden=160,depth=3,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO).to(DEVICE)
        net.load_state_dict(torch.load(model_paths[j],map_location=DEVICE)); net.eval()
        xt=torch.tensor(((xmu_one.reshape(1,-1)-norm['x_mu_mean'])/norm['x_mu_std']),dtype=torch.float32,device=DEVICE)
        with torch.no_grad(): _,w,mu,sigma=net.forward_gmm_single(xt,sample=False)
        w=w.cpu().numpy().reshape(-1); mu=mu.cpu().numpy().reshape(-1); sigma=sigma.cpu().numpy().reshape(-1)
        hm,hs=float(norm['h_mean'][0,0]),float(norm['h_std'][0,0])
        for q in hq:
            q_pred_norm = float(np.asarray(gmm_quantile(q, w, mu, sigma)).reshape(-1)[0])
            hq[q].append(float(hm + hs * q_pred_norm))
    polys={q:support_values_to_polygon(theta_list,np.array(v,dtype=float)) for q,v in hq.items()}
    for q,p in polys.items():
        if p is None or len(p)<3: print(f'[v15-combine] warning: invalid polygon q={q}', flush=True)
    def close(p): return np.r_[p[:,0],p[0,0]], np.r_[p[:,1],p[0,1]]
    plt.figure(figsize=(7,6),dpi=220)
    for q,c in [(0.05,'#60a5fa'),(0.50,'#1d4ed8'),(0.95,'#0f766e')]:
        p=polys[q]
        if p is not None and len(p)>2:
            x,y=close(p); plt.plot(x,y,color=c,lw=2,label=f'BPINN q{int(q*100):02d}')
    if YH is not None and YH.shape[0]>i:
        ymc=YH[i,:,:]
        for q,c in [(0.05,'#93c5fd'),(0.50,'#3b82f6'),(0.95,'#14b8a6')]:
            h=np.quantile(ymc,q,axis=0); p=support_values_to_polygon(theta_list,h)
            if p is not None and len(p)>2:
                x,y=close(p); plt.plot(x,y,'--',color=c,lw=1.5,label=f'MC q{int(q*100):02d}')
    plt.title('theta-independent v8-style support BPINN')
    plt.axis('equal'); plt.grid(alpha=0.2); plt.legend(); plt.tight_layout()
    out=f"{TRAINING_RESULT_DIR}/FlexDomain_theta_independent_q05_q50_q95_v15.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out,dpi=240); plt.close()
    return out



def draw_theta_external_test_scenario(case, seed):
    rng = np.random.default_rng(seed)
    return sample_scenario_means(case, rng)

def build_external_mc_labels_for_theta(case, theta_idx, theta_list, pd_mu, qd_mu, pr_mu, qr_mu, mc_eval, seed):
    theta=float(theta_list[theta_idx]); alpha=float(np.cos(theta)); beta=float(np.sin(theta))
    rng=np.random.default_rng(seed)
    h_list=[]; p0_list=[]; q0_list=[]; infeasible_count=0
    t0=datetime.now(); report_every=250 if mc_eval>=1000 else 100
    for m in range(mc_eval):
        pd = pd_mu.copy(); pr = pr_mu.copy()
        std_pd = 0.10 * np.maximum(pd_mu, 1e-3)
        std_pr = 0.12 * np.maximum(pr_mu, 1e-3)
        for i in range(case.n_bus):
            if pd_mu[i] > 1e-9:
                pd[i] = sample_trunc_normal(pd_mu[i], std_pd[i], 0.0, None)
        for k, b in enumerate(case.pv_buses):
            pr[b] = sample_trunc_normal(
                pr_mu[b],
                std_pr[b],
                0.0,
                float(case.pv_pmax[k])
            )
        qd = qd_mu * (pd / np.maximum(pd_mu, 1e-6))
        qr = pr * np.tan(np.arccos(case.pv_pf))
        sol = solve_flex_support_gurobi_33bus(case,pd,qd,pr,qr,alpha,beta,return_detail=True,stabilize_dispatch=STABILIZE_OPF_DISPATCH)
        if sol.get('ok',False):
            h_list.append(float(sol['h'])); p0_list.append(float(sol['P0'])); q0_list.append(float(sol['Q0']))
        else:
            infeasible_count += 1
        if (m+1) % report_every == 0 or (m+1)==mc_eval:
            elapsed=(datetime.now()-t0).total_seconds(); eta=(mc_eval-m-1)*(elapsed/max(1,m+1))
            print(f"[p0-style-external-mc] theta_idx={theta_idx:02d} {m+1}/{mc_eval} success={len(h_list)} infeasible={infeasible_count} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    if len(h_list)==0:
        raise RuntimeError(f"[p0-style-external] all MC OPF samples infeasible for theta_idx={theta_idx}")
    return {'theta_idx':theta_idx,'theta':theta,'alpha':alpha,'beta':beta,'x_mu':make_feature_vector(case,pd_mu,pr_mu),'h_mc':np.array(h_list,dtype=float),'p0_mc':np.array(p0_list,dtype=float),'q0_mc':np.array(q0_list,dtype=float),'n_success':len(h_list),'n_infeasible':infeasible_count,'mc_eval':mc_eval}

def load_theta_independent_model_for_eval(case, theta_idx):
    model_path=Path(f"{TRAINING_RESULT_DIR}/theta_independent_theta{theta_idx:02d}_model.pt")
    norm_path=Path(f"{TRAINING_RESULT_DIR}/theta_independent_theta{theta_idx:02d}_norm.pkl")
    config_path=Path(f"{TRAINING_RESULT_DIR}/theta_independent_theta{theta_idx:02d}_config.json")
    if (not model_path.exists()) or (not norm_path.exists()):
        print(f"[p0-style-external-warning] missing model or norm for theta_idx={theta_idx}: {model_path} / {norm_path}", flush=True)
        return None
    with open(norm_path,'rb') as f: norm=pickle.load(f)
    in_dim = int(np.asarray(norm['x_mu_mean']).shape[1])
    net=BayesSingleThetaGMM2SupportNet(in_dim,case,hidden=160,depth=3,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO).to(DEVICE)
    net.load_state_dict(torch.load(model_path,map_location=DEVICE)); net.eval()
    return net,norm,{'model_path':str(model_path),'norm_path':str(norm_path),'config_path':str(config_path)}

def _external_metrics_from_mc_and_model(case, theta_idx, theta_list, net, norm, h_mc, x_mu):
    theta=float(theta_list[theta_idx]); alpha=float(np.cos(theta)); beta=float(np.sin(theta))
    hm=float(norm['h_mean'][0,0]); hs=float(norm['h_std'][0,0])
    margin=max(0.1,0.1*(float(np.max(h_mc))-float(np.min(h_mc))+1e-6))
    z_grid=np.linspace(float(np.min(h_mc))-margin,float(np.max(h_mc))+margin,600)
    cdf_mc=np.searchsorted(np.sort(h_mc),z_grid,side='right')/max(1,len(h_mc))
    x_n=(x_mu.reshape(1,-1)-norm['x_mu_mean'])/norm['x_mu_std']
    xt=torch.tensor(x_n,dtype=torch.float32,device=DEVICE)
    with torch.no_grad(): _,w,mu,sigma=net.forward_gmm_single(xt,sample=False)
    w=w.cpu().numpy().reshape(-1); mu=mu.cpu().numpy().reshape(-1); sigma=sigma.cpu().numpy().reshape(-1)
    z_norm=(z_grid-hm)/(hs+1e-9); cdf_pred=gmm_cdf(z_norm,w,mu,sigma)
    arms=float(np.sqrt(np.mean((cdf_pred-cdf_mc)**2))*100.0)
    q_err={}
    for q in [0.05,0.25,0.50,0.75,0.95]:
        qn=float(np.asarray(gmm_quantile(q,w,mu,sigma)).reshape(-1)[0]); qp=hm+hs*qn; qe=float(np.quantile(h_mc,q)); q_err[q]=float(qp-qe)
    return {'theta_idx':theta_idx,'theta':theta,'alpha':alpha,'beta':beta,'cdf_arms':arms,'q05_error':q_err[0.05],'q25_error':q_err[0.25],'q50_error':q_err[0.50],'q75_error':q_err[0.75],'q95_error':q_err[0.95],'q50_rmse':abs(q_err[0.50]),'h_mc_min':float(np.min(h_mc)),'h_mc_max':float(np.max(h_mc)),'h_mc_mean':float(np.mean(h_mc)),'h_mc_std':float(np.std(h_mc)),'z_grid':z_grid,'cdf_mc':cdf_mc,'cdf_pred':cdf_pred}

def eval_and_plot_single_theta_p0_style_external(case, theta_idx, theta_list, mc_eval=P0_STYLE_SINGLE_SCENARIO_MC, seed=P0_STYLE_EVAL_SEED, save_plot=True):
    loaded=load_theta_independent_model_for_eval(case,theta_idx)
    if loaded is None: return None
    net,norm,paths=loaded
    pd_mu,qd_mu,pr_mu,qr_mu=draw_theta_external_test_scenario(case,seed)
    theta=float(theta_list[theta_idx]); alpha=float(np.cos(theta)); beta=float(np.sin(theta))
    print('[p0-style-external-start]', flush=True)
    print(f'theta_idx = {theta_idx}', flush=True); print(f'theta = {theta}', flush=True); print(f'alpha = {alpha}', flush=True); print(f'beta = {beta}', flush=True)
    print(f'mc_eval = {mc_eval}', flush=True); print(f'seed = {seed}', flush=True); print('source = newly sampled external scenario, not data800 validation', flush=True); print(f"using model path = {paths['model_path']}", flush=True)
    ext=build_external_mc_labels_for_theta(case,theta_idx,theta_list,pd_mu,qd_mu,pr_mu,qr_mu,mc_eval,seed)
    met=_external_metrics_from_mc_and_model(case,theta_idx,theta_list,net,norm,ext['h_mc'],ext['x_mu'])
    plot_path=''
    if save_plot:
        plot_path=f"{P0_STYLE_EXTERNAL_EVAL_DIR}/CDF_theta_independent_theta{theta_idx:02d}_p0style_external_mc{mc_eval}_v15.png"
        Path(plot_path).parent.mkdir(parents=True,exist_ok=True)
        plt.figure(figsize=(7,5),dpi=200)
        plt.plot(met['z_grid'],met['cdf_mc'],'k-',lw=2,label='External MC-OPF empirical CDF')
        plt.plot(met['z_grid'],met['cdf_pred'],'r--',lw=2,label='Predicted GMM CDF')
        plt.title(f"theta_idx={theta_idx:02d}, theta={theta:.4f}, external mc_eval={mc_eval}, ARMS={met['cdf_arms']:.3f}")
        plt.grid(alpha=0.2); plt.legend(); plt.tight_layout(); plt.savefig(plot_path,dpi=220); plt.close()
    out={k:met[k] for k in ['theta_idx','theta','alpha','beta','cdf_arms','q05_error','q25_error','q50_error','q75_error','q95_error','q50_rmse','h_mc_min','h_mc_max','h_mc_mean','h_mc_std']}
    out.update({'eval_type':'p0_style_external_single_scenario','mc_eval':mc_eval,'n_success':ext['n_success'],'n_infeasible':ext['n_infeasible'],'plot_path':plot_path})
    metrics_path=f"{P0_STYLE_EXTERNAL_EVAL_DIR}/theta_independent_theta{theta_idx:02d}_p0style_external_single_metrics_v15.csv"
    pd.DataFrame([out]).to_csv(metrics_path,index=False)
    print('[p0-style-external-finished]', flush=True)
    print(f"theta_idx = {theta_idx}", flush=True); print(f"mc_eval = {mc_eval}", flush=True); print(f"n_success = {ext['n_success']}", flush=True); print(f"n_infeasible = {ext['n_infeasible']}", flush=True)
    print(f"cdf_arms = {out['cdf_arms']}", flush=True); print(f"q50_error = {out['q50_error']}", flush=True); print(f"h_mc_mean = {out['h_mc_mean']}", flush=True); print(f"h_mc_std = {out['h_mc_std']}", flush=True)
    print(f"plot path = {plot_path}", flush=True); print(f"metrics path = {metrics_path}", flush=True)
    return out

def eval_multiple_external_test_scenarios_single_theta(case, theta_idx, theta_list, n_scenarios=P0_STYLE_N_TEST_SCENARIOS, mc_eval_multi=P0_STYLE_MULTI_SCENARIO_MC, seed=P0_STYLE_EVAL_SEED):
    loaded=load_theta_independent_model_for_eval(case,theta_idx)
    if loaded is None: return None
    net,norm,paths=loaded
    rows=[]; total_succ=0; total_inf=0
    for s in range(n_scenarios):
        seed_s=seed+1000+s
        pd_mu,qd_mu,pr_mu,qr_mu=draw_theta_external_test_scenario(case,seed_s)
        ext=build_external_mc_labels_for_theta(case,theta_idx,theta_list,pd_mu,qd_mu,pr_mu,qr_mu,mc_eval_multi,seed_s)
        met=_external_metrics_from_mc_and_model(case,theta_idx,theta_list,net,norm,ext['h_mc'],ext['x_mu'])
        row={'scenario_idx':s,'seed':seed_s,'n_success':ext['n_success'],'n_infeasible':ext['n_infeasible'],**{k:met[k] for k in ['cdf_arms','q05_error','q25_error','q50_error','q75_error','q95_error','q50_rmse']}}
        rows.append(row); total_succ+=ext['n_success']; total_inf+=ext['n_infeasible']
        print(f"[p0-style-external-multi] theta_idx={theta_idx:02d} scenario {s+1:02d}/{n_scenarios} arms={met['cdf_arms']:.4f} n_success={ext['n_success']} n_infeasible={ext['n_infeasible']}", flush=True)
    by_path=f"{P0_STYLE_EXTERNAL_EVAL_DIR}/theta_independent_theta{theta_idx:02d}_p0style_external_multiscen_by_scenario_v15.csv"
    pd.DataFrame(rows).to_csv(by_path,index=False)
    arr=np.array([r['cdf_arms'] for r in rows],dtype=float); q50abs=np.array([abs(r['q50_error']) for r in rows],dtype=float)
    summary={'theta_idx':theta_idx,'n_scenarios':n_scenarios,'mc_eval_multi':mc_eval_multi,'mean_arms':float(np.mean(arr)),'median_arms':float(np.median(arr)),'max_arms':float(np.max(arr)),'q90_arms':float(np.quantile(arr,0.9)),'mean_q50_abs_error':float(np.mean(q50abs)),'max_q50_abs_error':float(np.max(q50abs)),'total_success':int(total_succ),'total_infeasible':int(total_inf)}
    sum_path=f"{P0_STYLE_EXTERNAL_EVAL_DIR}/theta_independent_theta{theta_idx:02d}_p0style_external_multiscen_summary_v15.csv"
    pd.DataFrame([summary]).to_csv(sum_path,index=False)
    return summary

def write_visualization_manifest_theta_independent_v15():
    Path(TRAINING_RESULT_DIR).mkdir(parents=True, exist_ok=True)
    candidates=[f"{TRAINING_RESULT_DIR}/theta_independent_theta02_training_log.csv",f"{TRAINING_RESULT_DIR}/theta_independent_theta02_val_metrics.csv",f"{TRAINING_RESULT_DIR}/CDF_theta_independent_theta02_v15.png",f"{TRAINING_RESULT_DIR}/theta_independent_all_theta_metrics_v15.csv",f"{TRAINING_RESULT_DIR}/FlexDomain_theta_independent_q05_q50_q95_v15.png",f"{TRAINING_RESULT_DIR}/theta_independent_theta02_p0style_external_single_metrics_v15.csv",f"{TRAINING_RESULT_DIR}/theta_independent_theta02_p0style_external_multiscen_by_scenario_v15.csv",f"{TRAINING_RESULT_DIR}/theta_independent_theta02_p0style_external_multiscen_summary_v15.csv"]
    candidates += [str(p) for p in Path(TRAINING_RESULT_DIR).glob('CDF_theta_independent_theta*_p0style_external_mc*_v15.png')]
    rows=[[Path(f).name] for f in candidates if Path(f).exists()]
    pd.DataFrame(rows,columns=['file']).to_csv(f"{TRAINING_RESULT_DIR}/visualization_manifest_theta_independent_v15.csv",index=False)

def main_theta_independent_v5():
    print('[v15-theta-independent] start', flush=True)
    case = build_ieee33_case()
    XMU, XREAL, THETA_FEAT, YH, YP0, YQ0, YPG, YQG, active_records, all_names = get_or_build_flex_dataset_cache(case)
    if RUN_TRAINING:
        train_theta_independent_models(case,XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,THETA_LIST)
    if RUN_COMBINE_THETA_FLEX_DOMAIN or THETA_TRAIN_MODE == 'all':
        combine_theta_independent_flex_domain(case,XMU,YH,THETA_LIST)
    if RUN_P0_STYLE_EXTERNAL_EVAL:
        for theta_idx in P0_STYLE_EVAL_THETA_LIST:
            try:
                eval_and_plot_single_theta_p0_style_external(case,theta_idx,THETA_LIST,mc_eval=P0_STYLE_SINGLE_SCENARIO_MC,seed=P0_STYLE_EVAL_SEED,save_plot=True)
                if P0_STYLE_MULTI_SCENARIO_EVAL:
                    eval_multiple_external_test_scenarios_single_theta(case,theta_idx,THETA_LIST,n_scenarios=P0_STYLE_N_TEST_SCENARIOS,mc_eval_multi=P0_STYLE_MULTI_SCENARIO_MC,seed=P0_STYLE_EVAL_SEED)
            except Exception as e:
                print(f"[p0-style-external-warning] theta_idx={theta_idx} skipped due to error: {e}", flush=True)
    write_visualization_manifest_theta_independent_v15()
    print('[v15-theta-independent] done', flush=True)



# ===== v5 v7-style theta-independent branch =====
import time
V7STYLE_EXPERIMENT = True
V7STYLE_RUN_TAG = "v5_theta_independent_v7style_worst_theta"
USE_V7STYLE_QUANTILE_PHYSICS = True
FIXED_QUANTILE_TAUS = [0.05, 0.25, 0.50, 0.75, 0.95]
USE_RANDOM_PHYS_TAUS = True
N_RANDOM_PHYS_TAUS = 4
RANDOM_TAU_LOW = 0.02
RANDOM_TAU_HIGH = 0.98
TAIL_WEIGHTED_PHYS = False
DETACH_QUANTILE_Z = False
N_GMM_COMPONENTS_V7STYLE = 2
EVAL_POSTERIOR_SAMPLES = 100
BAND_METHOD = "percentile"
BAND_LO = 0.05
BAND_HI = 0.95
MEASURE_TRAIN_TIME = True
MEASURE_PREDICT_TIME = True
PREDICT_TIMING_REPEAT = 1000
RUN_COMBINE_THETA_FLEX_DOMAIN = False
P0_STYLE_EVAL_THETA_LIST = DEBUG_THETA_LIST

COMPARISON_MODE = True
RUN_COMPARE_THETA02_V7_VS_V8 = True
RUN_COMPARE_THETA06_MULTI_SCENARIO = True
COMPARE_THETA02_SEED = 1
COMPARE_THETA02_MC = 2500
COMPARE_THETA06_SEEDS = [1,2,3,4,5]
COMPARE_THETA06_MC = 800
TRAIN_V7_MODEL_IF_MISSING = True
SKIP_TRAIN_IF_MODEL_EXISTS = True
V7STYLE_RESULT_DIR = "training_results_theta_independent_v7style"
V8STYLE_RESULT_DIR = "training_results_theta_independent"
COMPARISON_RESULT_DIR = "training_results_theta_independent_comparison"
V7_MODEL_TEMPLATE = "theta_independent_theta{theta_idx:02d}_v7style_model.pt"
V7_NORM_TEMPLATE = "theta_independent_theta{theta_idx:02d}_v7style_norm.pkl"
V7_CONFIG_TEMPLATE = "theta_independent_theta{theta_idx:02d}_v7style_config.json"
V8_MODEL_TEMPLATE = "theta_independent_theta{theta_idx:02d}_model.pt"
V8_NORM_TEMPLATE = "theta_independent_theta{theta_idx:02d}_norm.pkl"
V8_CONFIG_TEMPLATE = "theta_independent_theta{theta_idx:02d}_config.json"


def flatten_single_theta_dataset_v7style(XMU, XREAL, YH, YP0, YQ0, YPG, YQG, theta_idx, theta_list):
    d=flatten_single_theta_dataset(XMU,XREAL,YH,YP0,YQ0,YPG,YQG,theta_idx,theta_list)
    N,M=YH.shape[0],YH.shape[1]
    d.update({'XMU_scen':XMU,'YH_theta_scen':YH[:,:,theta_idx],'YP0_theta_scen':YP0[:,:,theta_idx],'YQ0_theta_scen':YQ0[:,:,theta_idx],'YT_theta_scen':(-d['beta']*YP0[:,:,theta_idx]+d['alpha']*YQ0[:,:,theta_idx]),'YPG_theta_scen':YPG[:,:,theta_idx,:],'YQG_theta_scen':YQG[:,:,theta_idx,:]})
    return d

class BayesSingleThetaV7StyleGMM2SupportNet(nn.Module):
    def __init__(self,in_dim,case,hidden=160,depth=3,prior_sigma=1.0,init_rho=-5.0):
        super().__init__(); self.n_gen=len(case.gen_buses); self.layers=nn.ModuleList(); d=in_dim
        for _ in range(depth): self.layers.append(BayesLinear(d,hidden,prior_sigma=prior_sigma,init_rho=init_rho)); d=hidden
        self.gmm_out=BayesLinear(hidden,6,prior_sigma=prior_sigma,init_rho=init_rho); self.rec_out=BayesLinear(hidden+1,1+2*self.n_gen,prior_sigma=prior_sigma,init_rho=init_rho); self.act=nn.ReLU()
        self.register_buffer('pg_min_t',torch.tensor(case.pg_min,dtype=torch.float32).view(1,-1)); self.register_buffer('pg_max_t',torch.tensor(case.pg_max,dtype=torch.float32).view(1,-1)); self.register_buffer('qg_min_t',torch.tensor(case.qg_min,dtype=torch.float32).view(1,-1)); self.register_buffer('qg_max_t',torch.tensor(case.qg_max,dtype=torch.float32).view(1,-1))
    def encode(self,x_mu_norm,sample=True):
        h=x_mu_norm
        for l in self.layers: h=self.act(l(h,sample=sample))
        return h
    def gmm_head(self,h,sample=True):
        out=self.gmm_out(h,sample=sample); w=torch.softmax(out[:,:2],dim=1); mu=torch.cat([out[:,2:3],out[:,4:5]],dim=1); sigma=torch.cat([torch.nn.functional.softplus(out[:,3:4])+1e-3,torch.nn.functional.softplus(out[:,5:6])+1e-3],dim=1); return w,mu,sigma
    def forward_gmm_single(self,x_mu_norm,sample=True):
        h=self.encode(x_mu_norm,sample=sample); w,mu,sigma=self.gmm_head(h,sample=sample); return h,w,mu,sigma
    def recover_dispatch_from_h(self,h_feature,h_label,h_mean,h_std,t_mean,t_std,sample=True):
        h_norm=(h_label-h_mean)/(h_std+1e-9); out=self.rec_out(torch.cat([h_feature,h_norm],dim=1),sample=sample); t_norm=torch.tanh(out[:,0:1]); pg_raw=out[:,1:1+self.n_gen]; qg_raw=out[:,1+self.n_gen:1+2*self.n_gen]
        t=t_mean+t_std*t_norm; pg=self.pg_min_t+torch.sigmoid(pg_raw)*(self.pg_max_t-self.pg_min_t); qg=self.qg_min_t+torch.sigmoid(qg_raw)*(self.qg_max_t-self.qg_min_t); return t,pg,qg
    def kl_divergence(self): return sum(l.kl_divergence() for l in self.layers)+self.gmm_out.kl_divergence()+self.rec_out.kl_divergence()

def make_v7style_physics_taus(training,device,dtype):
    taus=list(FIXED_QUANTILE_TAUS)
    if training and USE_RANDOM_PHYS_TAUS:
        rng=np.random.default_rng(); taus += rng.uniform(RANDOM_TAU_LOW,RANDOM_TAU_HIGH,size=N_RANDOM_PHYS_TAUS).tolist()
    taus=np.array(sorted(taus),dtype=float)
    return torch.tensor(taus,dtype=dtype,device=device)

def physics_loss_flex_quantile_v7style(case,x_mu_raw,h_quantiles,t_quantiles,pg_quantiles,qg_quantiles,alpha,beta,tau_weights=None,return_parts=False):
    B,K=h_quantiles.shape; h=h_quantiles.reshape(-1,1); t=t_quantiles.reshape(-1,1); p0=alpha*h-beta*t; q0=beta*h+alpha*t
    pg=pg_quantiles.reshape(B*K,-1); qg=qg_quantiles.reshape(B*K,-1); xrep=x_mu_raw.repeat_interleave(K,dim=0)
    return physics_loss_flex(case,xrep,p0,q0,pg,qg)

def train_single_theta_v7style_model(case,XMU,XREAL,YH,YP0,YQ0,YPG,YQG,theta_idx,theta_list):
    Path(TRAINING_RESULT_DIR).mkdir(parents=True,exist_ok=True); d=flatten_single_theta_dataset_v7style(XMU,XREAL,YH,YP0,YQ0,YPG,YQG,theta_idx,theta_list)
    rng=np.random.default_rng(SEED_TRAIN); n_scen=XMU.shape[0]; mc=YH.shape[1]; perm=rng.permutation(n_scen); n_val=max(1,int(0.1*n_scen)); va=perm[-n_val:]; tr=perm[:-n_val]; sf=np.repeat(np.arange(n_scen),mc); trm=np.isin(sf,tr); vam=np.isin(sf,va)
    xmu_tr=d['xmu_flat'][trm]; yh_tr=d['yh_flat'][trm]; yt_tr=d['yt_flat'][trm]; ypg_tr=d['ypg_flat'][trm]; yqg_tr=d['yqg_flat'][trm]
    norm={'x_mu_mean':xmu_tr.mean(0,keepdims=True),'x_mu_std':xmu_tr.std(0,keepdims=True)+1e-9,'h_mean':np.array([[yh_tr.mean()]]),'h_std':np.array([[yh_tr.std()+1e-9]]),'t_mean':np.array([[yt_tr.mean()]]),'t_std':np.array([[yt_tr.std()+1e-9]]),'alpha':d['alpha'],'beta':d['beta'],'theta':d['theta'],'theta_idx':theta_idx}
    t=lambda a: torch.tensor(a,dtype=torch.float32,device=DEVICE)
    xmu_n=t((xmu_tr-norm['x_mu_mean'])/norm['x_mu_std']); xmu_raw=t(xmu_tr); yh=t(yh_tr); yt=t(yt_tr); ypg=t(ypg_tr); yqg=t(yqg_tr); hm,hs,tm,ts=t(norm['h_mean']),t(norm['h_std']),t(norm['t_mean']),t(norm['t_std'])
    net=BayesSingleThetaV7StyleGMM2SupportNet(XMU.shape[1],case,hidden=160,depth=3,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO).to(DEVICE); opt=torch.optim.Adam(net.parameters(),lr=LR)
    n=xmu_n.shape[0]; nb=(n+BATCH_SIZE-1)//BATCH_SIZE; rows=[]; et=[]; train_start=time.perf_counter()
    for ep in range(EPOCHS):
      epst=time.perf_counter(); p=np.random.default_rng(SEED_TRAIN+ep).permutation(n); beta_kl=min(1.0,(ep+1)/max(1,KL_WARMUP_EPOCHS))*BETA_KL_MAX; st={k:0.0 for k in ['total','nll','dispatch','t','phys','kl']}
      for b in range(nb):
        ii=p[b*BATCH_SIZE:min((b+1)*BATCH_SIZE,n)]; hf,w,mu,sigma=net.forward_gmm_single(xmu_n[ii],sample=True); hnorm=(yh[ii]-hm)/(hs+1e-9); nll=(-gmm_log_prob(hnorm,w,mu,sigma)).mean(); t_hat,pg_hat,qg_hat=net.recover_dispatch_from_h(hf,yh[ii],hm,hs,tm,ts,sample=True)
        t_sup=((t_hat-yt[ii])**2).mean(); disp=((pg_hat-ypg[ii])**2).mean()+((qg_hat-yqg[ii])**2).mean(); taus=make_v7style_physics_taus(True,xmu_n.device,xmu_n.dtype); hq_n=gmm_quantile_torch(w,mu,sigma,taus); hq=hm+hs*hq_n
        tq=[]; pgq=[]; qgq=[]
        for k in range(taus.numel()):
          tk,pgk,qgk=net.recover_dispatch_from_h(hf,hq[:,k:k+1],hm,hs,tm,ts,sample=True); tq.append(tk); pgq.append(pgk); qgq.append(qgk)
        tq=torch.cat(tq,dim=1); pgq=torch.stack(pgq,dim=1); qgq=torch.stack(qgq,dim=1)
        phys=physics_loss_flex_quantile_v7style(case,xmu_raw[ii],hq,tq,pgq,qgq,d['alpha'],d['beta']); kl=net.kl_divergence()/max(1,n); loss=nll+LAM_T_SUP*t_sup+LAM_DISPATCH_SUP*disp+LAM_PHYS_FLEX*phys+beta_kl*kl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
        for k,v in zip(st.keys(),[loss,nll,disp,t_sup,phys,kl]): st[k]+=float(v.detach().cpu())
      for k in st: st[k]/=max(1,nb)
      ep_time=time.perf_counter()-epst; et.append(ep_time); cum=time.perf_counter()-train_start
      rows.append({'epoch':ep+1,'total_loss':st['total'],'nll_loss':st['nll'],'dispatch_sup_loss':st['dispatch'],'t_sup_loss':st['t'],'phys_loss':st['phys'],'kl_loss':st['kl'],'epoch_time_sec':ep_time,'cumulative_train_time_sec':cum,'samples_per_sec_epoch':n/max(1e-9,ep_time)})
      if should_print_epoch(ep,EPOCHS): print(f"[v7style-train] theta_idx={theta_idx:02d} epoch {ep+1}/{EPOCHS} total={st['total']:.4f}",flush=True)
    tag=f"theta_independent_theta{theta_idx:02d}_v7style"; torch.save(net.state_dict(),f"{TRAINING_RESULT_DIR}/{tag}_model.pt"); pickle.dump(norm,open(f"{TRAINING_RESULT_DIR}/{tag}_norm.pkl","wb")); json.dump({'theta_idx':theta_idx,'theta':d['theta'],'alpha':d['alpha'],'beta':d['beta'],'run_tag':V7STYLE_RUN_TAG},open(f"{TRAINING_RESULT_DIR}/{tag}_config.json","w"),indent=2)
    pd.DataFrame(rows).to_csv(f"{TRAINING_RESULT_DIR}/{tag}_training_log.csv",index=False)
    print('[v7style-train-time]',flush=True); print(f'theta_idx = {theta_idx}',flush=True); print(f'total_train_time_sec = {time.perf_counter()-train_start:.3f}',flush=True); print(f'mean_epoch_time_sec = {float(np.mean(et)):.3f}',flush=True); print(f'min_epoch_time_sec = {float(np.min(et)):.3f}',flush=True); print(f'max_epoch_time_sec = {float(np.max(et)):.3f}',flush=True); print(f'samples_per_sec = {n*EPOCHS/max(1e-9,time.perf_counter()-train_start):.3f}',flush=True); print(f'train_samples = {n}',flush=True); print(f'epochs = {EPOCHS}',flush=True)
    return net,norm,d

def eval_single_theta_v7style_external(case,theta_idx,theta_list,net,norm,mc_eval=P0_STYLE_SINGLE_SCENARIO_MC,seed=P0_STYLE_EVAL_SEED):
    t0=time.perf_counter(); pd_mu,qd_mu,pr_mu,qr_mu=draw_theta_external_test_scenario(case,seed); ext=build_external_mc_labels_for_theta(case,theta_idx,theta_list,pd_mu,qd_mu,pr_mu,qr_mu,mc_eval,seed); opf_t=time.perf_counter()-t0
    xmu=ext['x_mu']; xn=(xmu.reshape(1,-1)-norm['x_mu_mean'])/norm['x_mu_std']; xt=torch.tensor(xn,dtype=torch.float32,device=DEVICE)
    if DEVICE=='cuda': torch.cuda.synchronize()
    ts=[]
    for _ in range(PREDICT_TIMING_REPEAT):
      t1=time.perf_counter();
      with torch.no_grad(): hf,w,mu,sigma=net.forward_gmm_single(xt,sample=False)
      if DEVICE=='cuda': torch.cuda.synchronize(); ts.append((time.perf_counter()-t1)*1000)
      else: ts.append((time.perf_counter()-t1)*1000)
    hm=float(norm['h_mean'][0,0]); hs=float(norm['h_std'][0,0]); hmc=ext['h_mc']; z=np.linspace(hmc.min()-0.2,hmc.max()+0.2,600); cdf_mc=np.searchsorted(np.sort(hmc),z,side='right')/len(hmc)
    w=w.cpu().numpy().reshape(-1); mu=mu.cpu().numpy().reshape(-1); sigma=sigma.cpu().numpy().reshape(-1)
    c0=time.perf_counter(); cdf_det=gmm_cdf((z-hm)/(hs+1e-9),w,mu,sigma); cdf_ms=(time.perf_counter()-c0)*1000
    s0=time.perf_counter(); cdf_s=[]; qpost={0.05:[],0.5:[],0.95:[]}
    for _ in range(EVAL_POSTERIOR_SAMPLES):
      with torch.no_grad(): _,ws,mus,sigs=net.forward_gmm_single(xt,sample=True)
      ws=ws.cpu().numpy().reshape(-1); mus=mus.cpu().numpy().reshape(-1); sigs=sigs.cpu().numpy().reshape(-1); cdf_s.append(gmm_cdf((z-hm)/(hs+1e-9),ws,mus,sigs))
      for q in qpost: qpost[q].append(hm+hs*float(np.asarray(gmm_quantile(q,ws,mus,sigs)).reshape(-1)[0]))
    band_t=time.perf_counter()-s0; cdf_s=np.asarray(cdf_s)
    p05=np.nanquantile(cdf_s,BAND_LO,axis=0); p50=np.nanquantile(cdf_s,0.5,axis=0); p95=np.nanquantile(cdf_s,BAND_HI,axis=0)
    Path(TRAINING_RESULT_DIR).mkdir(parents=True,exist_ok=True); fig=f"{TRAINING_RESULT_DIR}/CDF_theta_independent_theta{theta_idx:02d}_v7style_external_mc{mc_eval}_bandfix.png"
    plt.figure(figsize=(7,5),dpi=180); plt.plot(z,cdf_mc,'k-',lw=2,label='External MC-OPF empirical CDF'); plt.plot(z,cdf_det,'r--',lw=2,label='Deterministic GMM CDF');
    for i in range(min(15,cdf_s.shape[0])): plt.plot(z,cdf_s[i],color='gray',lw=0.6,alpha=0.35)
    plt.fill_between(z,p05,p95,color='#93c5fd',alpha=0.45,label='posterior CDF 5–95% band (epistemic)'); plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(fig,dpi=220); plt.close()
    met={'theta_idx':theta_idx,'external_cdf_arms':float(np.sqrt(np.mean((cdf_det-cdf_mc)**2))*100),'deterministic_cdf_arms':float(np.sqrt(np.mean((cdf_det-cdf_mc)**2))*100),'posterior_mean_cdf_arms':float(np.sqrt(np.mean((np.nanmean(cdf_s,axis=0)-cdf_mc)**2))*100),'posterior_median_cdf_arms':float(np.sqrt(np.mean((p50-cdf_mc)**2))*100),'band_coverage_grid':float(np.mean((cdf_mc>=p05)&(cdf_mc<=p95))),'mean_band_width':float(np.mean(p95-p05)),'max_band_width':float(np.max(p95-p05)),'h_mc_mean':float(np.mean(hmc)),'h_mc_std':float(np.std(hmc)),'n_success':int(ext['n_success']),'n_infeasible':int(ext['n_infeasible']),'opf_mc_total_time_sec':opf_t,'opf_per_sample_ms':opf_t*1000/max(1,mc_eval),'nn_forward_mean_ms':float(np.mean(ts)),'nn_forward_std_ms':float(np.std(ts)),'cdf_compute_time_ms':cdf_ms,'posterior_cdf_band_time_sec':band_t,'total_prediction_eval_time_sec':band_t+(sum(ts)/1000.0)+cdf_ms/1000.0}
    for q in [0.05,0.25,0.5,0.75,0.95]:
      qemp=float(np.quantile(hmc,q)); qdet=hm+hs*float(np.asarray(gmm_quantile(q,w,mu,sigma)).reshape(-1)[0]); met[f'external_q{int(q*100):02d}_error']=float(qdet-qemp)
    met.update({'pred_q05_post_mean':float(np.mean(qpost[0.05])),'pred_q50_post_mean':float(np.mean(qpost[0.5])),'pred_q95_post_mean':float(np.mean(qpost[0.95])),'pred_q05_post_p05':float(np.quantile(qpost[0.05],0.05)),'pred_q05_post_p95':float(np.quantile(qpost[0.05],0.95)),'pred_q50_post_p05':float(np.quantile(qpost[0.5],0.05)),'pred_q50_post_p95':float(np.quantile(qpost[0.5],0.95)),'pred_q95_post_p05':float(np.quantile(qpost[0.95],0.05)),'pred_q95_post_p95':float(np.quantile(qpost[0.95],0.95)),'mc_q05':float(np.quantile(hmc,0.05)),'mc_q50':float(np.quantile(hmc,0.5)),'mc_q95':float(np.quantile(hmc,0.95))})
    out_csv=f"{TRAINING_RESULT_DIR}/theta_independent_theta{theta_idx:02d}_v7style_external_single_metrics.csv"; pd.DataFrame([met]).to_csv(out_csv,index=False)
    print('[nn-predict-time]',flush=True); print(f'theta_idx = {theta_idx}',flush=True); print(f'repeat = {PREDICT_TIMING_REPEAT}',flush=True); print(f'mean_forward_ms = {met["nn_forward_mean_ms"]}',flush=True); print(f'std_forward_ms = {met["nn_forward_std_ms"]}',flush=True); print(f'min_forward_ms = {float(np.min(ts))}',flush=True); print(f'max_forward_ms = {float(np.max(ts))}',flush=True); print(f'cdf_compute_ms = {cdf_ms}',flush=True); print(f'posterior_band_time_sec = {band_t}',flush=True); print(f'opf_mc_total_time_sec = {opf_t}',flush=True); print(f'opf_per_sample_ms = {met["opf_per_sample_ms"]}',flush=True)
    return met

def main_theta_independent_v5():
    print('[v5-theta-independent] start',flush=True); case=build_ieee33_case(); XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active_records,all_names=get_or_build_flex_dataset_cache(case)
    theta_list_to_train=DEBUG_THETA_LIST if THETA_TRAIN_MODE=='subset' else TRAIN_THETA_LIST
    for j in theta_list_to_train:
      if RUN_TRAINING:
        net,norm,_=train_single_theta_v7style_model(case,XMU,XREAL,YH,YP0,YQ0,YPG,YQG,j,THETA_LIST)
      else:
        mp=Path(f"{TRAINING_RESULT_DIR}/theta_independent_theta{j:02d}_v7style_model.pt"); npk=Path(f"{TRAINING_RESULT_DIR}/theta_independent_theta{j:02d}_v7style_norm.pkl")
        if (not mp.exists()) or (not npk.exists()): print(f"[v5-warning] missing model for theta_idx={j}",flush=True); continue
        norm=pickle.load(open(npk,'rb')); net=BayesSingleThetaV7StyleGMM2SupportNet(XMU.shape[1],case,hidden=160,depth=3,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO).to(DEVICE); net.load_state_dict(torch.load(mp,map_location=DEVICE)); net.eval()
      if RUN_P0_STYLE_EXTERNAL_EVAL: eval_single_theta_v7style_external(case,j,THETA_LIST,net,norm,mc_eval=P0_STYLE_SINGLE_SCENARIO_MC,seed=P0_STYLE_EVAL_SEED)
    print('[v5-theta-independent] done',flush=True)



def build_external_reference_for_theta(case, theta_idx, theta_list, seed, mc_eval):
    pd_mu, qd_mu, pr_mu, qr_mu = draw_theta_external_test_scenario(case, seed)
    np.random.seed(seed + 100000 + theta_idx)
    theta=float(theta_list[theta_idx]); alpha=float(np.cos(theta)); beta=float(np.sin(theta))
    h_list=[]; p0_list=[]; q0_list=[]; infeasible=0
    t0=time.perf_counter()
    std_pd = 0.10 * np.maximum(pd_mu, 1e-3)
    std_pr = 0.12 * np.maximum(pr_mu, 1e-3)
    for _ in range(mc_eval):
        pd = pd_mu.copy(); pr = pr_mu.copy()
        for i in range(case.n_bus):
            if pd_mu[i] > 1e-9: pd[i] = sample_trunc_normal(pd_mu[i], std_pd[i], 0.0, None)
        for k,b in enumerate(case.pv_buses):
            pr[b] = sample_trunc_normal(pr_mu[b], std_pr[b], 0.0, float(case.pv_pmax[k]))
        qd = qd_mu * (pd / np.maximum(pd_mu, 1e-6)); qr = pr * math.tan(math.acos(case.pv_pf))
        sol=solve_flex_support_gurobi_33bus(case,pd,qd,pr,qr,alpha,beta,return_detail=True,stabilize_dispatch=STABILIZE_OPF_DISPATCH)
        if sol.get('ok',False):
            h_list.append(float(sol['h'])); p0_list.append(float(sol['P0'])); q0_list.append(float(sol['Q0']))
        else: infeasible += 1
    if len(h_list)==0: raise RuntimeError(f'[compare] all infeasible theta_idx={theta_idx} seed={seed}')
    opf_t=time.perf_counter()-t0
    return {'theta_idx':theta_idx,'theta':theta,'alpha':alpha,'beta':beta,'seed':seed,'mc_eval':mc_eval,'x_mu':make_feature_vector(case,pd_mu,pr_mu),'pd_mu':pd_mu,'qd_mu':qd_mu,'pr_mu':pr_mu,'qr_mu':qr_mu,'h_mc':np.array(h_list),'p0_mc':np.array(p0_list),'q0_mc':np.array(q0_list),'n_success':len(h_list),'n_infeasible':infeasible,'opf_mc_total_time_sec':opf_t,'opf_per_sample_ms':opf_t*1000/max(1,mc_eval)}

def ensure_v7style_model_for_theta(case, theta_idx, data_dict):
    mp=Path(V7STYLE_RESULT_DIR)/V7_MODEL_TEMPLATE.format(theta_idx=theta_idx)
    npk=Path(V7STYLE_RESULT_DIR)/V7_NORM_TEMPLATE.format(theta_idx=theta_idx)
    if mp.exists() and npk.exists() and SKIP_TRAIN_IF_MODEL_EXISTS:
        norm=pickle.load(open(npk,'rb')); net=BayesSingleThetaV7StyleGMM2SupportNet(data_dict[0].shape[1],case,hidden=160,depth=3,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO).to(DEVICE); net.load_state_dict(torch.load(mp,map_location=DEVICE)); net.eval(); return net,norm,{'model_path':str(mp),'status':'loaded'}
    if (not mp.exists() or not npk.exists()) and TRAIN_V7_MODEL_IF_MISSING:
        XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,*_ = data_dict
        old=DEBUG_THETA_LIST.copy();
        net,norm,_=train_single_theta_v7style_model(case,XMU,XREAL,YH,YP0,YQ0,YPG,YQG,theta_idx,THETA_LIST)
        return net,norm,{'model_path':str(mp),'status':'trained'}
    print(f'[compare-warning] missing v7 model theta={theta_idx}',flush=True); return None,None,{'model_path':str(mp),'status':'missing_model'}

def load_v8style_theta_model_for_compare(case, theta_idx):
    mp=Path(V8STYLE_RESULT_DIR)/V8_MODEL_TEMPLATE.format(theta_idx=theta_idx)
    npk=Path(V8STYLE_RESULT_DIR)/V8_NORM_TEMPLATE.format(theta_idx=theta_idx)
    if (not mp.exists()) or (not npk.exists()):
        print(f'[compare-warning] missing v8 model theta={theta_idx}',flush=True); return None
    norm=pickle.load(open(npk,'rb')); net=BayesSingleThetaGMM2SupportNet(len(norm['x_mu_mean'][0]),case,hidden=160,depth=3,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO).to(DEVICE); net.load_state_dict(torch.load(mp,map_location=DEVICE)); net.eval(); return net,norm,{'model_path':str(mp),'status':'loaded'}

def _timed_forward(net,xt,sample=False):
    ts=[]
    for _ in range(PREDICT_TIMING_REPEAT):
        if DEVICE=='cuda': torch.cuda.synchronize()
        t0=time.perf_counter();
        with torch.no_grad(): out=net.forward_gmm_single(xt,sample=sample)
        if DEVICE=='cuda': torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    return out,ts

def predict_cdf_v8style_on_external_reference(case,theta_idx,ref,z_grid):
    loaded=load_v8style_theta_model_for_compare(case,theta_idx)
    if loaded is None: return {'method':'v8style','theta_idx':theta_idx,'seed':ref['seed'],'mc_eval':ref['mc_eval'],'status':'missing_model'},None
    net,norm,paths=loaded; x=(ref['x_mu'].reshape(1,-1)-norm['x_mu_mean'])/norm['x_mu_std']; xt=torch.tensor(x,dtype=torch.float32,device=DEVICE)
    (_,w,mu,sigma),ts=_timed_forward(net,xt,sample=False); w=w.cpu().numpy().reshape(-1); mu=mu.cpu().numpy().reshape(-1); sigma=sigma.cpu().numpy().reshape(-1)
    hm=float(norm['h_mean'][0,0]); hs=float(norm['h_std'][0,0]); cdf=gmm_cdf((z_grid-hm)/(hs+1e-9),w,mu,sigma); cdf_mc=np.searchsorted(np.sort(ref['h_mc']),z_grid,side='right')/len(ref['h_mc'])
    m={'method':'v8style','theta_idx':theta_idx,'seed':ref['seed'],'mc_eval':ref['mc_eval'],'external_cdf_arms':float(np.sqrt(np.mean((cdf-cdf_mc)**2))*100),'posterior_mean_cdf_arms':np.nan,'q50_error':float(hm+hs*float(np.asarray(gmm_quantile(0.5,w,mu,sigma)).reshape(-1)[0])-np.quantile(ref['h_mc'],0.5)),'q95_error':float(hm+hs*float(np.asarray(gmm_quantile(0.95,w,mu,sigma)).reshape(-1)[0])-np.quantile(ref['h_mc'],0.95)),'opf_per_sample_ms':ref['opf_per_sample_ms'],'nn_forward_mean_ms':float(np.mean(ts)),'status':'ok','model_path':paths['model_path']}
    return m,cdf

def predict_cdf_v7style_on_external_reference(case,theta_idx,ref,z_grid,save_band_plot=False,output_prefix=None,net=None,norm=None):
    if net is None or norm is None:
        loaded=ensure_v7style_model_for_theta(case,theta_idx,get_or_build_flex_dataset_cache(case)); net,norm,_=loaded
    x=(ref['x_mu'].reshape(1,-1)-norm['x_mu_mean'])/norm['x_mu_std']; xt=torch.tensor(x,dtype=torch.float32,device=DEVICE)
    (_,w,mu,sigma),ts=_timed_forward(net,xt,sample=False); w=w.cpu().numpy().reshape(-1); mu=mu.cpu().numpy().reshape(-1); sigma=sigma.cpu().numpy().reshape(-1); hm=float(norm['h_mean'][0,0]); hs=float(norm['h_std'][0,0])
    c0=time.perf_counter(); cdf=gmm_cdf((z_grid-hm)/(hs+1e-9),w,mu,sigma); cdf_ms=(time.perf_counter()-c0)*1000
    cdf_mc=np.searchsorted(np.sort(ref['h_mc']),z_grid,side='right')/len(ref['h_mc'])
    s0=time.perf_counter(); cs=[]
    for _ in range(EVAL_POSTERIOR_SAMPLES):
        with torch.no_grad(): _,ws,mus,sigs=net.forward_gmm_single(xt,sample=True)
        cs.append(gmm_cdf((z_grid-hm)/(hs+1e-9),ws.cpu().numpy().reshape(-1),mus.cpu().numpy().reshape(-1),sigs.cpu().numpy().reshape(-1)))
    cs=np.asarray(cs); band_t=time.perf_counter()-s0; p05=np.nanquantile(cs,0.05,axis=0); p50=np.nanquantile(cs,0.5,axis=0); p95=np.nanquantile(cs,0.95,axis=0)
    if save_band_plot and output_prefix:
        Path(V7STYLE_RESULT_DIR).mkdir(parents=True,exist_ok=True); fig=f"{V7STYLE_RESULT_DIR}/CDF_theta_independent_theta{theta_idx:02d}_v7style_external_seed{ref['seed']:03d}_mc{ref['mc_eval']}_bandfix.png"; plt.figure(figsize=(7,5),dpi=180); plt.plot(z_grid,cdf_mc,'k-',lw=2,label='External MC-OPF empirical CDF'); plt.plot(z_grid,cdf,'r--',lw=2,label='v7 deterministic GMM CDF'); plt.fill_between(z_grid,p05,p95,color='#93c5fd',alpha=0.4,label='v7 posterior CDF 5-95%'); plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(fig,dpi=220); plt.close()
    m={'method':'v7style','theta_idx':theta_idx,'seed':ref['seed'],'mc_eval':ref['mc_eval'],'external_cdf_arms':float(np.sqrt(np.mean((cdf-cdf_mc)**2))*100),'posterior_mean_cdf_arms':float(np.sqrt(np.mean((np.nanmean(cs,axis=0)-cdf_mc)**2))*100),'posterior_median_cdf_arms':float(np.sqrt(np.mean((p50-cdf_mc)**2))*100),'q05_error':float(hm+hs*float(np.asarray(gmm_quantile(0.05,w,mu,sigma)).reshape(-1)[0])-np.quantile(ref['h_mc'],0.05)),'q25_error':float(hm+hs*float(np.asarray(gmm_quantile(0.25,w,mu,sigma)).reshape(-1)[0])-np.quantile(ref['h_mc'],0.25)),'q50_error':float(hm+hs*float(np.asarray(gmm_quantile(0.5,w,mu,sigma)).reshape(-1)[0])-np.quantile(ref['h_mc'],0.5)),'q75_error':float(hm+hs*float(np.asarray(gmm_quantile(0.75,w,mu,sigma)).reshape(-1)[0])-np.quantile(ref['h_mc'],0.75)),'q95_error':float(hm+hs*float(np.asarray(gmm_quantile(0.95,w,mu,sigma)).reshape(-1)[0])-np.quantile(ref['h_mc'],0.95)),'band_coverage_grid':float(np.mean((cdf_mc>=p05)&(cdf_mc<=p95))),'mean_band_width':float(np.mean(p95-p05)),'max_band_width':float(np.max(p95-p05)),'opf_per_sample_ms':ref['opf_per_sample_ms'],'nn_forward_mean_ms':float(np.mean(ts)),'nn_forward_std_ms':float(np.std(ts)),'nn_forward_min_ms':float(np.min(ts)),'nn_forward_max_ms':float(np.max(ts)),'posterior_cdf_band_time_sec':band_t,'cdf_compute_time_ms':cdf_ms,'total_prediction_eval_time_sec':band_t+cdf_ms/1000+sum(ts)/1000,'status':'ok'}
    return m,cdf,p05,p95,cdf_mc

def run_compare_theta02_v7_vs_v8(case,data_dict):
    theta_idx=2; seed=COMPARE_THETA02_SEED; mc=COMPARE_THETA02_MC; Path(COMPARISON_RESULT_DIR).mkdir(parents=True,exist_ok=True)
    net,norm,mp=ensure_v7style_model_for_theta(case,theta_idx,data_dict)
    ref=build_external_reference_for_theta(case,theta_idx,THETA_LIST,seed,mc); h=ref['h_mc']; margin=0.05*(h.max()-h.min()+1e-9); z=np.linspace(h.min()-margin,h.max()+margin,600)
    v7,c7,p05,p95,cdf_mc=predict_cdf_v7style_on_external_reference(case,theta_idx,ref,z,save_band_plot=True,output_prefix='t02',net=net,norm=norm)
    v8,c8=predict_cdf_v8style_on_external_reference(case,theta_idx,ref,z)
    plt.figure(figsize=(7,5),dpi=200); plt.plot(z,cdf_mc,'k-',lw=2,label='External MC-OPF empirical CDF'); plt.plot(z,c7,'r--',lw=2,label='v7 deterministic'); plt.fill_between(z,p05,p95,color='#93c5fd',alpha=0.4,label='v7 posterior 5-95%')
    if c8 is not None: plt.plot(z,c8,color='orange',ls='-.',lw=2,label='v8 deterministic')
    plt.title(f"theta=02 seed={seed} mc={mc} v7={v7['external_cdf_arms']:.3f} v8={v8.get('external_cdf_arms',np.nan):.3f}"); plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(f"{COMPARISON_RESULT_DIR}/compare_theta02_v7_vs_v8_seed{seed:03d}_mc{mc}_cdf.png",dpi=220); plt.close()
    df=pd.DataFrame([v7,v8]); df.to_csv(f"{COMPARISON_RESULT_DIR}/compare_theta02_v7_vs_v8_seed{seed:03d}_mc{mc}_metrics.csv",index=False)
    return df

def run_theta06_v7style_multiseed_external(case,data_dict):
    theta_idx=6; net,norm,_=ensure_v7style_model_for_theta(case,theta_idx,data_dict); rows=[]
    for seed in COMPARE_THETA06_SEEDS:
        ref=build_external_reference_for_theta(case,theta_idx,THETA_LIST,seed,COMPARE_THETA06_MC); h=ref['h_mc']; margin=0.05*(h.max()-h.min()+1e-9); z=np.linspace(h.min()-margin,h.max()+margin,600)
        m,_,_,_,_=predict_cdf_v7style_on_external_reference(case,theta_idx,ref,z,save_band_plot=True,output_prefix='t06',net=net,norm=norm)
        rows.append(m); pd.DataFrame([m]).to_csv(f"{V7STYLE_RESULT_DIR}/theta_independent_theta06_v7style_external_seed{seed:03d}_mc{COMPARE_THETA06_MC}_metrics.csv",index=False)
    by=pd.DataFrame(rows); Path(COMPARISON_RESULT_DIR).mkdir(parents=True,exist_ok=True); by.to_csv(f"{COMPARISON_RESULT_DIR}/theta06_v7style_multiseed_external_by_seed.csv",index=False)
    arr=by['external_cdf_arms'].astype(float).values; q50=np.abs(by['q50_error'].astype(float).values)
    summ={'theta_idx':6,'n_seeds':len(rows),'mc_eval':COMPARE_THETA06_MC,'mean_arms':float(np.mean(arr)),'median_arms':float(np.median(arr)),'max_arms':float(np.max(arr)),'min_arms':float(np.min(arr)),'q90_arms':float(np.quantile(arr,0.9)),'std_arms':float(np.std(arr)),'mean_q50_abs_error':float(np.mean(q50)),'max_q50_abs_error':float(np.max(q50)),'mean_band_coverage_grid':float(by['band_coverage_grid'].mean()),'mean_band_width':float(by['mean_band_width'].mean()),'mean_opf_per_sample_ms':float(by['opf_per_sample_ms'].mean()),'mean_nn_forward_ms':float(by['nn_forward_mean_ms'].mean())}
    pd.DataFrame([summ]).to_csv(f"{COMPARISON_RESULT_DIR}/theta06_v7style_multiseed_external_summary.csv",index=False)
    return by

def write_global_comparison_summary():
    Path(COMPARISON_RESULT_DIR).mkdir(parents=True,exist_ok=True); rows=[]
    p1=Path(f"{COMPARISON_RESULT_DIR}/compare_theta02_v7_vs_v8_seed{COMPARE_THETA02_SEED:03d}_mc{COMPARE_THETA02_MC}_metrics.csv")
    if p1.exists():
        d=pd.read_csv(p1)
        for _,r in d.iterrows(): rows.append({'experiment_name':'theta02_compare','method':r.get('method'),'theta_idx':r.get('theta_idx'),'seed':r.get('seed'),'mc_eval':r.get('mc_eval'),'external_cdf_arms':r.get('external_cdf_arms'),'posterior_mean_cdf_arms':r.get('posterior_mean_cdf_arms',np.nan),'posterior_median_cdf_arms':r.get('posterior_median_cdf_arms',np.nan),'q05_error':r.get('q05_error',np.nan),'q50_error':r.get('q50_error',np.nan),'q95_error':r.get('q95_error',np.nan),'band_coverage_grid':r.get('band_coverage_grid',np.nan),'mean_band_width':r.get('mean_band_width',np.nan),'opf_per_sample_ms':r.get('opf_per_sample_ms',np.nan),'nn_forward_mean_ms':r.get('nn_forward_mean_ms',np.nan),'total_train_time_sec':np.nan,'model_path':r.get('model_path',''),'status':r.get('status','ok')})
    p2=Path(f"{COMPARISON_RESULT_DIR}/theta06_v7style_multiseed_external_by_seed.csv")
    if p2.exists():
        d=pd.read_csv(p2)
        for _,r in d.iterrows(): rows.append({'experiment_name':'theta06_multiseed','method':'v7style','theta_idx':6,'seed':r.get('seed',np.nan),'mc_eval':COMPARE_THETA06_MC,'external_cdf_arms':r.get('external_cdf_arms'),'posterior_mean_cdf_arms':r.get('posterior_mean_cdf_arms'),'posterior_median_cdf_arms':r.get('posterior_median_cdf_arms'),'q05_error':r.get('q05_error'),'q50_error':r.get('q50_error'),'q95_error':r.get('q95_error'),'band_coverage_grid':r.get('band_coverage_grid'),'mean_band_width':r.get('mean_band_width'),'opf_per_sample_ms':r.get('opf_per_sample_ms'),'nn_forward_mean_ms':r.get('nn_forward_mean_ms'),'total_train_time_sec':np.nan,'model_path':'','status':r.get('status','ok')})
    pd.DataFrame(rows).to_csv(f"{COMPARISON_RESULT_DIR}/comparison_summary.csv",index=False)

def write_comparison_plots():
    Path(COMPARISON_RESULT_DIR).mkdir(parents=True,exist_ok=True)
    p=Path(f"{COMPARISON_RESULT_DIR}/compare_theta02_v7_vs_v8_seed{COMPARE_THETA02_SEED:03d}_mc{COMPARE_THETA02_MC}_metrics.csv")
    if p.exists():
        d=pd.read_csv(p); dd=d[d['status']!='missing_model'];
        if len(dd)>0:
            plt.figure(figsize=(5,4)); plt.bar(dd['method'],dd['external_cdf_arms']); plt.ylabel('ARMS'); plt.tight_layout(); plt.savefig(f"{COMPARISON_RESULT_DIR}/compare_theta02_v7_vs_v8_arms_bar.png",dpi=200); plt.close()
            if 'nn_forward_mean_ms' in dd.columns:
                plt.figure(figsize=(5,4)); plt.bar(dd['method'],dd['nn_forward_mean_ms']); plt.ylabel('nn_forward_mean_ms'); plt.tight_layout(); plt.savefig(f"{COMPARISON_RESULT_DIR}/compare_prediction_time_ms.png",dpi=200); plt.close()
    p2=Path(f"{COMPARISON_RESULT_DIR}/theta06_v7style_multiseed_external_by_seed.csv")
    if p2.exists():
        d=pd.read_csv(p2); plt.figure(figsize=(6,4)); plt.plot(d['seed'],d['external_cdf_arms'],'o-'); plt.xlabel('seed'); plt.ylabel('ARMS'); plt.tight_layout(); plt.savefig(f"{COMPARISON_RESULT_DIR}/theta06_v7style_multiseed_arms_by_seed.png",dpi=200); plt.close(); plt.figure(figsize=(5,4)); plt.boxplot(d['external_cdf_arms']); plt.ylabel('ARMS'); plt.tight_layout(); plt.savefig(f"{COMPARISON_RESULT_DIR}/theta06_v7style_multiseed_arms_box.png",dpi=200); plt.close()

def main_theta_independent_v5_1_comparison():
    print('[v5_1-comparison] start',flush=True)
    case=build_ieee33_case(); data_dict=get_or_build_flex_dataset_cache(case); Path(COMPARISON_RESULT_DIR).mkdir(parents=True,exist_ok=True)
    if RUN_COMPARE_THETA02_V7_VS_V8: run_compare_theta02_v7_vs_v8(case,data_dict)
    if RUN_COMPARE_THETA06_MULTI_SCENARIO: run_theta06_v7style_multiseed_external(case,data_dict)
    write_global_comparison_summary(); write_comparison_plots(); print('[v5_1-comparison] done',flush=True)



# ===== v6 all-theta train/eval/synthesis =====
ALL_THETA_MODE = True
ALL_THETA_LIST = list(range(N_THETA))
ALL_THETA_TRAIN = False
ALL_THETA_EVAL = False
ALL_THETA_FLEX_SYNTHESIS = False
ALL_THETA_RESULT_DIR = "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_qcal_only_alltheta"
EXISTING_MC2500_DIR = "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_qcal_only_comparison_mc2500"
ALL_THETA_COMPARISON_DIR = "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_qcal_only_comparison_mc2500_seeds04_10"
COMBINED_MC2500_DIR = "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_qcal_only_comparison_mc2500_seeds01_10_combined"
ADDITIONAL_EVAL_SEEDS = list(range(4,11))
ALL_THETA_FLEX_DIR = "training_results_theta_independent_v7style_no_pcc_branch_limit_flex_synthesis"
TRAIN_V7_MODEL_IF_MISSING = True
SKIP_TRAIN_IF_MODEL_EXISTS = True
ALL_THETA_EVAL_SEEDS = ADDITIONAL_EVAL_SEEDS
ALL_THETA_EVAL_MC = 2500
FLEX_SYNTHESIS_SEEDS = [1]
FLEX_SYNTHESIS_MC = 2500
FLEX_SYNTHESIS_QUANTILES = [0.05,0.50,0.95]

def _copy_theta_artifacts_to_alltheta(theta_idx):
    Path(ALL_THETA_RESULT_DIR).mkdir(parents=True, exist_ok=True)
    src=f"{TRAINING_RESULT_DIR}/theta_independent_theta{theta_idx:02d}_v7style"
    for ext in ['model.pt','norm.pkl','config.json','training_log.csv']:
        sp=Path(f"{src}_{ext}")
        if sp.exists():
            dp=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{theta_idx:02d}_v7style_{ext}")
            dp.write_bytes(sp.read_bytes())

def load_existing_all_theta_models(case):
    mp={}
    for j in ALL_THETA_LIST:
        m=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{j:02d}_v7style_model.pt")
        n=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{j:02d}_v7style_norm.pkl")
        if m.exists() and n.exists():
            norm=pickle.load(open(n,'rb')); net=BayesSingleThetaV7StyleGMM2SupportNet(len(norm['x_mu_mean'][0]),case,hidden=160,depth=3,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO).to(DEVICE); net.load_state_dict(torch.load(m,map_location=DEVICE)); net.eval(); mp[j]={'net':net,'norm':norm,'model_path':str(m),'status':'loaded'}
        else:
            mp[j]={'net':None,'norm':None,'model_path':str(m),'status':'missing_model'}
    return mp

def train_or_load_all_theta_v7style_models(case,data_dict):
    XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,*_=data_dict; out=[]; mp={}
    Path(ALL_THETA_RESULT_DIR).mkdir(parents=True, exist_ok=True)
    for j in ALL_THETA_LIST:
        m=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{j:02d}_v7style_model.pt"); n=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{j:02d}_v7style_norm.pkl"); c=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{j:02d}_v7style_config.json")
        if m.exists() and n.exists() and SKIP_TRAIN_IF_MODEL_EXISTS:
            print(f"[all-theta-load] theta_idx={j:02d} existing model found, skip training",flush=True)
            norm=pickle.load(open(n,'rb')); net=BayesSingleThetaV7StyleGMM2SupportNet(XMU.shape[1],case,hidden=160,depth=3,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO).to(DEVICE); net.load_state_dict(torch.load(m,map_location=DEVICE)); net.eval(); status='loaded'
        elif TRAIN_V7_MODEL_IF_MISSING:
            t0=time.perf_counter(); net,norm,_=train_single_theta_v7style_model(case,XMU,XREAL,YH,YP0,YQ0,YPG,YQG,j,THETA_LIST); status='trained'; _copy_theta_artifacts_to_alltheta(j); m=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{j:02d}_v7style_model.pt"); n=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{j:02d}_v7style_norm.pkl"); c=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{j:02d}_v7style_config.json")
        else:
            print(f"[all-theta-warning] theta_idx={j:02d} missing and training disabled",flush=True); mp[j]={'net':None,'norm':None,'model_path':str(m),'status':'missing_model'}; continue
        mp[j]={'net':net,'norm':norm,'model_path':str(m),'status':status}
        out.append({'theta_idx':j,'theta':float(THETA_LIST[j]),'alpha':float(np.cos(THETA_LIST[j])),'beta':float(np.sin(THETA_LIST[j])),'status':status,'model_path':str(m),'norm_path':str(n),'config_path':str(c),'total_train_time_sec':np.nan,'mean_epoch_time_sec':np.nan,'final_total_loss':np.nan,'final_nll_loss':np.nan,'final_phys_loss':np.nan,'final_kl_loss':np.nan,'final_weighted_kl_loss':np.nan,'final_dispatch_loss':np.nan,'final_t_loss':np.nan})
        pd.DataFrame([out[-1]]).to_csv(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{j:02d}_v7style_train_summary.csv",index=False)
    pd.DataFrame(out).to_csv(f"{ALL_THETA_RESULT_DIR}/all_theta_v7style_train_summary.csv",index=False)
    return mp

def draw_external_scenario_by_seed(case, seed):
    rng=np.random.default_rng(seed); return sample_scenario_means(case,rng)

def build_external_reference_for_theta_given_scenario(case,theta_idx,theta_list,pd_mu,qd_mu,pr_mu,qr_mu,seed,mc_eval):
    np.random.seed(seed+100000+theta_idx); theta=float(theta_list[theta_idx]); alpha=float(np.cos(theta)); beta=float(np.sin(theta)); h=[]; p0=[]; q0=[]; infeas=0; t0=time.perf_counter(); std_pd=0.10*np.maximum(pd_mu,1e-3); std_pr=0.12*np.maximum(pr_mu,1e-3)
    for sample_idx in range(mc_eval):
        pdv=pd_mu.copy(); prv=pr_mu.copy()
        for i in range(case.n_bus):
            if pd_mu[i]>1e-9: pdv[i]=sample_trunc_normal(pd_mu[i],std_pd[i],0.0,None)
        for k,b in enumerate(case.pv_buses):
            prv[b]=sample_trunc_normal(pr_mu[b],std_pr[b],0.0,float(case.pv_pmax[k]))
        qdv=qd_mu*(pdv/np.maximum(pd_mu,1e-6)); qrv=prv*math.tan(math.acos(case.pv_pf))
        sol=solve_flex_support_gurobi_33bus(case,pdv,qdv,prv,qrv,alpha,beta,return_detail=True,stabilize_dispatch=STABILIZE_OPF_DISPATCH)
        if sol.get('ok',False): h.append(float(sol['h'])); p0.append(float(sol['P0'])); q0.append(float(sol['Q0']))
        else: infeas+=1
        if mc_eval >= 2500 and ((sample_idx+1) % 500 == 0 or (sample_idx+1) == mc_eval):
            print(f"[mc2500-opf] theta={theta_idx:02d} seed={seed:03d} sample={sample_idx+1}/{mc_eval}", flush=True)
    if len(h)==0: raise RuntimeError(f'all infeasible theta={theta_idx} seed={seed}')
    opf=time.perf_counter()-t0
    return {'theta_idx':theta_idx,'theta':theta,'alpha':alpha,'beta':beta,'seed':seed,'mc_eval':mc_eval,'x_mu':make_feature_vector(case,pd_mu,pr_mu),'pd_mu':pd_mu,'qd_mu':qd_mu,'pr_mu':pr_mu,'qr_mu':qr_mu,'h_mc':np.array(h),'p0_mc':np.array(p0),'q0_mc':np.array(q0),'n_success':len(h),'n_infeasible':infeas,'opf_mc_total_time_sec':opf,'opf_per_sample_ms':opf*1000/max(1,mc_eval)}

def eval_all_theta_multiseed_v7style(case,data_dict,model_map):
    rows=[]; Path(ALL_THETA_COMPARISON_DIR).mkdir(parents=True,exist_ok=True)
    total_jobs=len(ALL_THETA_EVAL_SEEDS)*len(ALL_THETA_LIST); done_jobs=0
    for seed in ALL_THETA_EVAL_SEEDS:
        pd_mu,qd_mu,pr_mu,qr_mu=draw_external_scenario_by_seed(case,seed)
        for j in ALL_THETA_LIST:
            if j not in model_map or model_map[j]['net'] is None: continue
            print(f"[more-seeds-eval] seed={seed:03d} theta={j:02d} start", flush=True)
            ref=build_external_reference_for_theta_given_scenario(case,j,THETA_LIST,pd_mu,qd_mu,pr_mu,qr_mu,seed,ALL_THETA_EVAL_MC)
            h=ref['h_mc']; margin=0.05*(h.max()-h.min()+1e-9); z=np.linspace(h.min()-margin,h.max()+margin,600)
            m,cdf,p05,p95,cdf_mc=predict_cdf_v7style_on_external_reference(case,j,ref,z,save_band_plot=False,output_prefix='all',net=model_map[j]['net'],norm=model_map[j]['norm'])
            fig=f"{ALL_THETA_COMPARISON_DIR}/CDF_theta{j:02d}_seed{seed:03d}_mc{ALL_THETA_EVAL_MC}_v7style_external_bandfix.png"; plt.figure(figsize=(6,4),dpi=170); plt.plot(z,cdf_mc,'k-',lw=2); plt.plot(z,cdf,'r--',lw=2); plt.fill_between(z,p05,p95,color='#93c5fd',alpha=0.4); plt.tight_layout(); plt.savefig(fig,dpi=200); plt.close()
            m.update({'theta':float(THETA_LIST[j]),'h_mc_mean':float(np.mean(h)),'h_mc_std':float(np.std(h)),'n_success':ref['n_success'],'n_infeasible':ref['n_infeasible'],'opf_per_sample_ms':ref['opf_per_sample_ms'],'model_path':model_map[j]['model_path'],'status':'ok'})
            pd.DataFrame([m]).to_csv(f"{ALL_THETA_COMPARISON_DIR}/metrics_theta{j:02d}_seed{seed:03d}_mc{ALL_THETA_EVAL_MC}_v7style_external.csv",index=False)
            rows.append(m)
            done_jobs += 1
            print(f"[more-seeds-eval] seed={seed:03d} theta={j:02d} done, ARMS={m['external_cdf_arms']:.4f}, opf_per_sample_ms={m['opf_per_sample_ms']:.3f}", flush=True)
            print(f"[more-seeds-eval] progress {done_jobs}/{total_jobs}", flush=True)
    by=pd.DataFrame(rows); by.to_csv(f"{ALL_THETA_COMPARISON_DIR}/all_theta_multiseed_external_by_theta_seed.csv",index=False)
    s1=by.groupby('theta_idx').agg(theta=('theta','first'),n_seeds=('seed','count'),mean_arms=('external_cdf_arms','mean'),median_arms=('external_cdf_arms','median'),max_arms=('external_cdf_arms','max'),min_arms=('external_cdf_arms','min'),q90_arms=('external_cdf_arms',lambda x: np.quantile(x,0.9)),std_arms=('external_cdf_arms','std'),mean_q50_abs_error=('q50_error',lambda x: np.mean(np.abs(x))),max_q50_abs_error=('q50_error',lambda x: np.max(np.abs(x))),mean_q95_abs_error=('q95_error',lambda x: np.mean(np.abs(x))),mean_band_coverage_grid=('band_coverage_grid','mean'),mean_band_width=('mean_band_width','mean'),mean_opf_per_sample_ms=('opf_per_sample_ms','mean'),mean_nn_forward_ms=('nn_forward_mean_ms','mean')).reset_index(); s1['status']='ok'; s1.to_csv(f"{ALL_THETA_COMPARISON_DIR}/all_theta_external_summary_by_theta.csv",index=False)
    s2=by.groupby('seed').agg(n_theta=('theta_idx','count'),mean_arms=('external_cdf_arms','mean'),median_arms=('external_cdf_arms','median'),max_arms=('external_cdf_arms','max'),min_arms=('external_cdf_arms','min'),q90_arms=('external_cdf_arms',lambda x: np.quantile(x,0.9)),std_arms=('external_cdf_arms','std')).reset_index()
    worst=[]
    for sd in s2['seed']:
        b=by[by['seed']==sd]; i=b['external_cdf_arms'].idxmax(); worst.append((int(by.loc[i,'theta_idx']),float(by.loc[i,'external_cdf_arms'])))
    s2['worst_theta_idx']=[w[0] for w in worst]; s2['worst_theta_arms']=[w[1] for w in worst]; s2.to_csv(f"{ALL_THETA_COMPARISON_DIR}/all_theta_external_summary_by_seed.csv",index=False)

def plot_all_theta_evaluation_results():
    p=Path(f"{ALL_THETA_COMPARISON_DIR}/all_theta_multiseed_external_by_theta_seed.csv"); p1=Path(f"{ALL_THETA_COMPARISON_DIR}/all_theta_external_summary_by_theta.csv"); p2=Path(f"{ALL_THETA_COMPARISON_DIR}/all_theta_external_summary_by_seed.csv")
    if not p.exists() or not p1.exists() or not p2.exists(): print('[v6-warning] missing csv for plots',flush=True); return
    by=pd.read_csv(p); s1=pd.read_csv(p1)
    plt.figure(figsize=(7,4)); plt.bar(s1['theta_idx'],s1['mean_arms']); plt.title('v7style all theta external evaluation mean arms'); plt.tight_layout(); plt.savefig(f"{ALL_THETA_COMPARISON_DIR}/all_theta_mean_arms_bar.png",dpi=200); plt.close()
    plt.figure(figsize=(7,4)); plt.bar(s1['theta_idx'],s1['max_arms']); plt.title('v7style all theta external evaluation max arms'); plt.tight_layout(); plt.savefig(f"{ALL_THETA_COMPARISON_DIR}/all_theta_max_arms_bar.png",dpi=200); plt.close()
    piv=by.pivot_table(index='theta_idx',columns='seed',values='external_cdf_arms'); plt.figure(figsize=(7,5)); plt.imshow(piv.values,aspect='auto',cmap='viridis'); plt.colorbar(label='ARMS'); plt.title('v7style all theta external evaluation arms heatmap'); plt.yticks(range(len(piv.index)),piv.index); plt.xticks(range(len(piv.columns)),piv.columns); plt.tight_layout(); plt.savefig(f"{ALL_THETA_COMPARISON_DIR}/all_theta_seed_arms_heatmap.png",dpi=200); plt.close()
    plt.figure(figsize=(8,4)); data=[by[by['theta_idx']==t]['external_cdf_arms'].values for t in sorted(by['theta_idx'].unique())]; plt.boxplot(data,labels=[int(t) for t in sorted(by['theta_idx'].unique())]); plt.title('v7style all theta external evaluation arms boxplot by theta'); plt.tight_layout(); plt.savefig(f"{ALL_THETA_COMPARISON_DIR}/all_theta_arms_boxplot_by_theta.png",dpi=200); plt.close()
    plt.figure(figsize=(7,4)); plt.bar(s1['theta_idx'],s1['mean_q50_abs_error']); plt.title('v7style all theta external evaluation mean q50 abs error'); plt.tight_layout(); plt.savefig(f"{ALL_THETA_COMPARISON_DIR}/all_theta_mean_q50_abs_error_bar.png",dpi=200); plt.close()
    plt.figure(figsize=(7,4)); plt.bar(s1['theta_idx'],s1['mean_nn_forward_ms']); plt.title('v7style all theta external evaluation nn forward time'); plt.tight_layout(); plt.savefig(f"{ALL_THETA_COMPARISON_DIR}/all_theta_nn_forward_time_bar.png",dpi=200); plt.close()
    plt.figure(figsize=(7,4)); plt.bar(s1['theta_idx'],s1['mean_opf_per_sample_ms']); plt.title('v7style all theta external evaluation opf time'); plt.tight_layout(); plt.savefig(f"{ALL_THETA_COMPARISON_DIR}/all_theta_opf_time_bar.png",dpi=200); plt.close()

def plot_flex_synthesis_for_seed(seed,mc_eval,pred_polys,mc_polys,support_df,area_summary):
    Path(ALL_THETA_FLEX_DIR).mkdir(parents=True,exist_ok=True)
    def cl(p): return np.r_[p[:,0],p[0,0]],np.r_[p[:,1],p[0,1]]
    plt.figure(figsize=(7,6));
    for q,c in [(0.05,'red'),(0.50,'blue'),(0.95,'orange')]:
        if q in mc_polys and mc_polys[q] is not None: x,y=cl(mc_polys[q]); plt.plot(x,y,color='black',lw=1.5,alpha=0.6,label=f'MC q{int(q*100):02d}')
        if q in pred_polys and pred_polys[q] is not None: x,y=cl(pred_polys[q]); plt.plot(x,y,color=c,lw=2,label=f'Pred q{int(q*100):02d}')
    plt.xlabel('P0'); plt.ylabel('Q0'); plt.legend(); plt.title(f'seed={seed} mc={mc_eval} area errs={area_summary.get("area_error_pct_q50",np.nan):.2f}%'); plt.axis('equal'); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(f"{ALL_THETA_FLEX_DIR}/flex_domain_seed{seed:03d}_mc{mc_eval}_q05_q50_q95_overlay.png",dpi=220); plt.close()
    plt.figure(figsize=(7,6));
    if 0.5 in mc_polys and mc_polys[0.5] is not None: x,y=cl(mc_polys[0.5]); plt.plot(x,y,'k-',lw=2,label='MC q50')
    if 0.5 in pred_polys and pred_polys[0.5] is not None: x,y=cl(pred_polys[0.5]); plt.plot(x,y,'b--',lw=2,label='Pred q50')
    plt.xlabel('P0'); plt.ylabel('Q0'); plt.legend(); plt.axis('equal'); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(f"{ALL_THETA_FLEX_DIR}/flex_domain_seed{seed:03d}_mc{mc_eval}_q50_compare.png",dpi=220); plt.close()
    plt.figure(figsize=(7,4)); plt.plot(support_df['theta_idx'],np.abs(support_df['err_q05']),label='q05'); plt.plot(support_df['theta_idx'],np.abs(support_df['err_q50']),label='q50'); plt.plot(support_df['theta_idx'],np.abs(support_df['err_q95']),label='q95'); plt.legend(); plt.tight_layout(); plt.savefig(f"{ALL_THETA_FLEX_DIR}/flex_synthesis_seed{seed:03d}_mc{mc_eval}_support_error_by_theta.png",dpi=220); plt.close()
    plt.figure(figsize=(5,4)); vals=[area_summary['area_error_pct_q05'],area_summary['area_error_pct_q50'],area_summary['area_error_pct_q95']]; plt.bar(['q05','q50','q95'],vals); plt.tight_layout(); plt.savefig(f"{ALL_THETA_FLEX_DIR}/flex_synthesis_seed{seed:03d}_mc{mc_eval}_area_error_bar.png",dpi=220); plt.close()

def synthesize_flex_domain_for_seed(case,model_map,seed,mc_eval):
    if set(ALL_THETA_LIST)!=set(range(N_THETA)):
        print('flex synthesis requires all theta directions; skip synthesis',flush=True); return
    pd_mu,qd_mu,pr_mu,qr_mu=draw_external_scenario_by_seed(case,seed)
    hpred={q:[] for q in FLEX_SYNTHESIS_QUANTILES}; hmc={q:[] for q in FLEX_SYNTHESIS_QUANTILES}; rows=[]
    for j in ALL_THETA_LIST:
        if j not in model_map or model_map[j]['net'] is None: print('[v6-warning] missing model for synthesis',flush=True); return
        ref=build_external_reference_for_theta_given_scenario(case,j,THETA_LIST,pd_mu,qd_mu,pr_mu,qr_mu,seed,mc_eval)
        x=(ref['x_mu'].reshape(1,-1)-model_map[j]['norm']['x_mu_mean'])/model_map[j]['norm']['x_mu_std']; xt=torch.tensor(x,dtype=torch.float32,device=DEVICE)
        with torch.no_grad(): _,w,mu,sigma=model_map[j]['net'].forward_gmm_single(xt,sample=False)
        w=w.cpu().numpy().reshape(-1); mu=mu.cpu().numpy().reshape(-1); sigma=sigma.cpu().numpy().reshape(-1); hm=float(model_map[j]['norm']['h_mean'][0,0]); hs=float(model_map[j]['norm']['h_std'][0,0])
        rr={'theta_idx':j,'theta':float(THETA_LIST[j]),'alpha':float(np.cos(THETA_LIST[j])),'beta':float(np.sin(THETA_LIST[j]))}
        for q in FLEX_SYNTHESIS_QUANTILES:
            qp=hm+hs*float(np.asarray(gmm_quantile(q,w,mu,sigma)).reshape(-1)[0]); qm=float(np.quantile(ref['h_mc'],q)); hpred[q].append(qp); hmc[q].append(qm); rr[f'h_pred_q{int(q*100):02d}']=qp; rr[f'h_mc_q{int(q*100):02d}']=qm; rr[f'err_q{int(q*100):02d}']=qp-qm
        rows.append(rr)
    support_df=pd.DataFrame(rows); support_df.to_csv(f"{ALL_THETA_FLEX_DIR}/flex_synthesis_seed{seed:03d}_mc{mc_eval}_support_values.csv",index=False)
    pred_polys={}; mc_polys={}; area={}
    for q in FLEX_SYNTHESIS_QUANTILES:
        pp=support_values_to_polygon(THETA_LIST,np.array(hpred[q])); mp=support_values_to_polygon(THETA_LIST,np.array(hmc[q])); pred_polys[q]=pp; mc_polys[q]=mp
        pa=polygon_area(pp) if pp is not None else np.nan; ma=polygon_area(mp) if mp is not None else np.nan; area[f'pred_area_q{int(q*100):02d}']=pa; area[f'mc_area_q{int(q*100):02d}']=ma; area[f'area_error_pct_q{int(q*100):02d}']=100*(pa-ma)/max(1e-9,abs(ma)) if np.isfinite(pa) and np.isfinite(ma) else np.nan; area[f'polygon_status_q{int(q*100):02d}']='ok' if (pp is not None and mp is not None) else 'failed'
    area.update({'seed':seed,'mc_eval':mc_eval}); pd.DataFrame([area]).to_csv(f"{ALL_THETA_FLEX_DIR}/flex_synthesis_seed{seed:03d}_mc{mc_eval}_area_summary.csv",index=False)
    plot_flex_synthesis_for_seed(seed,mc_eval,pred_polys,mc_polys,support_df,area)

def main_v6_all_theta_train_eval_flex_synthesis():
    print('[v6-alltheta] start',flush=True)
    case=build_ieee33_case(); data_dict=get_or_build_flex_dataset_cache(case)
    Path(ALL_THETA_RESULT_DIR).mkdir(parents=True,exist_ok=True); Path(ALL_THETA_COMPARISON_DIR).mkdir(parents=True,exist_ok=True); Path(ALL_THETA_FLEX_DIR).mkdir(parents=True,exist_ok=True)
    model_map=train_or_load_all_theta_v7style_models(case,data_dict) if ALL_THETA_TRAIN else load_existing_all_theta_models(case)
    if ALL_THETA_EVAL:
        eval_all_theta_multiseed_v7style(case,data_dict,model_map); plot_all_theta_evaluation_results()
    if ALL_THETA_FLEX_SYNTHESIS:
        for sd in FLEX_SYNTHESIS_SEEDS: synthesize_flex_domain_for_seed(case,model_map,sd,FLEX_SYNTHESIS_MC)
    print('[v6-alltheta] done',flush=True)




def apply_v7_1_runtime_config():
    global NUM_SCENARIOS, MC_PER_SCENARIO, N_THETA, EPOCHS, THETA_LIST
    global DATASET_CACHE_MODE, ALL_THETA_LIST, ALL_THETA_EVAL_SEEDS, ALL_THETA_EVAL_MC
    global RUN_FLEX_SYNTHESIS, EVAL_POSTERIOR_SAMPLES, PREDICT_TIMING_REPEAT
    if SMOKE_TEST:
        NUM_SCENARIOS=2; MC_PER_SCENARIO=3; N_THETA=12; EPOCHS=2
        THETA_LIST=np.linspace(0,2*np.pi,N_THETA,endpoint=False); ALL_THETA_LIST=[3]
        ALL_THETA_EVAL_SEEDS=[1]; ALL_THETA_EVAL_MC=20; RUN_FLEX_SYNTHESIS=False
        EVAL_POSTERIOR_SAMPLES=10; PREDICT_TIMING_REPEAT=20; DATASET_CACHE_MODE='rebuild'
    else:
        NUM_SCENARIOS=1000; MC_PER_SCENARIO=60; N_THETA=12; EPOCHS=340
        THETA_LIST=np.linspace(0,2*np.pi,N_THETA,endpoint=False); ALL_THETA_LIST=list(range(N_THETA))
        ALL_THETA_EVAL_SEEDS=[1,2,3]; ALL_THETA_EVAL_MC=800; EVAL_POSTERIOR_SAMPLES=100; PREDICT_TIMING_REPEAT=1000
        if BUILD_NEW_DATASET: DATASET_CACHE_MODE='rebuild'

def assert_no_old_model_or_cache_paths():
    for v in [DATASET_CACHE_DIR,DATASET_CACHE_TAG,TRAINING_RESULT_DIR,ALL_THETA_RESULT_DIR,ALL_THETA_COMPARISON_DIR]:
        if 'no_pcc_branch_limit' not in str(v):
            raise RuntimeError(f'Path must include no_pcc_branch_limit: {v}')

def print_branch_limit_policy(case):
    print('[branch-limit-policy]',flush=True)
    print(f'line_limit_mode = {case.line_limit_mode}',flush=True)
    print(f'pcc_branch_indices = {case.pcc_branch_indices.tolist()}',flush=True)
    for l in case.pcc_branch_indices: print(f'pcc branch l{l}: {int(case.from_bus[l])}->{int(case.to_bus[l])}',flush=True)
    print(f'use_explicit_pcc_limits = {case.use_explicit_pcc_limits}',flush=True)
    print(f'PCC_P=[{case.pcc_p_min},{case.pcc_p_max}] PCC_Q=[{case.pcc_q_min},{case.pcc_q_max}]',flush=True)
    print(f'internal fmax_p range=({float(np.min(case.fmax_p[case.internal_branch_mask]))},{float(np.max(case.fmax_p[case.internal_branch_mask]))})',flush=True)

def sanity_check_theta3_no_pcc_cap(case):
    seed=1; mc=50; theta_idx=3; pd_mu,qd_mu,pr_mu,qr_mu=draw_external_scenario_by_seed(case,seed)
    ref=build_external_reference_for_theta_given_scenario(case,theta_idx,THETA_LIST,pd_mu,qd_mu,pr_mu,qr_mu,seed,mc)
    h=np.sort(ref['h_mc']); tol=1e-4; hb=np.round(h/tol)*tol; mass=float(pd.Series(hb).value_counts().max()/max(1,len(h)))
    df=pd.DataFrame({'h':h}); df.to_csv('branch_limit_sanity_theta03_seed001_mc50.csv',index=False)
    summ={'seed':seed,'mc':mc,'h_mean':float(np.mean(h)),'h_std':float(np.std(h)),'theta3_h_atom_mass_tol1e-4':mass}
    pd.DataFrame([summ]).to_csv('branch_limit_sanity_summary.csv',index=False)
    if mass>0.30: print('theta=3 still has strong atom-like mass; check other binding constraints',flush=True)

def compare_old_vs_new_theta3_distribution(case):
    old=Path('diagnostics_jump/mc_opf_diag_theta03_seed001_mc2500_clip_samples.csv')
    if not old.exists():
        print('[warning] old diagnostics missing, skip compare_old_vs_new',flush=True); return
    od=pd.read_csv(old); od=od[od['ok']==1]
    pd_mu,qd_mu,pr_mu,qr_mu=draw_external_scenario_by_seed(case,1)
    ref=build_external_reference_for_theta_given_scenario(case,3,THETA_LIST,pd_mu,qd_mu,pr_mu,qr_mu,1,2500)
    nh=np.array(ref['h_mc']); oh=od['h'].values
    plt.figure(figsize=(7,5));
    for arr,l in [(oh,'old'),(nh,'new')]:
        hs=np.sort(arr); c=np.arange(1,len(hs)+1)/len(hs); plt.plot(hs,c,label=l)
    plt.legend(); plt.tight_layout(); plt.savefig('old_vs_new_theta03_seed001_h_cdf_compare.png',dpi=220); plt.close()
    tol=1e-4
    out={'old_max_bin_mass_tol1e-4':float(pd.Series(np.round(oh/tol)*tol).value_counts().max()/len(oh)),'new_max_bin_mass_tol1e-4':float(pd.Series(np.round(nh/tol)*tol).value_counts().max()/len(nh)),'old_max_window_mass_1e-3':np.nan,'new_max_window_mass_1e-3':np.nan,'old_h_mean':float(np.mean(oh)),'new_h_mean':float(np.mean(nh)),'old_h_std':float(np.std(oh)),'new_h_std':float(np.std(nh)),'old_top_active_signature':'unknown','new_top_active_signature':'unknown','old_Qline_pos_l00_active_rate':float((od.get('active_names','').astype(str).str.contains('Qline_pos_l0')).mean()) if 'active_names' in od else np.nan,'new_Qline_pos_l00_active_rate':np.nan}
    pd.DataFrame([out]).to_csv('old_vs_new_theta03_seed001_h_atom_compare.csv',index=False)


def _diag_quantiles(arr, prefix):
    arr=np.asarray(arr,dtype=float).reshape(-1)
    out={f'{prefix}_mean':float(np.mean(arr)),f'{prefix}_std':float(np.std(arr)),f'{prefix}_min':float(np.min(arr)),f'{prefix}_max':float(np.max(arr))}
    for q in DIAG_QUANTILES:
        out[f'{prefix}_q{int(round(q*100)):02d}']=float(np.quantile(arr,q))
    return out

def _scalar_norm_value(v):
    return float(np.asarray(v).reshape(-1)[0])

def _empirical_cdf(values,z_grid):
    hs=np.sort(np.asarray(values,dtype=float).reshape(-1))
    return np.searchsorted(hs,z_grid,side='right')/max(1,len(hs))

def build_external_reference_for_theta_given_scenario_with_active(case,theta_idx,theta_list,pd_mu,qd_mu,pr_mu,qr_mu,seed,mc_eval,progress_every=100):
    np.random.seed(seed+100000+theta_idx)
    theta=float(theta_list[theta_idx]); alpha=float(np.cos(theta)); beta=float(np.sin(theta))
    h=[]; p0=[]; q0=[]; xreal=[]; active=[]; infeas=0; t0=time.perf_counter()
    std_pd=0.10*np.maximum(pd_mu,1e-3); std_pr=0.12*np.maximum(pr_mu,1e-3)
    for m in range(mc_eval):
        pdv=pd_mu.copy(); prv=pr_mu.copy()
        for i in range(case.n_bus):
            if pd_mu[i]>1e-9:
                pdv[i]=sample_trunc_normal(pd_mu[i],std_pd[i],0.0,None)
        for k,b in enumerate(case.pv_buses):
            prv[b]=sample_trunc_normal(pr_mu[b],std_pr[b],0.0,float(case.pv_pmax[k]))
        qdv=qd_mu*(pdv/np.maximum(pd_mu,1e-6)); qrv=prv*math.tan(math.acos(case.pv_pf))
        sol=solve_flex_support_gurobi_33bus(case,pdv,qdv,prv,qrv,alpha,beta,return_detail=True,stabilize_dispatch=STABILIZE_OPF_DISPATCH)
        if sol.get('ok',False):
            h.append(float(sol['h'])); p0.append(float(sol['P0'])); q0.append(float(sol['Q0'])); xreal.append(make_feature_vector(case,pdv,prv))
            sig,act,names=get_active_constraint_signature(case,sol)
            active.append({'theta_idx':theta_idx,'theta':theta,'seed':seed,'mc_idx':m,'signature':sig,'active_names':act,'active_names_str':';'.join(act)})
        else:
            infeas+=1
        if progress_every and ((m+1)%progress_every==0 or (m+1)==mc_eval):
            elapsed=time.perf_counter()-t0; eta=elapsed/(m+1)*max(0,mc_eval-m-1)
            print(f"[targeted-mc-opf] theta_idx={theta_idx:02d} seed={seed:03d} {m+1}/{mc_eval} success={len(h)} infeasible={infeas} elapsed={elapsed:.1f}s eta={eta:.1f}s",flush=True)
    if len(h)==0:
        raise RuntimeError(f'all infeasible theta={theta_idx} seed={seed}')
    opf=time.perf_counter()-t0
    return {'theta_idx':theta_idx,'theta':theta,'alpha':alpha,'beta':beta,'seed':seed,'mc_eval':mc_eval,'x_mu':make_feature_vector(case,pd_mu,pr_mu),'pd_mu':pd_mu,'qd_mu':qd_mu,'pr_mu':pr_mu,'qr_mu':qr_mu,'h_mc':np.asarray(h),'p0_mc':np.asarray(p0),'q0_mc':np.asarray(q0),'xreal_mc':np.asarray(xreal),'active_records_external':active,'n_success':len(h),'n_infeasible':infeas,'opf_mc_total_time_sec':opf,'opf_per_sample_ms':opf*1000/max(1,mc_eval)}

def run_targeted_mc2500_case(case,data_dict,model_map,theta_idx,seed):
    Path(TARGETED_DIAG_DIR).mkdir(parents=True,exist_ok=True)
    pd_mu,qd_mu,pr_mu,qr_mu=draw_external_scenario_by_seed(case,seed)
    ref=build_external_reference_for_theta_given_scenario_with_active(case,theta_idx,THETA_LIST,pd_mu,qd_mu,pr_mu,qr_mu,seed,DIAG_MC_EVAL,DIAG_PROGRESS_EVERY)
    h=ref['h_mc']; margin=0.05*(h.max()-h.min()+1e-9); z=np.linspace(h.min()-margin,h.max()+margin,DIAG_CDF_GRID_N)
    if theta_idx not in model_map or model_map[theta_idx]['net'] is None:
        raise FileNotFoundError(f"missing data1000_mc60 v7style model for theta_idx={theta_idx}: {model_map.get(theta_idx,{})}")
    metrics,cdf_pred,p05,p95,cdf_mc=predict_cdf_v7style_on_external_reference(case,theta_idx,ref,z,save_band_plot=False,output_prefix=None,net=model_map[theta_idx]['net'],norm=model_map[theta_idx]['norm'])
    metrics.update({'h_mc_min':float(np.min(h)),'h_mc_max':float(np.max(h)),'h_mc_mean':float(np.mean(h)),'h_mc_std':float(np.std(h)),'n_success':ref['n_success'],'n_infeasible':ref['n_infeasible'],'opf_per_sample_ms':ref['opf_per_sample_ms'],'model_path':model_map[theta_idx]['model_path']})
    fig=f"{TARGETED_DIAG_DIR}/CDF_theta{theta_idx:02d}_seed{seed:03d}_mc{DIAG_MC_EVAL}_targeted.png"
    plt.figure(figsize=(7,5),dpi=180); plt.plot(z,cdf_mc,'k-',lw=2,label='External MC-OPF empirical CDF'); plt.plot(z,cdf_pred,'r--',lw=2,label='BPINN deterministic GMM CDF'); plt.fill_between(z,p05,p95,color='#93c5fd',alpha=0.42,label='posterior CDF 5-95%')
    plt.title(f"theta={theta_idx:02d} seed={seed:03d} MC={DIAG_MC_EVAL} ARMS={metrics['external_cdf_arms']:.3f} q50err={metrics['q50_error']:.4f}"); plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(fig,dpi=220); plt.close()
    metrics['plot_path']=fig
    pd.DataFrame([metrics]).to_csv(f"{TARGETED_DIAG_DIR}/targeted_metrics_theta{theta_idx:02d}_seed{seed:03d}_mc{DIAG_MC_EVAL}.csv",index=False)
    return metrics,ref

def compare_targeted_mc2500_with_existing_mc800(targeted_rows):
    src=Path(ALL_THETA_COMPARISON_DIR)/'all_theta_multiseed_external_by_theta_seed.csv'
    out=[]
    old=pd.read_csv(src) if src.exists() else pd.DataFrame()
    if old.empty:
        print(f"[targeted-warning] missing existing MC800 summary: {src}",flush=True)
    for r in targeted_rows:
        theta_idx=int(r['theta_idx']); seed=int(r['seed']); m=old[(old.get('theta_idx',pd.Series(dtype=int))==theta_idx)&(old.get('seed',pd.Series(dtype=int))==seed)] if not old.empty else pd.DataFrame()
        oldr=m.iloc[0] if len(m)>0 else pd.Series(dtype=float)
        arms800=float(oldr.get('external_cdf_arms',np.nan)); arms2500=float(r.get('external_cdf_arms',np.nan))
        if np.isfinite(arms800) and abs(arms2500-arms800)<=1.0:
            conclusion='MC800 result is stable; error is likely model/location bias.'
        elif np.isfinite(arms800) and arms2500 < arms800-2.0:
            conclusion='MC800 finite-sample noise contributed significantly.'
        elif np.isfinite(arms800):
            conclusion='Partial MC effect; model bias still likely.'
        else:
            conclusion='Existing MC800 result missing; cannot compare finite-sample effect.'
        out.append({'theta_idx':theta_idx,'seed':seed,'arms_mc800':arms800,'arms_mc2500':arms2500,'delta_arms_2500_minus_800':arms2500-arms800 if np.isfinite(arms800) else np.nan,'q50_error_mc800':oldr.get('q50_error',np.nan),'q50_error_mc2500':r.get('q50_error',np.nan),'delta_q50_error':r.get('q50_error',np.nan)-oldr.get('q50_error',np.nan) if np.isfinite(oldr.get('q50_error',np.nan)) else np.nan,'q05_error_mc800':oldr.get('q05_error',np.nan),'q05_error_mc2500':r.get('q05_error',np.nan),'q95_error_mc800':oldr.get('q95_error',np.nan),'q95_error_mc2500':r.get('q95_error',np.nan),'h_mc_std_mc800':oldr.get('h_mc_std',np.nan),'h_mc_std_mc2500':r.get('h_mc_std',np.nan),'conclusion':conclusion})
    df=pd.DataFrame(out); df.to_csv(f"{TARGETED_DIAG_DIR}/targeted_mc2500_vs_mc800_summary.csv",index=False); return df

def compute_nearest_training_scenarios(XMU,x_mu_ext,k_list=(20,50)):
    X=np.asarray(XMU,dtype=float); x=np.asarray(x_mu_ext,dtype=float).reshape(1,-1)
    xm=X.mean(axis=0,keepdims=True); xs=X.std(axis=0,keepdims=True)+1e-9
    dist=np.linalg.norm((X-xm)/xs-(x-xm)/xs,axis=1); order=np.argsort(dist)
    pd_sum=X[:,-2]; pv_sum=X[:,-1]; net=pd_sum-pv_sum; epd=float(x[0,-2]); epv=float(x[0,-1]); enet=epd-epv
    pct=lambda a,v: float(100.0*np.mean(a<=v))
    stats={'external_pd_sum':epd,'external_pv_sum':epv,'external_net_load':enet,'external_pd_sum_percentile':pct(pd_sum,epd),'external_pv_sum_percentile':pct(pv_sum,epv),'external_net_load_percentile':pct(net,enet)}
    return {int(k):{'nearest_idx':order[:int(k)],'nearest_dist':dist[order[:int(k)]]} for k in k_list},stats

def _prediction_quantiles_for_ref(model_entry,ref):
    norm=model_entry['norm']; net=model_entry['net']; x=(ref['x_mu'].reshape(1,-1)-norm['x_mu_mean'])/norm['x_mu_std']; xt=torch.tensor(x,dtype=torch.float32,device=DEVICE)
    with torch.no_grad(): _,w,mu,sigma=net.forward_gmm_single(xt,sample=False)
    w=w.cpu().numpy().reshape(-1); mu=mu.cpu().numpy().reshape(-1); sigma=sigma.cpu().numpy().reshape(-1); hm=_scalar_norm_value(norm['h_mean']); hs=_scalar_norm_value(norm['h_std'])
    return {f'pred_q{int(round(q*100)):02d}':float(hm+hs*float(np.asarray(gmm_quantile(q,w,mu,sigma)).reshape(-1)[0])) for q in DIAG_QUANTILES},w,mu,sigma,hm,hs

def _active_counter_rows(source,records,total,top_k=10):
    from collections import Counter
    labels=[]
    for r in records:
        an=r.get('active_names',r.get('active_names_str',''))
        if isinstance(an,list): labels.append(';'.join(an))
        else: labels.append(str(an))
    cnt=Counter(labels); rows=[]
    for rank,(names,count) in enumerate(cnt.most_common(top_k),1):
        rows.append({'source':source,'rank':rank,'count':int(count),'percentage':float(100*count/max(1,total)),'active_names':names})
    return rows

def summarize_active_patterns_for_indices(active_records,scenario_indices,theta_idx,mc_per_scenario,n_theta,top_k=10,external_records=None,output_path=None):
    train=[]; scenario_set=set(int(i) for i in scenario_indices)
    for r in active_records:
        if int(r.get('theta_idx',-1))==int(theta_idx) and int(r.get('scenario_idx',-999999)) in scenario_set:
            train.append(r)
    if not train:
        for si in scenario_indices:
            for mi in range(mc_per_scenario):
                idx=(int(si)*mc_per_scenario+mi)*n_theta+theta_idx
                if 0 <= idx < len(active_records): train.append(active_records[idx])
    rows=[]; rows.extend(_active_counter_rows('external_mc2500',external_records or [],len(external_records or []),top_k)); rows.extend(_active_counter_rows(f'nearest_train_k{len(scenario_indices)}',train,len(train),top_k))
    df=pd.DataFrame(rows)
    if output_path: df.to_csv(output_path,index=False)
    ext_top=df[df['source']=='external_mc2500']['active_names'].iloc[0] if len(df[df['source']=='external_mc2500']) else ''
    tr_top=df[df['source'].str.startswith('nearest_train')]['active_names'].iloc[0] if len(df[df['source'].str.startswith('nearest_train')]) else ''
    match=bool(ext_top==tr_top and ext_top!='')
    concl='Top active pattern is covered by nearest training scenarios.' if match else 'Active constraint pattern mismatch; critical-region coverage may be insufficient.'
    return df,match,concl

def summarize_nearest_training_coverage(case,data_dict,theta_idx,seed,ref,model_map,k_list=DIAG_K_LIST):
    XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active_records,all_names=data_dict
    near,stats=compute_nearest_training_scenarios(XMU,ref['x_mu'],k_list)
    predq,w,mu,sigma,hm,hs=_prediction_quantiles_for_ref(model_map[theta_idx],ref)
    rows=[]; extq=_diag_quantiles(ref['h_mc'],'ext')
    h=ref['h_mc']; margin=0.05*(h.max()-h.min()+1e-9); z=np.linspace(h.min()-margin,h.max()+margin,DIAG_CDF_GRID_N); cdf_ext=_empirical_cdf(h,z)
    cdf_pred=gmm_cdf((z-hm)/(hs+1e-9),w,mu,sigma)
    for K,obj in near.items():
        idx=obj['nearest_idx']; dist=obj['nearest_dist']; hnear=YH[idx,:,theta_idx].reshape(-1); nearq=_diag_quantiles(hnear,'near')
        detail=[]
        for rank,(si,dv) in enumerate(zip(idx,dist),1):
            ht=YH[si,:,theta_idx].reshape(-1)
            detail.append({'rank':rank,'scenario_idx':int(si),'distance':float(dv),'pd_sum':float(XMU[si,-2]),'pv_sum':float(XMU[si,-1]),'net_load':float(XMU[si,-2]-XMU[si,-1]),'h_train_mean':float(np.mean(ht)),'h_train_std':float(np.std(ht)),'h_train_q05':float(np.quantile(ht,0.05)),'h_train_q50':float(np.quantile(ht,0.5)),'h_train_q95':float(np.quantile(ht,0.95))})
        pd.DataFrame(detail).to_csv(f"{TARGETED_DIAG_DIR}/nearest_detail_theta{theta_idx:02d}_seed{seed:03d}_k{K}.csv",index=False)
        active_df,active_match,active_concl=summarize_active_patterns_for_indices(active_records,idx,theta_idx,MC_PER_SCENARIO,N_THETA,top_k=10,external_records=ref['active_records_external'],output_path=f"{TARGETED_DIAG_DIR}/active_pattern_compare_theta{theta_idx:02d}_seed{seed:03d}_k{K}.csv")
        near_minus_ext_q50=nearq['near_q50']-extq['ext_q50']; pred_minus_ext_q50=predq['pred_q50']-extq['ext_q50']
        if abs(near_minus_ext_q50)>0.03 and np.sign(near_minus_ext_q50)==np.sign(pred_minus_ext_q50): cov='Nearest training distribution is also shifted; local training coverage may be insufficient.'
        elif abs(near_minus_ext_q50)<=0.02 and abs(pred_minus_ext_q50)>0.03: cov='Nearest training distribution matches external MC; model/calibration bias is likely.'
        elif float(np.min(dist))>5.0 or stats['external_pv_sum_percentile']>95 or stats['external_net_load_percentile']<5: cov='External scenario is near edge of training distribution.'
        else: cov='Mixed evidence.'
        row={'theta_idx':theta_idx,'seed':seed,'K':K,'nearest_dist_min':float(np.min(dist)),'nearest_dist_mean':float(np.mean(dist)),'nearest_dist_max':float(np.max(dist)),**stats,**extq,**nearq,**predq,'pred_minus_ext_q50':pred_minus_ext_q50,'near_minus_ext_q50':near_minus_ext_q50,'pred_minus_near_q50':predq['pred_q50']-nearq['near_q50'],'pred_minus_ext_q05':predq['pred_q05']-extq['ext_q05'],'near_minus_ext_q05':nearq['near_q05']-extq['ext_q05'],'pred_minus_ext_q95':predq['pred_q95']-extq['ext_q95'],'near_minus_ext_q95':nearq['near_q95']-extq['ext_q95'],'active_top1_match':active_match,'active_conclusion':active_concl,'coverage_conclusion':cov}
        rows.append(row)
        cdf_near=_empirical_cdf(hnear,z); arms=float(np.sqrt(np.mean((cdf_pred-cdf_ext)**2))*100)
        plt.figure(figsize=(7,5),dpi=180); plt.plot(z,cdf_ext,'k-',lw=2,label='external MC2500 empirical'); plt.plot(z,cdf_near,color='green',lw=2,label=f'nearest train K={K} empirical'); plt.plot(z,cdf_pred,'r--',lw=2,label='BPINN deterministic')
        plt.title(f"theta={theta_idx:02d} seed={seed:03d} K={K} ARMS={arms:.3f} pred-ext q50={pred_minus_ext_q50:.4f} near-ext q50={near_minus_ext_q50:.4f}"); plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(f"{TARGETED_DIAG_DIR}/nearest_cdf_theta{theta_idx:02d}_seed{seed:03d}_k{K}.png",dpi=220); plt.close()
    return rows


def get_lambda_scen_q(epoch):
    if epoch < SCEN_Q_START_EPOCH_B:
        return LAM_SCEN_Q_A
    if epoch < SCEN_Q_START_EPOCH_C:
        return LAM_SCEN_Q_B
    if epoch < SCEN_Q_START_EPOCH_D:
        return LAM_SCEN_Q_C
    return LAM_SCEN_Q_D

def precompute_single_theta_scenario_distribution_stats(YH, theta_idx, taus):
    """
    YH shape = (N, M, T).  For one theta, compute empirical scenario-level
    h_theta quantiles, means, and standard deviations across the M MC realizations.
    """
    h=YH[:,:,theta_idx]
    q_emp=np.quantile(h,np.asarray(taus,dtype=float),axis=1).T
    mean_emp=h.mean(axis=1,keepdims=True)
    std_emp=h.std(axis=1,keepdims=True)
    return q_emp.astype(np.float32),mean_emp.astype(np.float32),std_emp.astype(np.float32)

def scenario_quantile_moment_calibration_loss(net,xmu_scen_raw,scenario_indices,q_emp,mean_emp,std_emp,x_mu_mean,x_mu_std,h_mean,h_std,taus,q_weights,sample=True):
    if not torch.is_tensor(scenario_indices):
        scenario_indices=torch.as_tensor(scenario_indices,device=DEVICE,dtype=torch.long)
    scenario_indices=torch.unique(scenario_indices.detach().long())
    if scenario_indices.numel()==0:
        z=torch.tensor(0.0,device=DEVICE)
        return z,z,z
    if scenario_indices.numel()>SCENARIO_CALIB_BATCH_MAX_SCENARIOS:
        perm=torch.randperm(scenario_indices.numel(),device=scenario_indices.device)[:SCENARIO_CALIB_BATCH_MAX_SCENARIOS]
        scenario_indices=scenario_indices[perm]
    ids_np=scenario_indices.detach().cpu().numpy().astype(int)
    x=((xmu_scen_raw[ids_np]-x_mu_mean)/x_mu_std).astype(np.float32)
    xt=torch.tensor(x,dtype=torch.float32,device=DEVICE)
    _,w,mu,sigma=net.forward_gmm_single(xt,sample=sample)
    q_pred=gmm_quantile_torch(w,mu,sigma,taus)
    q_emp_norm=(q_emp[ids_np]-float(np.asarray(h_mean).reshape(-1)[0]))/(float(np.asarray(h_std).reshape(-1)[0])+1e-9)
    mean_emp_norm=(mean_emp[ids_np]-float(np.asarray(h_mean).reshape(-1)[0]))/(float(np.asarray(h_std).reshape(-1)[0])+1e-9)
    std_emp_norm=std_emp[ids_np]/(float(np.asarray(h_std).reshape(-1)[0])+1e-9)
    qe=torch.tensor(q_emp_norm,dtype=torch.float32,device=DEVICE)
    me=torch.tensor(mean_emp_norm,dtype=torch.float32,device=DEVICE)
    se=torch.tensor(std_emp_norm,dtype=torch.float32,device=DEVICE)
    qw=torch.tensor(q_weights,dtype=torch.float32,device=DEVICE).view(1,-1)
    scen_q_loss=(qw*(q_pred-qe).pow(2)).mean()
    mix_mean=(w*mu).sum(dim=1,keepdim=True)
    mix_var=(w*(sigma.pow(2)+mu.pow(2))).sum(dim=1,keepdim=True)-mix_mean.pow(2)
    mix_std=torch.sqrt(torch.clamp(mix_var,min=1e-12))
    scen_mean_loss=(mix_mean-me).pow(2).mean()
    scen_std_loss=(mix_std-se).pow(2).mean()
    return scen_q_loss,scen_mean_loss,scen_std_loss

def _scenario_calib_eval_metrics(net,XMU,scenario_ids,q_emp,mean_emp,std_emp,norm):
    if len(scenario_ids)==0:
        return {'scen_q_rmse':np.nan,'scen_q50_rmse':np.nan,'scen_mean_rmse':np.nan,'scen_std_rmse':np.nan}
    ids=np.unique(np.asarray(scenario_ids,dtype=int))
    x=((XMU[ids]-norm['x_mu_mean'])/norm['x_mu_std']).astype(np.float32)
    xt=torch.tensor(x,dtype=torch.float32,device=DEVICE)
    with torch.no_grad(): _,w,mu,sigma=net.forward_gmm_single(xt,sample=False)
    q_pred=gmm_quantile_torch(w,mu,sigma,SCENARIO_Q_TAUS).detach().cpu().numpy()
    hm=float(np.asarray(norm['h_mean']).reshape(-1)[0]); hs=float(np.asarray(norm['h_std']).reshape(-1)[0])
    q_emp_norm=(q_emp[ids]-hm)/(hs+1e-9)
    mean_emp_norm=(mean_emp[ids]-hm)/(hs+1e-9); std_emp_norm=std_emp[ids]/(hs+1e-9)
    mix_mean=(w*mu).sum(dim=1,keepdim=True).detach().cpu().numpy()
    mix_var=(w*(sigma.pow(2)+mu.pow(2))).sum(dim=1,keepdim=True)-((w*mu).sum(dim=1,keepdim=True)).pow(2)
    mix_std=torch.sqrt(torch.clamp(mix_var,min=1e-12)).detach().cpu().numpy()
    qerr=q_pred-q_emp_norm
    q50_idx=list(SCENARIO_Q_TAUS).index(0.50)
    return {'scen_q_rmse':float(np.sqrt(np.mean(qerr**2))),'scen_q50_rmse':float(np.sqrt(np.mean(qerr[:,q50_idx]**2))),'scen_mean_rmse':float(np.sqrt(np.mean((mix_mean-mean_emp_norm)**2))),'scen_std_rmse':float(np.sqrt(np.mean((mix_std-std_emp_norm)**2)))}

def train_single_theta_v7style_model_with_qcal(case,data_dict,theta_idx):
    XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active_records,all_names=data_dict
    Path(TRAINING_RESULT_DIR).mkdir(parents=True,exist_ok=True); Path(ALL_THETA_RESULT_DIR).mkdir(parents=True,exist_ok=True)
    d=flatten_single_theta_dataset_v7style(XMU,XREAL,YH,YP0,YQ0,YPG,YQG,theta_idx,THETA_LIST)
    N,M=YH.shape[0],YH.shape[1]
    scenario_ids=np.repeat(np.arange(N),M)
    rng=np.random.default_rng(SEED_TRAIN+theta_idx); perm=rng.permutation(N); n_val=max(1,int(round(0.1*N))); va=perm[-n_val:]; tr=perm[:-n_val]
    trm=np.isin(scenario_ids,tr); vam=np.isin(scenario_ids,va)
    xmu_tr=d['xmu_flat'][trm]; yh_tr=d['yh_flat'][trm]; yt_tr=d['yt_flat'][trm]; ypg_tr=d['ypg_flat'][trm]; yqg_tr=d['yqg_flat'][trm]; sid_tr=scenario_ids[trm]
    xmu_va=d['xmu_flat'][vam]; yh_va=d['yh_flat'][vam]; sid_va=scenario_ids[vam]
    norm={'x_mu_mean':xmu_tr.mean(0,keepdims=True),'x_mu_std':xmu_tr.std(0,keepdims=True)+1e-9,'h_mean':np.array([[yh_tr.mean()]]),'h_std':np.array([[yh_tr.std()+1e-9]]),'t_mean':np.array([[yt_tr.mean()]]),'t_std':np.array([[yt_tr.std()+1e-9]]),'alpha':d['alpha'],'beta':d['beta'],'theta':d['theta'],'theta_idx':theta_idx,'scenario_quantile_calibration':True}
    q_emp,mean_emp,std_emp=precompute_single_theta_scenario_distribution_stats(YH,theta_idx,SCENARIO_Q_TAUS)
    to=lambda a: torch.tensor(a,dtype=torch.float32,device=DEVICE)
    xmu_n=to((xmu_tr-norm['x_mu_mean'])/norm['x_mu_std']); xmu_raw=to(xmu_tr); yh=to(yh_tr); yt=to(yt_tr); ypg=to(ypg_tr); yqg=to(yqg_tr); hm,hs,tm,ts=to(norm['h_mean']),to(norm['h_std']),to(norm['t_mean']),to(norm['t_std']); sid_t=torch.tensor(sid_tr,dtype=torch.long,device=DEVICE)
    net=BayesSingleThetaV7StyleGMM2SupportNet(XMU.shape[1],case,hidden=160,depth=3,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO).to(DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=LR)
    n=xmu_n.shape[0]; nb=(n+BATCH_SIZE-1)//BATCH_SIZE; rows=[]; best=np.inf; best_path=f"{TRAINING_RESULT_DIR}/theta_independent_theta{theta_idx:02d}_v7style_qcal_best_model.pt"
    print('[v7.4-qcal-train-start]',flush=True); print(f'theta_idx = {theta_idx}',flush=True); print('loss_old = nll + LAM_T_SUP*t_sup + LAM_DISPATCH_SUP*disp + LAM_PHYS_FLEX*phys + beta_kl*kl',flush=True)
    for ep in range(EPOCHS):
        p=np.random.default_rng(SEED_TRAIN+ep+theta_idx*1000).permutation(n); beta_kl=min(1.0,(ep+1)/max(1,KL_WARMUP_EPOCHS))*BETA_KL_MAX; lambda_q=get_lambda_scen_q(ep)
        st={k:0.0 for k in ['total','nll','t_sup','dispatch','phys','kl','scen_q','scen_mean','scen_std']}
        for b in range(nb):
            ii=p[b*BATCH_SIZE:min((b+1)*BATCH_SIZE,n)]
            hf,w,mu,sigma=net.forward_gmm_single(xmu_n[ii],sample=True); hnorm=(yh[ii]-hm)/(hs+1e-9); nll=(-gmm_log_prob(hnorm,w,mu,sigma)).mean(); t_hat,pg_hat,qg_hat=net.recover_dispatch_from_h(hf,yh[ii],hm,hs,tm,ts,sample=True)
            t_sup=(t_hat-yt[ii]).pow(2).mean(); disp=(pg_hat-ypg[ii]).pow(2).mean()+(qg_hat-yqg[ii]).pow(2).mean(); taus=make_v7style_physics_taus(True,xmu_n.device,xmu_n.dtype); hq_n=gmm_quantile_torch(w,mu,sigma,taus); hq=hm+hs*hq_n
            tq=[]; pgq=[]; qgq=[]
            for k in range(taus.numel()):
                tk,pgk,qgk=net.recover_dispatch_from_h(hf,hq[:,k:k+1],hm,hs,tm,ts,sample=True); tq.append(tk); pgq.append(pgk); qgq.append(qgk)
            tq=torch.cat(tq,dim=1); pgq=torch.stack(pgq,dim=1); qgq=torch.stack(qgq,dim=1)
            phys=physics_loss_flex_quantile_v7style(case,xmu_raw[ii],hq,tq,pgq,qgq,d['alpha'],d['beta']); kl=net.kl_divergence()/max(1,n)
            if USE_SCENARIO_QUANTILE_CALIBRATION or USE_SCENARIO_MOMENT_CALIBRATION:
                scen_q,scen_mean,scen_std=scenario_quantile_moment_calibration_loss(net,XMU,sid_t[ii],q_emp,mean_emp,std_emp,norm['x_mu_mean'],norm['x_mu_std'],norm['h_mean'],norm['h_std'],SCENARIO_Q_TAUS,SCENARIO_Q_WEIGHTS,sample=True)
            else:
                scen_q=scen_mean=scen_std=torch.tensor(0.0,device=DEVICE)
            loss_old=nll+LAM_T_SUP*t_sup+LAM_DISPATCH_SUP*disp+LAM_PHYS_FLEX*phys+beta_kl*kl
            loss=loss_old+lambda_q*scen_q+(LAM_SCEN_MEAN*scen_mean if USE_SCENARIO_MOMENT_CALIBRATION else 0.0)+(LAM_SCEN_STD*scen_std if USE_SCENARIO_MOMENT_CALIBRATION else 0.0)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
            vals=[loss,nll,t_sup,disp,phys,kl,scen_q,scen_mean,scen_std]
            for key,val in zip(st.keys(),vals): st[key]+=float(val.detach().cpu())
        for key in st: st[key]/=max(1,nb)
        train_cal=_scenario_calib_eval_metrics(net,XMU,tr,q_emp,mean_emp,std_emp,norm); val_cal=_scenario_calib_eval_metrics(net,XMU,va,q_emp,mean_emp,std_emp,norm)
        val_nll=np.nan
        if len(xmu_va)>0:
            with torch.no_grad(): _,wv,mv,sv=net.forward_gmm_single(to((xmu_va-norm['x_mu_mean'])/norm['x_mu_std']),sample=False); val_nll=float((-gmm_log_prob((to(yh_va)-hm)/(hs+1e-9),wv,mv,sv)).mean().cpu())
        row={'epoch':ep+1,'theta_idx':theta_idx,'total_loss':st['total'],'nll':st['nll'],'t_sup':st['t_sup'],'dispatch':st['dispatch'],'phys':st['phys'],'kl':st['kl'],'beta_kl':beta_kl,'lambda_q':lambda_q,'scen_q_loss':st['scen_q'],'scen_mean_loss':st['scen_mean'],'scen_std_loss':st['scen_std'],'train_scen_q_rmse':train_cal['scen_q_rmse'],'train_scen_q50_rmse':train_cal['scen_q50_rmse'],'train_scen_mean_rmse':train_cal['scen_mean_rmse'],'train_scen_std_rmse':train_cal['scen_std_rmse'],'val_nll':val_nll,'val_scen_q_rmse':val_cal['scen_q_rmse'],'val_scen_q50_rmse':val_cal['scen_q50_rmse'],'val_scen_mean_rmse':val_cal['scen_mean_rmse'],'val_scen_std_rmse':val_cal['scen_std_rmse']}
        rows.append(row)
        score=row['val_scen_q50_rmse'] if np.isfinite(row['val_scen_q50_rmse']) else row['total_loss']
        if score<best:
            best=score; torch.save(net.state_dict(),best_path)
        print(f"[v7.4-qcal] theta={theta_idx:02d} epoch {ep+1}/{EPOCHS} total={st['total']:.4f} nll={st['nll']:.4f} t={st['t_sup']:.4f} disp={st['dispatch']:.4f} phys={st['phys']:.4f} kl={st['kl']:.4f} scen_q={st['scen_q']:.4f} scen_mean={st['scen_mean']:.4f} scen_std={st['scen_std']:.4f} beta_kl={beta_kl:.4f} lambda_q={lambda_q:.4f}",flush=True)
    tag=f"theta_independent_theta{theta_idx:02d}_v7style_qcal"
    last_path=f"{TRAINING_RESULT_DIR}/{tag}_last_model.pt"; torch.save(net.state_dict(),last_path)
    norm_path=f"{TRAINING_RESULT_DIR}/{tag}_norm.pkl"; pickle.dump(norm,open(norm_path,'wb'))
    cfg={'theta_idx':theta_idx,'theta':d['theta'],'alpha':d['alpha'],'beta':d['beta'],'USE_SCENARIO_QUANTILE_CALIBRATION':USE_SCENARIO_QUANTILE_CALIBRATION,'SCENARIO_Q_TAUS':SCENARIO_Q_TAUS,'SCENARIO_Q_WEIGHTS':SCENARIO_Q_WEIGHTS,'LAM_SCEN_Q_A':LAM_SCEN_Q_A,'LAM_SCEN_Q_B':LAM_SCEN_Q_B,'LAM_SCEN_Q_C':LAM_SCEN_Q_C,'LAM_SCEN_Q_D':LAM_SCEN_Q_D,'USE_SCENARIO_MOMENT_CALIBRATION':USE_SCENARIO_MOMENT_CALIBRATION,'LAM_SCEN_MEAN':LAM_SCEN_MEAN,'LAM_SCEN_STD':LAM_SCEN_STD,'loss_old':'nll + LAM_T_SUP*t_sup + LAM_DISPATCH_SUP*disp + LAM_PHYS_FLEX*phys + beta_kl*kl'}
    cfg_path=f"{TRAINING_RESULT_DIR}/{tag}_config.json"; json.dump(cfg,open(cfg_path,'w'),indent=2)
    pd.DataFrame(rows).to_csv(f"{TRAINING_RESULT_DIR}/training_log_qcal_theta{theta_idx:02d}.csv",index=False)
    for ext,src in [('model.pt',best_path),('norm.pkl',norm_path),('config.json',cfg_path)]:
        dst=Path(ALL_THETA_RESULT_DIR)/f"theta_independent_theta{theta_idx:02d}_v7style_{ext}"
        shutil.copy2(src,dst)
    return net,norm,{'model_path':best_path,'norm_path':norm_path,'config_path':cfg_path}

def load_qcal_model_map(case):
    mp={}
    for j in HARD_THETA_LIST:
        m=Path(ALL_THETA_RESULT_DIR)/f"theta_independent_theta{j:02d}_v7style_model.pt"; n=Path(ALL_THETA_RESULT_DIR)/f"theta_independent_theta{j:02d}_v7style_norm.pkl"
        if m.exists() and n.exists():
            norm=pickle.load(open(n,'rb')); net=BayesSingleThetaV7StyleGMM2SupportNet(norm['x_mu_mean'].shape[1],case,hidden=160,depth=3,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO).to(DEVICE); net.load_state_dict(torch.load(m,map_location=DEVICE)); net.eval(); mp[j]={'net':net,'norm':norm,'model_path':str(m),'status':'loaded'}
        else:
            mp[j]={'net':None,'norm':None,'model_path':str(m),'status':'missing_model'}
    return mp

def run_qcal_only_targeted_mc2500_eval(case,data_dict):
    Path(TARGETED_DIAG_DIR).mkdir(parents=True,exist_ok=True)
    model_map=load_qcal_model_map(case); rows=[]
    for theta_idx,seed in DIAG_TARGET_PAIRS:
        if theta_idx not in HARD_THETA_LIST:
            continue
        row,ref=run_targeted_mc2500_case(case,data_dict,model_map,theta_idx,seed); rows.append(row)
        src=Path(f"{TARGETED_DIAG_DIR}/CDF_theta{theta_idx:02d}_seed{seed:03d}_mc{DIAG_MC_EVAL}_targeted.png")
        if src.exists(): shutil.copy2(src,Path(TARGETED_DIAG_DIR)/f"qcal_CDF_theta{theta_idx:02d}_seed{seed:03d}_mc{DIAG_MC_EVAL}.png")
    df=pd.DataFrame(rows); df.to_csv(f"{TARGETED_DIAG_DIR}/qcal_only_targeted_mc2500_metrics_by_pair.csv",index=False)
    old_path=Path('diagnostics_no_pcc_data1000_mc60_targeted/targeted_mc2500_metrics_by_pair.csv')
    old=pd.read_csv(old_path) if old_path.exists() else pd.DataFrame(); comp=[]
    for _,r in df.iterrows():
        theta_idx=int(r['theta_idx']); seed=int(r['seed']); oldr=old[(old.get('theta_idx',pd.Series(dtype=int))==theta_idx)&(old.get('seed',pd.Series(dtype=int))==seed)] if not old.empty else pd.DataFrame(); o=oldr.iloc[0] if len(oldr)>0 else pd.Series(dtype=float)
        old_arms=float(o.get('external_cdf_arms',np.nan)); new_arms=float(r.get('external_cdf_arms',np.nan)); old_q50=float(o.get('q50_error',np.nan)); new_q50=float(r.get('q50_error',np.nan))
        if np.isfinite(old_arms) and new_arms < old_arms-2.0: concl='quantile calibration significantly improves this case'
        elif new_arms>5.0: concl='remaining error is still large; likely needs hard-region data augmentation or realization-conditioned physics'
        elif np.isfinite(old_q50) and abs(new_q50)<abs(old_q50)-0.02 and new_arms>3.0: concl='location bias improved but shape/tail mismatch remains'
        else: concl='mixed result'
        comp.append({'theta_idx':theta_idx,'seed':seed,'old_arms_mc2500':old_arms,'new_arms_mc2500':new_arms,'delta_arms':new_arms-old_arms if np.isfinite(old_arms) else np.nan,'old_q05_error':o.get('q05_error',np.nan),'new_q05_error':r.get('q05_error',np.nan),'old_q50_error':old_q50,'new_q50_error':new_q50,'old_q95_error':o.get('q95_error',np.nan),'new_q95_error':r.get('q95_error',np.nan),'old_band_coverage':o.get('band_coverage_grid',np.nan),'new_band_coverage':r.get('band_coverage_grid',np.nan),'conclusion':concl})
    pd.DataFrame(comp).to_csv(f"{TARGETED_DIAG_DIR}/qcal_only_vs_old_targeted_mc2500_summary.csv",index=False)
    return df


def check_qcal_alltheta_model_completeness():
    rows=[]; missing=[]
    Path(ALL_THETA_COMPARISON_DIR).mkdir(parents=True,exist_ok=True)
    for theta_idx in ALL_THETA_LIST:
        model_path=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{theta_idx:02d}_v7style_model.pt")
        norm_path=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{theta_idx:02d}_v7style_norm.pkl")
        config_path=Path(f"{ALL_THETA_RESULT_DIR}/theta_independent_theta{theta_idx:02d}_v7style_config.json")
        ok=model_path.exists() and norm_path.exists()
        rows.append({'theta_idx':theta_idx,'model_exists':model_path.exists(),'norm_exists':norm_path.exists(),'config_exists':config_path.exists(),'ok':ok,'model_path':str(model_path)})
        if not ok: missing.append(theta_idx)
    out=f"{ALL_THETA_COMPARISON_DIR}/qcal_alltheta_model_completeness.csv"
    pd.DataFrame(rows).to_csv(out,index=False)
    print(f"[v7.4-qcal-only] model completeness CSV = {out}",flush=True)
    print(f"[v7.4-qcal-only] missing theta list = {missing}",flush=True)
    if missing:
        raise RuntimeError(f"Missing qcal all-theta models for theta={missing}")
    print('[v7.4-qcal-only] all 12 qcal theta models are complete',flush=True)
    return rows

def write_qcal_alltheta_external_summary_aliases():
    src=Path(ALL_THETA_COMPARISON_DIR)/'all_theta_multiseed_external_by_theta_seed.csv'
    if not src.exists():
        print(f"[v7.4-qcal-only-warning] all-theta by-theta-seed CSV missing: {src}",flush=True)
        return
    df=pd.read_csv(src)
    if df.empty:
        return
    theta_summary=[]
    for theta_idx,g in df.groupby('theta_idx'):
        arms=g['external_cdf_arms'].astype(float)
        theta_summary.append({'theta_idx':int(theta_idx),'n_seeds':int(len(g)),'mean_arms':float(arms.mean()),'median_arms':float(arms.median()),'max_arms':float(arms.max()),'min_arms':float(arms.min()),'q90_arms':float(arms.quantile(0.9)),'std_arms':float(arms.std(ddof=0)),'mean_q50_abs_error':float(g['q50_error'].abs().mean()) if 'q50_error' in g else np.nan,'mean_q95_abs_error':float(g['q95_error'].abs().mean()) if 'q95_error' in g else np.nan,'mean_band_coverage_grid':float(g['band_coverage_grid'].mean()) if 'band_coverage_grid' in g else np.nan,'mean_band_width':float(g['mean_band_width'].mean()) if 'mean_band_width' in g else np.nan,'mean_opf_per_sample_ms':float(g['opf_per_sample_ms'].mean()) if 'opf_per_sample_ms' in g else np.nan,'mean_nn_forward_ms':float(g['nn_forward_mean_ms'].mean()) if 'nn_forward_mean_ms' in g else np.nan})
    pd.DataFrame(theta_summary).to_csv(f"{ALL_THETA_COMPARISON_DIR}/all_theta_multiseed_external_by_theta_summary.csv",index=False)
    seed_summary=[]
    for seed,g in df.groupby('seed'):
        arms=g['external_cdf_arms'].astype(float); imax=arms.idxmax()
        seed_summary.append({'seed':int(seed),'n_theta':int(len(g)),'mean_arms':float(arms.mean()),'median_arms':float(arms.median()),'max_arms':float(arms.max()),'min_arms':float(arms.min()),'q90_arms':float(arms.quantile(0.9)),'std_arms':float(arms.std(ddof=0)),'worst_theta_idx':int(df.loc[imax,'theta_idx']),'worst_theta_arms':float(df.loc[imax,'external_cdf_arms'])})
    pd.DataFrame(seed_summary).to_csv(f"{ALL_THETA_COMPARISON_DIR}/all_theta_multiseed_external_by_seed_summary.csv",index=False)
    overall={'n_rows':int(len(df)),'n_theta':int(df['theta_idx'].nunique()),'n_seeds':int(df['seed'].nunique()),'mean_arms':float(df['external_cdf_arms'].mean()),'median_arms':float(df['external_cdf_arms'].median()),'max_arms':float(df['external_cdf_arms'].max())}
    pd.DataFrame([overall]).to_csv(f"{ALL_THETA_COMPARISON_DIR}/all_theta_multiseed_external_summary.csv",index=False)
    pd.DataFrame([overall]).to_csv(f"{ALL_THETA_COMPARISON_DIR}/all_theta_external_summary_overall.csv",index=False)
    print(f"[v7.4-qcal-only] all-theta evaluation output dir = {ALL_THETA_COMPARISON_DIR}",flush=True)




def _summarize_mc2500_df(df,out_dir):
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    df=df.copy().drop_duplicates(subset=['theta_idx','seed']).sort_values(['seed','theta_idx']).reset_index(drop=True)
    df.to_csv(f"{out_dir}/all_theta_multiseed_external_by_theta_seed.csv",index=False)
    theta_rows=[]
    for theta_idx,g in df.groupby('theta_idx'):
        arms=g['external_cdf_arms'].astype(float)
        theta_rows.append({'theta_idx':int(theta_idx),'n_seeds':int(len(g)),'mean_arms':float(arms.mean()),'median_arms':float(arms.median()),'max_arms':float(arms.max()),'min_arms':float(arms.min()),'q90_arms':float(arms.quantile(0.9)),'q95_arms':float(arms.quantile(0.95)),'std_arms':float(arms.std(ddof=0)),'mean_q50_abs_error':float(g['q50_error'].abs().mean()) if 'q50_error' in g else np.nan,'mean_q95_abs_error':float(g['q95_error'].abs().mean()) if 'q95_error' in g else np.nan,'mean_opf_per_sample_ms':float(g['opf_per_sample_ms'].mean()) if 'opf_per_sample_ms' in g else np.nan,'mean_nn_forward_ms':float(g['nn_forward_mean_ms'].mean()) if 'nn_forward_mean_ms' in g else np.nan})
    pd.DataFrame(theta_rows).to_csv(f"{out_dir}/all_theta_multiseed_external_by_theta_summary.csv",index=False)
    seed_rows=[]
    for seed,g in df.groupby('seed'):
        arms=g['external_cdf_arms'].astype(float); imax=arms.idxmax()
        seed_rows.append({'seed':int(seed),'n_theta':int(len(g)),'mean_arms':float(arms.mean()),'median_arms':float(arms.median()),'max_arms':float(arms.max()),'min_arms':float(arms.min()),'q90_arms':float(arms.quantile(0.9)),'q95_arms':float(arms.quantile(0.95)),'std_arms':float(arms.std(ddof=0)),'worst_theta_idx':int(df.loc[imax,'theta_idx']),'worst_theta_arms':float(df.loc[imax,'external_cdf_arms'])})
    pd.DataFrame(seed_rows).to_csv(f"{out_dir}/all_theta_multiseed_external_by_seed_summary.csv",index=False)
    arms=df['external_cdf_arms'].astype(float)
    overall={'n_total':int(len(df)),'n_seeds':int(df['seed'].nunique()),'n_theta':int(df['theta_idx'].nunique()),'mean_arms':float(arms.mean()),'median_arms':float(arms.median()),'q90_arms':float(arms.quantile(0.9)),'q95_arms':float(arms.quantile(0.95)),'max_arms':float(arms.max())}
    pd.DataFrame([overall]).to_csv(f"{out_dir}/all_theta_multiseed_external_summary.csv",index=False)
    pd.DataFrame([overall]).to_csv(f"{out_dir}/all_theta_external_summary_overall.csv",index=False)
    worst=df.sort_values('external_cdf_arms',ascending=False).head(20)
    worst.to_csv(f"{out_dir}/all_theta_external_worst20.csv",index=False)
    threshold={'n_total':int(len(df)),'n_gt_3pct':int((arms>3.0).sum()),'n_gt_4pct':int((arms>4.0).sum()),'n_gt_5pct':int((arms>5.0).sum()),'rate_gt_3pct':float((arms>3.0).mean()),'rate_gt_4pct':float((arms>4.0).mean()),'rate_gt_5pct':float((arms>5.0).mean()),'mean_arms':float(arms.mean()),'median_arms':float(arms.median()),'q90_arms':float(arms.quantile(0.9)),'q95_arms':float(arms.quantile(0.95)),'max_arms':float(arms.max())}
    pd.DataFrame([threshold]).to_csv(f"{out_dir}/all_theta_external_threshold_counts.csv",index=False)
    return overall,threshold,worst

def merge_existing_and_additional_mc2500_results():
    Path(COMBINED_MC2500_DIR).mkdir(parents=True,exist_ok=True)
    old_path=Path(EXISTING_MC2500_DIR)/'all_theta_multiseed_external_by_theta_seed.csv'
    new_path=Path(ALL_THETA_COMPARISON_DIR)/'all_theta_multiseed_external_by_theta_seed.csv'
    if not old_path.exists():
        raise FileNotFoundError(f'missing existing seed=1..3 MC2500 results: {old_path}')
    if not new_path.exists():
        raise FileNotFoundError(f'missing additional seed=4..10 MC2500 results: {new_path}')
    old=pd.read_csv(old_path); new=pd.read_csv(new_path)
    combo=pd.concat([old,new],ignore_index=True).drop_duplicates(subset=['theta_idx','seed']).sort_values(['seed','theta_idx']).reset_index(drop=True)
    overall,threshold,worst=_summarize_mc2500_df(combo,COMBINED_MC2500_DIR)
    print(f"[v7.4.3] combined n_seeds = {overall['n_seeds']}",flush=True)
    print(f"[v7.4.3] combined n_rows = {overall['n_total']}",flush=True)
    print(f"[v7.4.3] mean ARMS = {overall['mean_arms']:.4f}",flush=True)
    print(f"[v7.4.3] median ARMS = {overall['median_arms']:.4f}",flush=True)
    print(f"[v7.4.3] q90 ARMS = {overall['q90_arms']:.4f}",flush=True)
    print(f"[v7.4.3] q95 ARMS = {overall['q95_arms']:.4f}",flush=True)
    print(f"[v7.4.3] max ARMS = {overall['max_arms']:.4f}",flush=True)
    print(f"[v7.4.3] n/rate >3% = {threshold['n_gt_3pct']} / {threshold['rate_gt_3pct']:.4f}",flush=True)
    print(f"[v7.4.3] n/rate >4% = {threshold['n_gt_4pct']} / {threshold['rate_gt_4pct']:.4f}",flush=True)
    print(f"[v7.4.3] n/rate >5% = {threshold['n_gt_5pct']} / {threshold['rate_gt_5pct']:.4f}",flush=True)
    print('[v7.4.3] worst 20 theta/seed rows:',flush=True)
    print(worst[['theta_idx','seed','external_cdf_arms','q50_error']].to_string(index=False),flush=True)
    return combo,overall,threshold,worst

def main_v7_4_3_qcal_alltheta_mc2500_more_external_seeds():
    print('[v7.4.3-qcal-alltheta-more-seeds] start',flush=True)
    apply_v7_1_runtime_config()
    global ALL_THETA_LIST, TRAIN_THETA_LIST, HARD_THETA_LIST, ALL_THETA_EVAL_SEEDS, ALL_THETA_EVAL_MC
    ALL_THETA_LIST=list(range(N_THETA)); TRAIN_THETA_LIST=[]; HARD_THETA_LIST=[]
    ALL_THETA_EVAL_SEEDS=list(range(4,11)); ALL_THETA_EVAL_MC=2500
    assert BUILD_NEW_DATASET is False
    assert DATASET_CACHE_MODE == 'load_only'
    assert RUN_QUANTILE_CALIBRATION_TRAIN is False
    assert RUN_ALL_THETA_TRAIN is False
    assert RUN_ALL_THETA_EVAL is False
    assert RUN_FLEX_SYNTHESIS is False
    Path(ALL_THETA_COMPARISON_DIR).mkdir(parents=True,exist_ok=True); Path(COMBINED_MC2500_DIR).mkdir(parents=True,exist_ok=True)
    case=build_ieee33_case(); data_dict=get_or_build_flex_dataset_cache(case)
    check_qcal_alltheta_model_completeness()
    model_map=load_existing_all_theta_models(case)
    missing=[j for j in ALL_THETA_LIST if j not in model_map or model_map[j]['net'] is None]
    print(f'[v7.4.3] missing theta list = {missing}',flush=True)
    if missing:
        raise RuntimeError(f'Missing qcal model_map entries for theta={missing}')
    print('[v7.4.3] start additional external MC2500 evaluation: seeds=4..10, 12 theta, MC=2500',flush=True)
    eval_all_theta_multiseed_v7style(case,data_dict,model_map)
    write_qcal_alltheta_external_summary_aliases()
    merge_existing_and_additional_mc2500_results()
    print(f'[v7.4.3] additional output dir = {ALL_THETA_COMPARISON_DIR}',flush=True)
    print(f'[v7.4.3] combined output dir = {COMBINED_MC2500_DIR}',flush=True)
    print('[v7.4.3-qcal-alltheta-more-seeds] done',flush=True)

if __name__ == '__main__':
    main_v7_4_3_qcal_alltheta_mc2500_more_external_seeds()
