# -*- coding: utf-8 -*-
"""
flex-domain v1 support-function B-PINN
"""
import math, dataclasses, csv
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

import gurobipy as gp
from gurobipy import GRB

FLEX_FAST_MODE = True
if FLEX_FAST_MODE:
    NUM_SCENARIOS, MC_PER_SCENARIO, N_THETA, EPOCHS = 300, 30, 12, 400
else:
    NUM_SCENARIOS, MC_PER_SCENARIO, N_THETA, EPOCHS = 1000, 60, 16, 700
THETA_LIST = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
BATCH_SIZE, LR = 2048, 1e-3
LAM_BOUNDARY_SUP, LAM_DISPATCH_SUP, LAM_PHYS_FLEX, LAM_SUPPORT_CONSIST = 0.05, 0.05, 0.08, 0.10
LAM_H_QUANTILE, USE_H_QUANTILE_LOSS, H_QUANTILE_TAUS = 0.10, True, [0.05,0.25,0.5,0.75,0.95]
STABILIZE_OPF_DISPATCH, SUPPORT_EPS, W_QG_STAB, W_PG_STAB, W_PQ0_STAB = True, 1e-5, 1.0, 0.01, 1e-5
PRIOR_SIGMA, INIT_RHO, BETA_KL_MAX, KL_WARMUP_EPOCHS = 1.0, -5.0, 1.0, 500
RUN_SANITY_CHECKS, RUN_MULTI_TEST = True, True
N_TEST_SCENARIOS, MC_EVAL_MULTI, EVAL_THETA_SAMPLES = 20, 400, 40
SEED_DATA, SEED_TRAIN, SEED_EVAL = 0, 0, 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE_TEST = False
POLYGON_COORD_MAX = 20.0
POLYGON_CONVEX_CLEANUP = True
H_QUANTILE_BATCH_SCENARIOS = 32
H_QUANTILE_BATCH_THETAS = None
if SMOKE_TEST:
    NUM_SCENARIOS = 2
    MC_PER_SCENARIO = 3
    N_THETA = 4
    EPOCHS = 2
    RUN_MULTI_TEST = False
    MC_EVAL_MULTI = 50
    THETA_LIST = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
    print("[smoke-test] enabled")

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
    def pack(stabilized,h1):
        h=float(alpha*P0.X+beta*Q0.X)
        out={'ok':True,'h':h,'h_stage1':h1,'P0':float(P0.X),'Q0':float(Q0.X),'P':np.array([P[l].X for l in range(nl)]),'Q':np.array([Q[l].X for l in range(nl)]),'V':np.array([V[i].X for i in range(nb)]),'Pg':np.array([Pg[g].X for g in range(len(case.gen_buses))]),'Qg':np.array([Qg[g].X for g in range(len(case.gen_buses))]),'alpha':float(alpha),'beta':float(beta),'stabilized':stabilized}
        return out
    if not stabilize_dispatch:
        return pack(False,h_stage1) if return_detail else h_stage1
    m2,P,Q,V,P0,Q0,Pg,Qg=build_model(); add_constraints(m2,P,Q,V,P0,Q0,Pg,Qg)
    m2.addConstr(alpha*P0+beta*Q0>=h_stage1-SUPPORT_EPS)
    obj=gp.QuadExpr()
    for g in range(len(case.gen_buses)): obj += W_QG_STAB*Qg[g]*Qg[g] + W_PG_STAB*Pg[g]*Pg[g]
    obj += W_PQ0_STAB*(P0*P0+Q0*Q0)
    m2.setObjective(obj,GRB.MINIMIZE); m2.optimize()
    if m2.Status!=GRB.OPTIMAL:
        if return_detail:
            out=pack(False,h_stage1); out['stabilized']=False; return out
        return h_stage1
    out=pack(True,h_stage1)
    if abs(out['h']-(alpha*out['P0']+beta*out['Q0']))>1e-5:
        out['h']=alpha*out['P0']+beta*out['Q0']
    return out if return_detail else out['h']

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
        w=csv.writer(f); w.writerow(['theta_idx','unique_patterns','top1_ratio'])
        for k,v in sorted(by_theta.items()):
            c=Counter(v); top=c.most_common(1)[0][1]/len(v); w.writerow([k,len(c),f'{top:.6f}'])

def generate_flex_dataset(case,num_scenarios=NUM_SCENARIOS,mc_per_scenario=MC_PER_SCENARIO,theta_list=THETA_LIST,seed=SEED_DATA):
    rng=np.random.default_rng(seed); np.random.seed(seed)
    T=len(theta_list); theta_feat=np.stack([np.cos(theta_list),np.sin(theta_list)],axis=1)
    XMU=[]; XREAL=[]; YH=[]; YP0=[]; YQ0=[]; YPG=[]; YQG=[]; active=[]; alln=None; drop=0
    for s in range(num_scenarios):
        pd_mu,qd_mu,pr_mu,qr_mu=sample_scenario_means(case,rng); xmu=make_feature_vector(case,pd_mu,pr_mu)
        xr=[]; yh=np.zeros((mc_per_scenario,T)); yp0=np.zeros((mc_per_scenario,T)); yq0=np.zeros((mc_per_scenario,T)); ypg=np.zeros((mc_per_scenario,T,len(case.gen_buses))); yqg=np.zeros((mc_per_scenario,T,len(case.gen_buses)))
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
                    continue
                yh[m,j],yp0[m,j],yq0[m,j]=sol['h'],sol['P0'],sol['Q0']; ypg[m,j,:]=sol['Pg']; yqg[m,j,:]=sol['Qg']
                sig,an,alln=get_active_constraint_signature(case,sol); active.append({'signature':sig,'active_names':an,'theta_idx':j,'theta':float(th),'alpha':float(np.cos(th)),'beta':float(np.sin(th))})
            if not ok_scene: break
        # per-theta fallback fill
        fallback_fill_count = 0
        mean_fallback_count = 0
        infeasible_opf_count = int(np.sum(~np.isfinite(yh)))
        for j in range(T):
            valid=np.where(np.isfinite(yh[:,j]))[0]; miss=np.where(~np.isfinite(yh[:,j]))[0]
            if miss.size==0: continue
            if valid.size>0:
                pick=rng.choice(valid,size=miss.size,replace=True)
                for mm,pp in zip(miss,pick):
                    yh[mm,j],yp0[mm,j],yq0[mm,j]=yh[pp,j],yp0[pp,j],yq0[pp,j]
                    ypg[mm,j,:],yqg[mm,j,:]=ypg[pp,j,:],yqg[pp,j,:]
                    fallback_fill_count += 1
            else:
                solm=solve_flex_support_gurobi_33bus(case,pd_mu,qd_mu,pr_mu,qr_mu,float(np.cos(theta_list[j])),float(np.sin(theta_list[j])),return_detail=True,stabilize_dispatch=STABILIZE_OPF_DISPATCH)
                if solm['ok']:
                    yh[:,j]=solm['h']; yp0[:,j]=solm['P0']; yq0[:,j]=solm['Q0']; ypg[:,j,:]=solm['Pg']; yqg[:,j,:]=solm['Qg']
                    mean_fallback_count += mc_per_scenario
                else:
                    ok_scene=False
        if not ok_scene: drop+=1; continue
        XMU.append(xmu); XREAL.append(np.stack(xr)); YH.append(yh); YP0.append(yp0); YQ0.append(yq0); YPG.append(ypg); YQG.append(yqg)
    XMU=np.array(XMU); XREAL=np.array(XREAL); YH=np.array(YH); YP0=np.array(YP0); YQ0=np.array(YQ0); YPG=np.array(YPG); YQG=np.array(YQG)
    print(f'[flex-dataset] XMU shape={XMU.shape}'); print(f'[flex-dataset] XREAL shape={XREAL.shape}'); print(f'[flex-dataset] THETA_FEAT shape={theta_feat.shape}')
    print(f'[flex-dataset] YH shape={YH.shape}'); print(f'[flex-dataset] YP0/YQ0/YPG/YQG shape={YP0.shape}/{YQ0.shape}/{YPG.shape}/{YQG.shape}')
    print(f'[flex-dataset] total_OPF_labels = {max(0,XMU.shape[0])*mc_per_scenario*T}, dropped_scenarios={drop}')
    print(f'[flex-dataset] active_records count = {len(active)}, active_records_match = {len(active)==max(0,XMU.shape[0])*mc_per_scenario*T}')
    if XMU.size>0: print(f'[flex-dataset] h/P0/Q0/Pg/Qg ranges = {YH.min():.4f}-{YH.max():.4f} / {YP0.min():.4f}-{YP0.max():.4f} / {YQ0.min():.4f}-{YQ0.max():.4f} / {YPG.min():.4f}-{YPG.max():.4f} / {YQG.min():.4f}-{YQG.max():.4f}')
    return XMU,XREAL,theta_feat,YH,YP0,YQ0,YPG,YQG,active,alln

def flatten_flex_dataset(XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG):
    N,M,T=YH.shape; d=XMU.shape[1]; ng=YPG.shape[-1]
    xmu=np.repeat(XMU[:,None,None,:],M*T,axis=1).reshape(-1,d)
    xreal=np.repeat(XREAL[:,:,None,:],T,axis=2).reshape(-1,d)
    theta=np.repeat(THETA_FEAT[None,None,:,:],N*M,axis=0).reshape(-1,2)
    yh=YH.reshape(-1,1); yp0=YP0.reshape(-1,1); yq0=YQ0.reshape(-1,1); ypg=YPG.reshape(-1,ng); yqg=YQG.reshape(-1,ng)
    mask=np.isfinite(yh[:,0]) & np.isfinite(yp0[:,0]) & np.isfinite(yq0[:,0])
    print(f'[flex-dataset] flattened samples={mask.sum()}')
    return xmu[mask],xreal[mask],theta[mask],yh[mask],yp0[mask],yq0[mask],ypg[mask],yqg[mask]

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

def gmm2_log_prob(y,w,mu1,s1,mu2,s2):
    z1=(y-mu1)/s1; z2=(y-mu2)/s2
    l1=-0.5*z1**2-torch.log(s1)-0.5*math.log(2*math.pi); l2=-0.5*z2**2-torch.log(s2)-0.5*math.log(2*math.pi)
    logw=torch.log(w.clamp(min=1e-12)); return torch.logsumexp(torch.cat([logw[:,0:1]+l1,logw[:,1:2]+l2],dim=1),dim=1,keepdim=True)

def gmm2_cdf(z,w,mu1,s1,mu2,s2):
    return w[0]*norm.cdf((z-mu1)/(s1+1e-12))+w[1]*norm.cdf((z-mu2)/(s2+1e-12))

def gmm2_quantile_torch(w,mu1,s1,mu2,s2,taus,n_iter=50):
    B=w.shape[0]; tt=torch.tensor(taus,device=w.device,dtype=w.dtype).view(1,-1).repeat(B,1); K=tt.shape[1]
    lo=torch.minimum(mu1-8*s1,mu2-8*s2).repeat(1,K); hi=torch.maximum(mu1+8*s1,mu2+8*s2).repeat(1,K)
    for _ in range(n_iter):
        md=(lo+hi)/2; cdf=w[:,0:1]*(0.5*(1+torch.erf((md-mu1)/(s1+1e-12)/math.sqrt(2))))+w[:,1:2]*(0.5*(1+torch.erf((md-mu2)/(s2+1e-12)/math.sqrt(2))))
        go=cdf<tt; lo=torch.where(go,md,lo); hi=torch.where(go,hi,md)
    return (lo+hi)/2

class BayesFlexGMM2SupportNet(nn.Module):
    def __init__(self,in_dim,case,hidden=160,depth=3):
        super().__init__(); self.n_gen=len(case.gen_buses); self.x_dim=in_dim
        self.layers=nn.ModuleList(); d=in_dim+2
        for _ in range(depth): self.layers.append(BayesLinear(d,hidden,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO)); d=hidden
        self.gmm_out=BayesLinear(d,6,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO)
        self.rec_out=BayesLinear(d+in_dim+2+1,2+2*self.n_gen,prior_sigma=PRIOR_SIGMA,init_rho=INIT_RHO)
        self.act=nn.ReLU(); self.register_buffer('pg_min_t',torch.tensor(case.pg_min).view(1,-1).float()); self.register_buffer('pg_max_t',torch.tensor(case.pg_max).view(1,-1).float()); self.register_buffer('qg_min_t',torch.tensor(case.qg_min).view(1,-1).float()); self.register_buffer('qg_max_t',torch.tensor(case.qg_max).view(1,-1).float())
    def encode_gmm(self,x_mu_theta,sample=True):
        h=x_mu_theta
        for l in self.layers: h=self.act(l(h,sample=sample))
        return h
    def gmm_head(self,h,sample=True):
        o=self.gmm_out(h,sample=sample); w=torch.softmax(o[:,0:2],dim=1); mu1,mu2=o[:,2:3],o[:,4:5]; s1=torch.nn.functional.softplus(o[:,3:4])+1e-3; s2=torch.nn.functional.softplus(o[:,5:6])+1e-3
        return w,mu1,s1,mu2,s2
    def forward_gmm(self,x_mu_norm,theta_feat,sample=True):
        h=self.encode_gmm(torch.cat([x_mu_norm,theta_feat],dim=1),sample=sample); return h,*self.gmm_head(h,sample=sample)
    def recover_boundary_dispatch_from_h_theta(self,h_mu_theta,x_real_norm,theta_feat,h_label,h_mean,h_std,sample=True):
        h_norm=(h_label-h_mean)/(h_std+1e-9)
        o=self.rec_out(torch.cat([h_mu_theta,x_real_norm,theta_feat,h_norm],dim=1),sample=sample)
        p0=o[:,0:1]; q0=o[:,1:2]; pgr=o[:,2:2+self.n_gen]; qgr=o[:,2+self.n_gen:2+2*self.n_gen]
        pg=self.pg_min_t.to(o)+torch.sigmoid(pgr)*(self.pg_max_t.to(o)-self.pg_min_t.to(o)); qg=self.qg_min_t.to(o)+torch.sigmoid(qgr)*(self.qg_max_t.to(o)-self.qg_min_t.to(o))
        return p0,q0,pg,qg
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

def physics_loss_flex(case,x_real_raw,p0_hat,q0_hat,pg_hat,qg_hat,return_parts=False):
    rec=recover_flows_from_flex_dispatch_batch(case,x_real_raw,p0_hat,q0_hat,pg_hat,qg_hat); P,Q,V,pinj,qinj,pg,qg,pr=rec['P'],rec['Q'],rec['V'],rec['pinj'],rec['qinj'],rec['pg'],rec['qg'],rec['pr']; relu=torch.relu
    root_out=case.out_branches[case.root]; pccp=(P[:,root_out].sum(dim=1,keepdim=True)-p0_hat).pow(2).mean(); pccq=(Q[:,root_out].sum(dim=1,keepdim=True)-q0_hat).pow(2).mean()
    gp=(p0_hat+pinj.sum(dim=1,keepdim=True)).pow(2).mean(); gq=(q0_hat+qinj.sum(dim=1,keepdim=True)).pow(2).mean()
    v=((relu(case.vmin**2-V)**2)+(relu(V-case.vmax**2)**2)).mean(); fp=torch.tensor(case.fmax_p,device=x_real_raw.device).view(1,-1); fq=torch.tensor(case.fmax_q,device=x_real_raw.device).view(1,-1)
    lp=(relu(P.abs()-fp)**2).mean(); lq=(relu(Q.abs()-fq)**2).mean(); pgmn=torch.tensor(case.pg_min,device=x_real_raw.device).view(1,-1); pgmx=torch.tensor(case.pg_max,device=x_real_raw.device).view(1,-1); qgmn=torch.tensor(case.qg_min,device=x_real_raw.device).view(1,-1); qgmx=torch.tensor(case.qg_max,device=x_real_raw.device).view(1,-1)
    lpg=((relu(pgmn-pg)**2)+(relu(pg-pgmx)**2)).mean(); lqg=((relu(qgmn-qg)**2)+(relu(qg-qgmx)**2)).mean(); pv=(relu(pr[:,torch.tensor(case.pv_buses,device=x_real_raw.device)]-torch.tensor(case.pv_pmax,device=x_real_raw.device).view(1,-1))**2).mean()
    kcl=[]
    for i in range(case.n_bus):
        if i==case.root: continue
        kcl.append((P[:,case.in_branches[i]].sum(dim=1,keepdim=True)-P[:,case.out_branches[i]].sum(dim=1,keepdim=True)+pinj[:,i:i+1]).pow(2))
        kcl.append((Q[:,case.in_branches[i]].sum(dim=1,keepdim=True)-Q[:,case.out_branches[i]].sum(dim=1,keepdim=True)+qinj[:,i:i+1]).pow(2))
    lkcl=torch.cat(kcl,dim=1).mean(); loss=pccp+pccq+gp+gq+v+lp+lq+lpg+lqg+pv+0.1*lkcl
    if return_parts: return loss,{"pcc_p":pccp.detach(),"pcc_q":pccq.detach(),"global_p":gp.detach(),"global_q":gq.detach(),"voltage":v.detach(),"line_p":lp.detach(),"line_q":lq.detach(),"pg":lpg.detach(),"qg":lqg.detach(),"pv":pv.detach(),"kcl":lkcl.detach()}
    return loss

def compute_h_quantile_loss(net,XMU_scen,THETA_FEAT,YH_scen,x_mu_mean,x_mu_std,taus=H_QUANTILE_TAUS,n_scenarios_sample=H_QUANTILE_BATCH_SCENARIOS,n_thetas_sample=H_QUANTILE_BATCH_THETAS,sample=True):
    if XMU_scen.shape[0]==0: return torch.tensor(0.0,device=DEVICE)
    rng=np.random.default_rng(SEED_TRAIN)
    scen_idx=rng.choice(XMU_scen.shape[0],size=min(n_scenarios_sample,XMU_scen.shape[0]),replace=False)
    losses=[]
    for i in scen_idx:
        theta_idx=np.arange(THETA_FEAT.shape[0]) if n_thetas_sample is None else rng.choice(THETA_FEAT.shape[0],size=min(n_thetas_sample,THETA_FEAT.shape[0]),replace=False)
        xmu=((XMU_scen[i:i+1]-x_mu_mean)/x_mu_std)
        xmu_t=torch.tensor(xmu,dtype=torch.float32,device=DEVICE)
        for j in theta_idx:
            hs=YH_scen[i,:,j]
            if not np.all(np.isfinite(hs)): continue
            q_emp=np.quantile(hs,taus).reshape(1,-1)
            th_t=torch.tensor(THETA_FEAT[j:j+1],dtype=torch.float32,device=DEVICE)
            with torch.set_grad_enabled(sample):
                _,w,mu1,s1,mu2,s2=net.forward_gmm(xmu_t,th_t,sample=sample)
                q_pred=gmm2_quantile_torch(w,mu1,s1,mu2,s2,taus)
                losses.append(((q_pred-torch.tensor(q_emp,dtype=torch.float32,device=DEVICE))**2).mean())
    if len(losses)==0: return torch.tensor(0.0,device=DEVICE)
    return torch.stack(losses).mean()

def train_bayes_flex_gmm2(case,XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG):
    rng=np.random.default_rng(SEED_TRAIN); n_scen=XMU.shape[0]; n_val=max(1,int(0.1*n_scen)); idx=rng.permutation(n_scen); tr,va=idx[:-n_val],idx[-n_val:]
    def split(arr): return arr[tr],arr[va]
    XMU_tr,XMU_va=split(XMU); XREAL_tr,XREAL_va=split(XREAL); YH_tr,YH_va=split(YH); YP0_tr,YP0_va=split(YP0); YQ0_tr,YQ0_va=split(YQ0); YPG_tr,YPG_va=split(YPG); YQG_tr,YQG_va=split(YQG)
    xmu_mean=XMU_tr.mean(0,keepdims=True); xmu_std=XMU_tr.std(0,keepdims=True)+1e-9; xr_mean=XREAL_tr.reshape(-1,XREAL_tr.shape[-1]).mean(0,keepdims=True); xr_std=XREAL_tr.reshape(-1,XREAL_tr.shape[-1]).std(0,keepdims=True)+1e-9
    h_mean=np.array([[YH_tr.mean()]]); h_std=np.array([[YH_tr.std()+1e-9]]); p0_mean=np.array([[YP0_tr.mean()]]); p0_std=np.array([[YP0_tr.std()+1e-9]]); q0_mean=np.array([[YQ0_tr.mean()]]); q0_std=np.array([[YQ0_tr.std()+1e-9]])
    trf=flatten_flex_dataset(XMU_tr,XREAL_tr,THETA_FEAT,YH_tr,YP0_tr,YQ0_tr,YPG_tr,YQG_tr); vaf=flatten_flex_dataset(XMU_va,XREAL_va,THETA_FEAT,YH_va,YP0_va,YQ0_va,YPG_va,YQG_va)
    xmu_f,xr_f,th_f,yh_f,yp0_f,yq0_f,ypg_f,yqg_f=trf; xmu_v,xr_v,th_v,yh_v,yp0_v,yq0_v,ypg_v,yqg_v=vaf
    to=lambda a:torch.tensor(a,dtype=torch.float32,device=DEVICE)
    xmu_t,xr_t,th_t,yh_t,yp0_t,yq0_t,ypg_t,yqg_t=map(to,[ (xmu_f-xmu_mean)/xmu_std, (xr_f-xr_mean)/xr_std, th_f,yh_f,yp0_f,yq0_f,ypg_f,yqg_f ])
    xr_raw_t=to(xr_f)
    xmu_tv,xr_tv,th_tv,yh_tv,yp0_tv,yq0_tv,ypg_tv,yqg_tv=map(to,[ (xmu_v-xmu_mean)/xmu_std, (xr_v-xr_mean)/xr_std, th_v,yh_v,yp0_v,yq0_v,ypg_v,yqg_v ])
    xr_raw_v=to(xr_v)
    net=BayesFlexGMM2SupportNet(XMU.shape[1],case).to(DEVICE); opt=torch.optim.Adam(net.parameters(),lr=LR); n=xmu_t.shape[0]; nb=(n+BATCH_SIZE-1)//BATCH_SIZE
    hm,hs,p0s,q0s=to(h_mean),to(h_std),to(p0_std),to(q0_std)
    print('=== 开始训练 33-bus VI-BPINN Flex-Domain (support-function formulation) ===')
    for ep in range(EPOCHS):
        perm=rng.permutation(n); beta=BETA_KL_MAX*min(1.0,(ep+1)/KL_WARMUP_EPOCHS); acc=np.zeros(7)
        for b in range(nb):
            ii=perm[b*BATCH_SIZE:min((b+1)*BATCH_SIZE,n)]
            xmu, xr, xr_raw, th, yh, yp0, yq0, ypg, yqg = xmu_t[ii], xr_t[ii], xr_raw_t[ii], th_t[ii], yh_t[ii], yp0_t[ii], yq0_t[ii], ypg_t[ii], yqg_t[ii]
            henc,w,mu1,s1,mu2,s2=net.forward_gmm(xmu,th,sample=True); nll=(-gmm2_log_prob(yh,w,mu1,s1,mu2,s2)).mean()
            p0h,q0h,pgh,qgh=net.recover_boundary_dispatch_from_h_theta(henc,xr,th,yh,hm,hs,sample=True)
            bsup=((p0h-yp0)/p0s).pow(2).mean()+((q0h-yq0)/q0s).pow(2).mean()
            dsup=((pgh-ypg)/(net.pg_max_t-net.pg_min_t+1e-6)).pow(2).mean()+((qgh-yqg)/(net.qg_max_t-net.qg_min_t+1e-6)).pow(2).mean()
            phys=physics_loss_flex(case,xr_raw,p0h,q0h,pgh,qgh)
            scons=((th[:,0:1]*p0h+th[:,1:2]*q0h-yh)**2).mean()
            hq=torch.tensor(0.0,device=DEVICE)
            if USE_H_QUANTILE_LOSS:
                hq = compute_h_quantile_loss(net,XMU_tr,THETA_FEAT,YH_tr,xmu_mean,xmu_std,taus=H_QUANTILE_TAUS,n_scenarios_sample=H_QUANTILE_BATCH_SCENARIOS,n_thetas_sample=H_QUANTILE_BATCH_THETAS,sample=True)
            kl=net.kl_divergence(); loss=nll+LAM_BOUNDARY_SUP*bsup+LAM_DISPATCH_SUP*dsup+LAM_PHYS_FLEX*phys+LAM_SUPPORT_CONSIST*scons+LAM_H_QUANTILE*hq+beta*kl/n
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
            acc += np.array([float(loss.detach().cpu()),float(nll.detach().cpu()),float(bsup.detach().cpu()),float(dsup.detach().cpu()),float(phys.detach().cpu()),float(scons.detach().cpu()),float(hq.detach().cpu())])
        if ep==0 or (ep+1)%50==0 or ep+1==EPOCHS:
            with torch.no_grad():
                henc,w,mu1,s1,mu2,s2=net.forward_gmm(xmu_tv,th_tv,sample=False); nllv=(-gmm2_log_prob(yh_tv,w,mu1,s1,mu2,s2)).mean()
                p0h,q0h,pgh,qgh=net.recover_boundary_dispatch_from_h_theta(henc,xr_tv,th_tv,yh_tv,hm,hs,sample=False)
                bsv=((p0h-yp0_tv)/p0s).pow(2).mean()+((q0h-yq0_tv)/q0s).pow(2).mean(); dsv=((pgh-ypg_tv)/(net.pg_max_t-net.pg_min_t+1e-6)).pow(2).mean()+((qgh-yqg_tv)/(net.qg_max_t-net.qg_min_t+1e-6)).pow(2).mean()
                phv,parts=physics_loss_flex(case,xr_raw_v,p0h,q0h,pgh,qgh,return_parts=True); scv=((th_tv[:,0:1]*p0h+th_tv[:,1:2]*q0h-yh_tv)**2).mean(); hqv=compute_h_quantile_loss(net,XMU_va,THETA_FEAT,YH_va,xmu_mean,xmu_std,taus=H_QUANTILE_TAUS,n_scenarios_sample=min(H_QUANTILE_BATCH_SCENARIOS,max(1,XMU_va.shape[0])),n_thetas_sample=H_QUANTILE_BATCH_THETAS,sample=False)
                lv=nllv+LAM_BOUNDARY_SUP*bsv+LAM_DISPATCH_SUP*dsv+LAM_PHYS_FLEX*phv+LAM_SUPPORT_CONSIST*scv+LAM_H_QUANTILE*hqv+beta*net.kl_divergence()/n
            print(f"Epoch {ep+1:4d} | avg_loss={acc[0]/nb:.6f} avg_nll_h={acc[1]/nb:.6f} avg_boundary_sup={acc[2]/nb:.6f} avg_dispatch_sup={acc[3]/nb:.6f} avg_phys_flex={acc[4]/nb:.6f} avg_support_consist={acc[5]/nb:.6f} avg_h_quantile={acc[6]/nb:.6f} avg_kl/N={float((net.kl_divergence()/n).detach().cpu()):.6f} beta={beta:.4f} n_batches={nb} | val_loss={lv.item():.6f} val_nll_h={nllv.item():.6f} val_boundary_sup={bsv.item():.6f} val_dispatch_sup={dsv.item():.6f} val_phys_flex={phv.item():.6f} val_support_consist={scv.item():.6f} val_h_quantile={hqv.item():.6f} val_pcc_p={parts['pcc_p'].item():.3e} val_pcc_q={parts['pcc_q'].item():.3e} val_global_p={parts['global_p'].item():.3e} val_global_q={parts['global_q'].item():.3e} val_voltage={parts['voltage'].item():.3e} val_line_p={parts['line_p'].item():.3e} val_line_q={parts['line_q'].item():.3e} val_pg={parts['pg'].item():.3e} val_qg={parts['qg'].item():.3e}")
    norm={'x_mu_mean':xmu_mean,'x_mu_std':xmu_std,'x_real_mean':xr_mean,'x_real_std':xr_std,'h_mean':h_mean,'h_std':h_std,'p0_mean':p0_mean,'p0_std':p0_std,'q0_mean':q0_mean,'q0_std':q0_std}
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
                p0h,q0h,pgh,qgh=net.recover_boundary_dispatch_from_h_theta(henc,torch.tensor(xr_n,dtype=torch.float32,device=DEVICE),th_t,yh_t,hm,hs,sample=False)
                rec=physics_loss_flex(case,torch.tensor(xr,dtype=torch.float32,device=DEVICE),p0h,q0h,pgh,qgh,return_parts=True)[1]
            print(f"[flex-real] th={th:.2f} support_res={(alpha*float(p0h)-0 + beta*float(q0h)-sol['h']):+.2e} L1_Pg={np.abs(pgh.cpu().numpy().reshape(-1)-sol['Pg']).sum():.4f} L1_Qg={np.abs(qgh.cpu().numpy().reshape(-1)-sol['Qg']).sum():.4f} pccP={rec['pcc_p'].item():.2e} pccQ={rec['pcc_q'].item():.2e}")

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
        ys=np.sort(np.array(ys)); z=np.linspace(ys.min()-0.2,ys.max()+0.2,400); cdf_mc=np.searchsorted(ys,z,side='right')/len(ys)
        xmu=make_feature_vector(case,pdm,prm).reshape(1,-1); xmu_n=(xmu-norm['x_mu_mean'])/norm['x_mu_std']; xt=torch.tensor(xmu_n,dtype=torch.float32,device=DEVICE); th_t=torch.tensor([[alpha,beta]],dtype=torch.float32,device=DEVICE)
        cdfs=[]
        with torch.no_grad():
            for _ in range(EVAL_THETA_SAMPLES):
                _,w,mu1,s1,mu2,s2=net.forward_gmm(xt,th_t,sample=True)
                cdfs.append(gmm2_cdf(z,w.cpu().numpy().reshape(-1),float(mu1),float(s1),float(mu2),float(s2)))
        cdfs=np.array(cdfs); cm=cdfs.mean(0); clo=np.quantile(cdfs,0.025,0); chi=np.quantile(cdfs,0.975,0); arms=100*math.sqrt(np.mean((cm-cdf_mc)**2))
        ax.fill_between(z,clo,chi,alpha=0.2); ax.plot(z,cdf_mc,'k',lw=2); ax.plot(z,cm,'--',lw=2); ax.set_title(f'theta={th:.2f}, ARMS={arms:.2f}%')
    plt.tight_layout(); plt.savefig('FlexDomain_CDF_selected_directions_v1_full.png',dpi=250)

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
        ys=np.array(ys); h_mc_q05.append(np.quantile(ys,0.05)); h_mc_q50.append(np.quantile(ys,0.50)); h_mc_q95.append(np.quantile(ys,0.95))
        xmu_n=(xmu-norm['x_mu_mean'])/norm['x_mu_std']; xt=torch.tensor(xmu_n,dtype=torch.float32,device=DEVICE); th_t=torch.tensor([[alpha,beta]],dtype=torch.float32,device=DEVICE)
        with torch.no_grad():
            _,w,mu1,s1,mu2,s2=net.forward_gmm(xt,th_t,sample=False)
            q=gmm2_quantile_torch(w,mu1,s1,mu2,s2,[0.05,0.5,0.95]).cpu().numpy().reshape(-1)
            h_bn_q05.append(q[0]); h_bn_q50.append(q[1]); h_bn_q95.append(q[2])
    poly_mc50=support_values_to_polygon(theta_list,np.array(h_mc_q50)); poly_bn50=support_values_to_polygon(theta_list,np.array(h_bn_q50)); poly_mc05=support_values_to_polygon(theta_list,np.array(h_mc_q05)); poly_mc95=support_values_to_polygon(theta_list,np.array(h_mc_q95)); poly_bn05=support_values_to_polygon(theta_list,np.array(h_bn_q05)); poly_bn95=support_values_to_polygon(theta_list,np.array(h_bn_q95))
    plt.figure(figsize=(8,7),dpi=130)
    for poly,lab,c,ls in [(poly_mc50,'MC median','k','-'),(poly_bn50,'B-PINN median','#d95f02','--'),(poly_mc05,'MC q05','0.4',':'),(poly_mc95,'MC q95','0.4',':'),(poly_bn05,'B-PINN q05','#1b9e77',':'),(poly_bn95,'B-PINN q95','#1b9e77',':')]:
        if len(poly)>2: plt.plot(np.r_[poly[:,0],poly[0,0]],np.r_[poly[:,1],poly[0,1]],ls,color=c,label=lab)
    plt.xlabel('P0 (MW)'); plt.ylabel('Q0 (Mvar)'); plt.title('IEEE 33-bus Flexibility Domain — support-function B-PINN'); plt.legend(); plt.axis('equal'); plt.tight_layout(); plt.savefig('FlexDomain_33bus_v1_support_function_full.png',dpi=280)

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
                    _,w,mu1,s1,mu2,s2=net.forward_gmm(xt,th_t,sample=True); c.append(gmm2_cdf(z,w.cpu().numpy().reshape(-1),float(mu1),float(s1),float(mu2),float(s2)))
            arms=100*math.sqrt(np.mean((np.mean(np.array(c),axis=0)-cdf_mc)**2)); all_arms.append((s,th,arms))
    arr=np.array([a[2] for a in all_arms]); print(f'[multi-flex] overall mean={arr.mean():.4f}% median={np.median(arr):.4f}% q90={np.quantile(arr,0.9):.4f}% max={arr.max():.4f}%')



def polygon_area(poly):
    if poly is None or len(poly)<3: return 0.0
    x=poly[:,0]; y=poly[:,1]
    return 0.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))

def support_values_to_polygon(theta_list,h_values,eps=1e-9,do_convex_cleanup=True):
    pts=[]
    for j in range(len(theta_list)):
        k=(j+1)%len(theta_list)
        d1=np.array([np.cos(theta_list[j]),np.sin(theta_list[j])]); d2=np.array([np.cos(theta_list[k]),np.sin(theta_list[k])])
        A=np.vstack([d1,d2]); b=np.array([h_values[j],h_values[k]])
        if np.linalg.cond(A)>1e10: continue
        try: y=np.linalg.solve(A,b)
        except Exception: continue
        if np.all(np.isfinite(y)) and np.max(np.abs(y))<=POLYGON_COORD_MAX: pts.append(y)
    if len(pts)<3: return np.zeros((0,2))
    pts=np.unique(np.round(np.array(pts),10),axis=0)
    if do_convex_cleanup and len(pts)>=3:
        ang=np.arctan2(pts[:,1],pts[:,0]); pts=pts[np.argsort(ang)]
    return pts
def main():
    case=build_ieee33_case()
    if RUN_SANITY_CHECKS: flex_opf_sanity_check(case)
    print('生成 33 节点灵活域训练数据...')
    print(f'[flex-config] FLEX_FAST_MODE={FLEX_FAST_MODE}')
    print(f'[flex-config] NUM_SCENARIOS={NUM_SCENARIOS}, MC_PER_SCENARIO={MC_PER_SCENARIO}, N_THETA={N_THETA}, total_OPF_labels={NUM_SCENARIOS*MC_PER_SCENARIO*N_THETA}')
    XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG,active,alln=generate_flex_dataset(case,NUM_SCENARIOS,MC_PER_SCENARIO,THETA_LIST,SEED_DATA)
    summarize_active_patterns(active,alln,top_k=10)
    net,norm=train_bayes_flex_gmm2(case,XMU,XREAL,THETA_FEAT,YH,YP0,YQ0,YPG,YQG)
    if RUN_SANITY_CHECKS: flex_realization_sanity_check(case,net,norm,THETA_LIST)
    eval_and_plot_flex_domain(case,net,norm,THETA_LIST)
    eval_and_plot_direction_cdfs(case,net,norm,THETA_LIST)
    if RUN_MULTI_TEST: eval_multiple_flex_scenarios(case,net,norm,THETA_LIST)

if __name__=='__main__':
    main()
