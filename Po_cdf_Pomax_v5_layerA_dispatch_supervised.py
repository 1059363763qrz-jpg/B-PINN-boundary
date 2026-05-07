# -*- coding: utf-8 -*-
"""
Stage-1 extension: 3-bus VI-BPINN -> IEEE 33-bus single-period VI-BPINN.

[DIAG VERSION: layer-A dispatch physics + OPF dispatch supervision]

Compared with v4_layerA:
- keep Bayesian GMM-2 output for Pomax conditional distribution
- keep neural dispatch head for Pg/Qg
- keep layer-A physics loss based on predicted Pg/Qg
- additionally save OPF internal dispatch labels Pg_OPF/Qg_OPF from Gurobi
- add supervised Pg/Qg dispatch recovery loss during training
- still single-period, no ESS/EV/24h, no KKT, no differentiable OPF layer

Core kept from v4:
- IEEE 33-bus single-period LinDistFlow model
- Gurobi OPF label generator solve_pomax_gurobi_33bus()
- Bayesian NN (VI) + GMM-2 output for Pomax conditional distribution
- ELBO style training objective: NLL + supervised dispatch + physics loss + beta * KL/N
- evaluate with MC baseline CDF, ARMS, q0.05 comparison

Layer-A + dispatch-supervision notes:
- P0 supervision is still performed by the GMM negative log-likelihood (NLL), not plain MSE.
- Pg/Qg supervision is performed by MSE against OPF internal dispatch labels.
- Physics loss continues to constrain consistency among predicted Pg/Qg, predicted P0 mean,
  branch flows, voltages, line limits, and generator/PV constraints.
- OPF internal dispatch solutions can be non-unique, so the Pg/Qg supervised loss weight is
  intentionally small to avoid overwhelming P0 distribution learning.
- no 24h/ESS/EV, no KKT, no HMC/dropout, no differentiable OPF layer, no GMM3
"""

import math
import dataclasses
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise RuntimeError("需要安装并配置 Gurobi。") from e


# =============================
# 0) Hyperparameters
# =============================
FAST_MODE = False
if FAST_MODE:
    NUM_SCENARIOS = 600
    MC_PER_SCENARIO = 60
    EPOCHS = 400
else:
    NUM_SCENARIOS = 1000
    MC_PER_SCENARIO = 60
    EPOCHS = 700

BATCH_SIZE = 2048
LR = 1e-3
LAM_PHYS = 0.08
LAM_DISPATCH_SUP = 0.05
LAM_PG_SUP = 1.0
LAM_QG_SUP = 0.5
NORMALIZE_DISPATCH_SUP = True

PRIOR_SIGMA = 1.0
BETA_KL_MAX = 1.0
KL_WARMUP_EPOCHS = 500
INIT_RHO = -5.0
TRAIN_WEIGHT_SAMPLES = 1
EVAL_THETA_SAMPLES = 50
RUN_SANITY_CHECKS = True
RUN_MULTI_TEST = True
N_TEST_SCENARIOS = 20
MC_EVAL_MULTI = 800
LOG_EVERY = 50
VAL_RATIO = 0.10

SEED_DATA = 0
SEED_TRAIN = 0
SEED_EVAL = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)


# =============================
# 1) 33-bus case builder
# =============================
@dataclasses.dataclass
class GridCase:
    n_bus: int
    root: int
    from_bus: np.ndarray
    to_bus: np.ndarray
    r: np.ndarray
    x: np.ndarray
    vmin: float
    vmax: float
    fmax_p: np.ndarray
    fmax_q: np.ndarray
    pd_base: np.ndarray
    qd_base: np.ndarray
    pv_buses: np.ndarray
    pv_pmax: np.ndarray
    pv_pf: float
    gen_buses: np.ndarray
    pg_min: np.ndarray
    pg_max: np.ndarray
    qg_min: np.ndarray
    qg_max: np.ndarray
    children: List[List[int]]
    parent_branch: np.ndarray
    parent_bus: np.ndarray
    topo_order: List[int]
    rev_topo_order: List[int]
    in_branches: List[List[int]]
    out_branches: List[List[int]]


def _build_radial_topology(n_bus: int, root: int, fb: np.ndarray, tb: np.ndarray):
    n_br = fb.size
    children = [[] for _ in range(n_bus)]
    parent_branch = -np.ones(n_bus, dtype=int)
    parent_bus = -np.ones(n_bus, dtype=int)

    for l in range(n_br):
        i, j = int(fb[l]), int(tb[l])
        children[i].append(j)
        parent_branch[j] = l
        parent_bus[j] = i

    topo = []
    stack = [root]
    while stack:
        i = stack.pop()
        topo.append(i)
        for c in children[i][::-1]:
            stack.append(c)

    rev = topo[::-1]
    in_branches = [[] for _ in range(n_bus)]
    out_branches = [[] for _ in range(n_bus)]
    for l in range(n_br):
        i, j = int(fb[l]), int(tb[l])
        out_branches[i].append(l)
        in_branches[j].append(l)
    return children, parent_branch, parent_bus, topo, rev, in_branches, out_branches


def build_ieee33_case() -> GridCase:
    # Simplified IEEE-33 radial data in p.u.-style magnitudes (single-period).
    # buses: internal index 0..32 maps to bus number 1..33.
    n_bus = 33
    root = 0

    # standard 33-bus radial connectivity (1-2-... with branches)
    # 1-based branch list converted to 0-based.
    br_1based = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
        (2, 19), (19, 20), (20, 21), (21, 22), (3, 23), (23, 24), (24, 25), (6, 26),
        (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33),
    ]
    fb = np.array([i - 1 for i, _ in br_1based], dtype=int)
    tb = np.array([j - 1 for _, j in br_1based], dtype=int)
    n_br = fb.size

    # branch r/x: compact, runnable simplified set (can be replaced with full benchmark)
    r = np.array([
        0.0922, 0.4930, 0.3660, 0.3811, 0.8190, 0.1872, 0.7114, 1.0300,
        1.0440, 0.1966, 0.3744, 1.4680, 0.5416, 0.5910, 0.7463, 1.2890,
        0.7320, 0.1640, 1.5042, 0.4095, 0.7089, 0.4512, 0.8980, 0.8960,
        0.2030, 0.2842, 1.0590, 0.8042, 0.5075, 0.9744, 0.3105, 0.3410,
    ], dtype=float) * 0.01
    x = np.array([
        0.0470, 0.2511, 0.1864, 0.1941, 0.7070, 0.6188, 0.2351, 0.7400,
        0.7400, 0.0650, 0.1238, 1.1550, 0.7129, 0.5260, 0.5450, 1.7210,
        0.5740, 0.1565, 1.3554, 0.4784, 0.9373, 0.3083, 0.7091, 0.7011,
        0.1034, 0.1447, 0.9337, 0.7006, 0.2585, 0.9630, 0.3619, 0.5302,
    ], dtype=float) * 0.01

    # base load (MW/Mvar, simplified but close to IEEE-33 style levels)
    pd = np.zeros(n_bus, dtype=float)
    qd = np.zeros(n_bus, dtype=float)
    load_data = {
        2: (0.10, 0.06), 3: (0.09, 0.04), 4: (0.12, 0.08), 5: (0.06, 0.03), 6: (0.06, 0.02),
        7: (0.20, 0.10), 8: (0.20, 0.10), 9: (0.06, 0.02), 10: (0.06, 0.02), 11: (0.045, 0.03),
        12: (0.06, 0.035), 13: (0.06, 0.035), 14: (0.12, 0.08), 15: (0.06, 0.01), 16: (0.06, 0.02),
        17: (0.06, 0.02), 18: (0.09, 0.04), 19: (0.09, 0.04), 20: (0.09, 0.04), 21: (0.09, 0.04),
        22: (0.09, 0.04), 23: (0.09, 0.05), 24: (0.42, 0.20), 25: (0.42, 0.20), 26: (0.06, 0.025),
        27: (0.06, 0.025), 28: (0.06, 0.02), 29: (0.12, 0.07), 30: (0.20, 0.60), 31: (0.15, 0.07),
        32: (0.21, 0.10), 33: (0.06, 0.04),
    }
    for b1, (p, q) in load_data.items():
        pd[b1 - 1] = p
        qd[b1 - 1] = q

    # configurable DER and generators (stage-1)
    pv_buses = np.array([7, 14, 24, 30], dtype=int) - 1
    pv_pmax = np.array([0.35, 0.30, 0.60, 0.50], dtype=float)
    pv_pf = 0.98

    gen_buses = np.array([13, 18, 25, 33], dtype=int) - 1
    pg_min = np.array([0.00, 0.00, 0.00, 0.00], dtype=float)
    pg_max = np.array([0.45, 0.40, 0.90, 0.75], dtype=float)
    # keep Q ranges relatively loose in stage-1 to avoid over-filtering all samples
    # when PCC reactive exchange is fixed to zero.
    qg_min = np.array([-1.20, -1.00, -1.80, -1.50], dtype=float)
    qg_max = np.array([1.20, 1.00, 1.80, 1.50], dtype=float)

    fmax_p = np.full(n_br, 5.0, dtype=float)
    fmax_q = np.full(n_br, 5.0, dtype=float)

    children, parent_branch, parent_bus, topo, rev, in_branches, out_branches = _build_radial_topology(n_bus, root, fb, tb)

    return GridCase(
        n_bus=n_bus, root=root,
        from_bus=fb, to_bus=tb, r=r, x=x,
        vmin=0.95, vmax=1.05,
        fmax_p=fmax_p, fmax_q=fmax_q,
        pd_base=pd, qd_base=qd,
        pv_buses=pv_buses, pv_pmax=pv_pmax, pv_pf=pv_pf,
        gen_buses=gen_buses,
        pg_min=pg_min, pg_max=pg_max, qg_min=qg_min, qg_max=qg_max,
        children=children, parent_branch=parent_branch, parent_bus=parent_bus,
        topo_order=topo, rev_topo_order=rev,
        in_branches=in_branches, out_branches=out_branches,
    )


# =============================
# 2) Generic OPF labeler (Gurobi)
# =============================
def solve_pomax_gurobi_33bus(case: GridCase, pd: np.ndarray, qd: np.ndarray, pr: np.ndarray, qr: np.ndarray, return_detail: bool = False):
    """
    Unified sign convention (used by both OPF and physics loss):
    - Branch flow P_ij/Q_ij positive in from_bus -> to_bus direction.
    - Nodal net injection pinj/qinj = generation + renewable - load.
      Thus pinj>0 means net generation, pinj<0 means net load.
    - PCC variable P0 is import from upstream grid into root bus.
      Hence P0 = sum_{l in out(root)} P_l, and P0>0 means importing power.
    """
    nb = case.n_bus
    nl = case.from_bus.size

    m = gp.Model("pomax_33bus")
    m.Params.OutputFlag = 0

    P0 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P0")
    P = m.addVars(nl, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Pij")
    Q = m.addVars(nl, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Qij")
    V = m.addVars(nb, lb=case.vmin ** 2, ub=case.vmax ** 2, name="Vsq")

    Pg = m.addVars(len(case.gen_buses), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Pg")
    Qg = m.addVars(len(case.gen_buses), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Qg")

    for g in range(len(case.gen_buses)):
        m.addConstr(Pg[g] >= float(case.pg_min[g]))
        m.addConstr(Pg[g] <= float(case.pg_max[g]))
        m.addConstr(Qg[g] >= float(case.qg_min[g]))
        m.addConstr(Qg[g] <= float(case.qg_max[g]))

    m.addConstr(V[case.root] == 1.0, name="slackV")

    # PCC at root (bus-1): import active power from upstream.
    root_out = case.out_branches[case.root]
    m.addConstr(gp.quicksum(P[l] for l in root_out) == P0, name="pcc_p")
    m.addConstr(gp.quicksum(Q[l] for l in root_out) == 0.0, name="pcc_q0")

    bus_to_gen = {int(b): g for g, b in enumerate(case.gen_buses.tolist())}

    for i in range(nb):
        # Important fix: root KCL is represented by PCC constraints, and should not
        # be enforced again as a regular PQ bus KCL (otherwise P0 can be implicitly locked).
        if i == case.root:
            continue
        in_br = case.in_branches[i]
        out_br = case.out_branches[i]

        pg_i = Pg[bus_to_gen[i]] if i in bus_to_gen else 0.0
        qg_i = Qg[bus_to_gen[i]] if i in bus_to_gen else 0.0

        pinj = pg_i + float(pr[i]) - float(pd[i])
        qinj = qg_i + float(qr[i]) - float(qd[i])

        # KCL: incoming - outgoing + net_injection = 0
        m.addConstr(gp.quicksum(P[l] for l in in_br) - gp.quicksum(P[l] for l in out_br) + pinj == 0.0, name=f"kcl_p_{i}")
        m.addConstr(gp.quicksum(Q[l] for l in in_br) - gp.quicksum(Q[l] for l in out_br) + qinj == 0.0, name=f"kcl_q_{i}")

    for l in range(nl):
        i = int(case.from_bus[l])
        j = int(case.to_bus[l])
        m.addConstr(V[j] == V[i] - 2.0 * (float(case.r[l]) * P[l] + float(case.x[l]) * Q[l]), name=f"vd_{l}")
        m.addConstr(P[l] <= float(case.fmax_p[l]))
        m.addConstr(P[l] >= -float(case.fmax_p[l]))
        m.addConstr(Q[l] <= float(case.fmax_q[l]))
        m.addConstr(Q[l] >= -float(case.fmax_q[l]))

    m.setObjective(P0, GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return float("nan") if not return_detail else {"ok": False}

    if not return_detail:
        return float(P0.X)

    P_sol = np.array([P[l].X for l in range(nl)], dtype=float)
    Q_sol = np.array([Q[l].X for l in range(nl)], dtype=float)
    V_sol = np.array([V[i].X for i in range(nb)], dtype=float)
    Pg_sol = np.array([Pg[g].X for g in range(len(case.gen_buses))], dtype=float)
    Qg_sol = np.array([Qg[g].X for g in range(len(case.gen_buses))], dtype=float)
    return {
        "ok": True,
        "P0": float(P0.X),
        "P": P_sol,
        "Q": Q_sol,
        "V": V_sol,
        "Pg": Pg_sol,
        "Qg": Qg_sol,
    }


# =============================
# 3) Dataset generation
# =============================
def sample_trunc_normal(mu: float, sigma: float, lo: float = 0.0, hi: float = None) -> float:
    x = np.random.normal(mu, sigma)
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return float(x)


def sample_scenario_means(case: GridCase, rng: np.random.Generator):
    pd_mu = case.pd_base.copy()
    qd_mu = case.qd_base.copy()

    # load mean scaling by node around base
    for i in range(case.n_bus):
        if case.pd_base[i] > 1e-9:
            s = rng.uniform(0.75, 1.25)
            pd_mu[i] = case.pd_base[i] * s
            pf = case.pd_base[i] / max(case.qd_base[i], 1e-6)
            qd_mu[i] = pd_mu[i] / pf

    pr_mu = np.zeros(case.n_bus, dtype=float)
    for k, b in enumerate(case.pv_buses):
        pr_mu[b] = rng.uniform(0.15, 0.95) * case.pv_pmax[k]

    tan_pv = math.tan(math.acos(case.pv_pf))
    qr_mu = pr_mu * tan_pv

    return pd_mu, qd_mu, pr_mu, qr_mu


def make_feature_vector(case: GridCase, pd_mu: np.ndarray, pr_mu: np.ndarray) -> np.ndarray:
    # stage-1 base feature: all load-P means + PV-P means + 2 aggregate features
    feat = np.concatenate([
        pd_mu,
        pr_mu[case.pv_buses],
        np.array([pd_mu.sum(), pr_mu.sum()], dtype=float),
    ])
    return feat


def generate_dataset(case: GridCase, num_scenarios=NUM_SCENARIOS, mc_per_scenario=MC_PER_SCENARIO, seed=SEED_DATA):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    feats, yp0s, ypgs, yqgs = [], [], [], []
    dropped_scenarios = 0
    n_gen = len(case.gen_buses)

    for s in range(num_scenarios):
        pd_mu, qd_mu, pr_mu, qr_mu = sample_scenario_means(case, rng)
        x = make_feature_vector(case, pd_mu, pr_mu)

        std_pd = 0.10 * np.maximum(pd_mu, 1e-3)
        std_pr = 0.12 * np.maximum(pr_mu, 1e-3)

        y_p0_row = []
        y_pg_row = []
        y_qg_row = []
        for _ in range(mc_per_scenario):
            pd = pd_mu.copy()
            pr = pr_mu.copy()
            for i in range(case.n_bus):
                if pd_mu[i] > 1e-9:
                    pd[i] = sample_trunc_normal(pd_mu[i], std_pd[i], lo=0.0, hi=None)
            for k, b in enumerate(case.pv_buses):
                pr[b] = sample_trunc_normal(pr_mu[b], std_pr[b], lo=0.0, hi=float(case.pv_pmax[k]))

            qd = qd_mu * (pd / np.maximum(pd_mu, 1e-6))
            tan_pv = math.tan(math.acos(case.pv_pf))
            qr = pr * tan_pv

            sol = solve_pomax_gurobi_33bus(case, pd, qd, pr, qr, return_detail=True)
            if sol["ok"]:
                pg_sol = np.asarray(sol["Pg"], dtype=float).reshape(-1)
                qg_sol = np.asarray(sol["Qg"], dtype=float).reshape(-1)
                if pg_sol.size != n_gen or qg_sol.size != n_gen:
                    raise ValueError(f"OPF dispatch label shape mismatch: Pg={pg_sol.shape}, Qg={qg_sol.shape}, n_gen={n_gen}")
                y_p0_row.append(float(sol["P0"]))
                y_pg_row.append(pg_sol)
                y_qg_row.append(qg_sol)

        if len(y_p0_row) < max(5, mc_per_scenario // 8):
            # fallback-1: deterministic mean point, now also saving internal dispatch labels.
            sol_fb = solve_pomax_gurobi_33bus(case, pd_mu, qd_mu, pr_mu, qr_mu, return_detail=True)
            if sol_fb["ok"]:
                pg_fb = np.asarray(sol_fb["Pg"], dtype=float).reshape(-1)
                qg_fb = np.asarray(sol_fb["Qg"], dtype=float).reshape(-1)
                y_p0_row = [float(sol_fb["P0"]) for _ in range(mc_per_scenario)]
                y_pg_row = [pg_fb.copy() for _ in range(mc_per_scenario)]
                y_qg_row = [qg_fb.copy() for _ in range(mc_per_scenario)]
            else:
                # fallback-2: progressively relax the sampled operating point (loads down, PV clipped).
                for scale in [0.95, 0.90, 0.85, 0.80]:
                    pd_try = pd_mu * scale
                    qd_try = qd_mu * scale
                    pr_cap = np.zeros(case.n_bus, dtype=float)
                    pr_cap[case.pv_buses] = case.pv_pmax
                    pr_try = np.minimum(pr_mu, pr_cap) * scale
                    qr_try = pr_try * math.tan(math.acos(case.pv_pf))
                    sol_fb2 = solve_pomax_gurobi_33bus(case, pd_try, qd_try, pr_try, qr_try, return_detail=True)
                    if sol_fb2["ok"]:
                        pg_fb2 = np.asarray(sol_fb2["Pg"], dtype=float).reshape(-1)
                        qg_fb2 = np.asarray(sol_fb2["Qg"], dtype=float).reshape(-1)
                        y_p0_row = [float(sol_fb2["P0"]) for _ in range(mc_per_scenario)]
                        y_pg_row = [pg_fb2.copy() for _ in range(mc_per_scenario)]
                        y_qg_row = [qg_fb2.copy() for _ in range(mc_per_scenario)]
                        break

        if len(y_p0_row) > 0:
            if len(y_p0_row) < mc_per_scenario:
                fill_idx = rng.choice(len(y_p0_row), size=mc_per_scenario - len(y_p0_row), replace=True)
                y_p0_row += [y_p0_row[int(i)] for i in fill_idx]
                y_pg_row += [np.asarray(y_pg_row[int(i)], dtype=float).copy() for i in fill_idx]
                y_qg_row += [np.asarray(y_qg_row[int(i)], dtype=float).copy() for i in fill_idx]
            feats.append(x)
            yp0s.append(np.array(y_p0_row[:mc_per_scenario], dtype=float))
            ypgs.append(np.stack(y_pg_row[:mc_per_scenario], axis=0).astype(float))
            yqgs.append(np.stack(y_qg_row[:mc_per_scenario], axis=0).astype(float))
        else:
            dropped_scenarios += 1

        if (s + 1) % 20 == 0:
            print(f"已生成 {s+1}/{num_scenarios} 场景")

    if len(feats) == 0:
        raise RuntimeError(
            "数据集生成失败：所有场景均未获得可行 OPF 样本。"
            "请检查 Gurobi 可用性、网络参数（特别是Q约束/PCC_Q设定）或放宽场景随机范围。"
        )

    X = np.array(feats, dtype=float)
    YP0 = np.stack(yp0s, axis=0).astype(float)
    YPG = np.stack(ypgs, axis=0).astype(float)
    YQG = np.stack(yqgs, axis=0).astype(float)
    total_labels = num_scenarios * mc_per_scenario
    flat_samples = X.shape[0] * YP0.shape[1]
    print(f"[dataset] NUM_SCENARIOS={num_scenarios}, MC_PER_SCENARIO={mc_per_scenario}, total_labels={total_labels}")
    print(f"[dataset] feasible_scenarios={len(feats)}/{num_scenarios}, dropped={dropped_scenarios}")
    print(f"[dataset] X shape={X.shape}, YP0 shape={YP0.shape}, YPG shape={YPG.shape}, YQG shape={YQG.shape}")
    print(f"[dataset] flattened_train_samples={flat_samples}")
    print("[dataset] Pg label range:", np.nanmin(YPG), np.nanmax(YPG))
    print("[dataset] Qg label range:", np.nanmin(YQG), np.nanmax(YQG))
    return X, YP0, YPG, YQG


# =============================
# 4) Layer-A dispatch physics loss
# =============================
def recover_flows_from_pred_dispatch(
    case: GridCase,
    x_raw: torch.Tensor,
    mu_p0: torch.Tensor,
    pg_hat: torch.Tensor,
    qg_hat: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Recover radial branch flows and squared voltages from neural dispatch.

    Inputs:
    - x_raw: original (unstandardized) features [B, input_dim]
    - mu_p0: predicted Pomax GMM mean [B, 1] (kept for interface/diagnostics)
    - pg_hat/qg_hat: neural dispatch heads [B, n_gen]

    Sign convention:
    - Branch P/Q positive from from_bus -> to_bus.
    - Nodal pinj/qinj = generation + renewable - load.
    - For branch p -> j, P_pj = -sum subtree injections under j.
    """
    _ = mu_p0  # mu_p0 is used by the layer-A loss, not by the tree recovery itself.
    B = x_raw.shape[0]
    nb = case.n_bus
    nl = case.from_bus.size
    n_pv = len(case.pv_buses)
    device = x_raw.device
    dtype = x_raw.dtype

    pd = x_raw[:, :nb]
    pr_pv = x_raw[:, nb:nb + n_pv]
    pr = torch.zeros((B, nb), device=device, dtype=dtype)
    pv_buses_t = torch.tensor(case.pv_buses, device=device, dtype=torch.long)
    pr[:, pv_buses_t] = pr_pv

    tan_pv = math.tan(math.acos(case.pv_pf))
    qr = pr * tan_pv

    ratio_qp = np.divide(case.qd_base, np.maximum(case.pd_base, 1e-6))
    ratio_qp_t = torch.tensor(ratio_qp, device=device, dtype=dtype).view(1, -1)
    qd = pd * ratio_qp_t

    pg = pg_hat
    qg = qg_hat

    # Nodal net injection (generation + renewable - load).
    pinj = pr - pd
    qinj = qr - qd
    gen_buses_t = torch.tensor(case.gen_buses, device=device, dtype=torch.long)
    pinj[:, gen_buses_t] += pg
    qinj[:, gen_buses_t] += qg

    # Radial flow recovery by post-order subtree accumulation.
    P = torch.zeros((B, nl), device=device, dtype=dtype)
    Q = torch.zeros((B, nl), device=device, dtype=dtype)
    sub_p = pinj.clone()
    sub_q = qinj.clone()

    for j in case.rev_topo_order:
        if j == case.root:
            continue
        l = int(case.parent_branch[j])
        p = int(case.parent_bus[j])
        P[:, l] = -sub_p[:, j]
        Q[:, l] = -sub_q[:, j]
        sub_p[:, p] += sub_p[:, j]
        sub_q[:, p] += sub_q[:, j]

    # LinDistFlow voltage recovery by forward sweep from root V^2=1.0.
    V = torch.zeros((B, nb), device=device, dtype=dtype)
    V[:, case.root] = 1.0
    r_t = torch.tensor(case.r, device=device, dtype=dtype)
    x_t = torch.tensor(case.x, device=device, dtype=dtype)
    for j in case.topo_order:
        if j == case.root:
            continue
        l = int(case.parent_branch[j])
        p = int(case.parent_bus[j])
        V[:, j] = V[:, p] - 2.0 * (r_t[l] * P[:, l] + x_t[l] * Q[:, l])

    return {
        "pd": pd, "qd": qd, "pr": pr, "qr": qr,
        "pg": pg, "qg": qg,
        "pinj": pinj, "qinj": qinj,
        "P": P, "Q": Q, "V": V,
    }


def physics_loss_layerA(
    case: GridCase,
    x_raw: torch.Tensor,
    mu_p0: torch.Tensor,
    pg_hat: torch.Tensor,
    qg_hat: torch.Tensor,
    alpha_phys: float = 1.0,
    return_parts: bool = False,
):
    rec = recover_flows_from_pred_dispatch(case, x_raw, mu_p0, pg_hat, qg_hat)
    relu = torch.relu
    device = x_raw.device
    dtype = x_raw.dtype

    P = rec["P"]
    Q = rec["Q"]
    V = rec["V"]
    pg = rec["pg"]
    qg = rec["qg"]
    pr = rec["pr"]
    pinj = rec["pinj"]
    qinj = rec["qinj"]

    root_out = case.out_branches[case.root]
    pcc_out = P[:, root_out].sum(dim=1, keepdim=True)
    qcc_out = Q[:, root_out].sum(dim=1, keepdim=True)
    loss_pcc_p = ((pcc_out - mu_p0) ** 2).mean()
    loss_pcc_q = (qcc_out ** 2).mean()

    loss_global_p = ((mu_p0 + pinj.sum(dim=1, keepdim=True)) ** 2).mean()
    loss_global_q = ((qinj.sum(dim=1, keepdim=True)) ** 2).mean()

    vmin2, vmax2 = case.vmin ** 2, case.vmax ** 2
    loss_v = ((relu(vmin2 - V) ** 2) + (relu(V - vmax2) ** 2)).mean()

    fmax_p = torch.tensor(case.fmax_p, device=device, dtype=dtype).view(1, -1)
    fmax_q = torch.tensor(case.fmax_q, device=device, dtype=dtype).view(1, -1)
    loss_line_p = (relu(torch.abs(P) - fmax_p) ** 2).mean()
    loss_line_q = (relu(torch.abs(Q) - fmax_q) ** 2).mean()

    pg_min = torch.tensor(case.pg_min, device=device, dtype=dtype).view(1, -1)
    pg_max = torch.tensor(case.pg_max, device=device, dtype=dtype).view(1, -1)
    qg_min = torch.tensor(case.qg_min, device=device, dtype=dtype).view(1, -1)
    qg_max = torch.tensor(case.qg_max, device=device, dtype=dtype).view(1, -1)
    loss_pg = ((relu(pg_min - pg) ** 2) + (relu(pg - pg_max) ** 2)).mean()
    loss_qg = ((relu(qg_min - qg) ** 2) + (relu(qg - qg_max) ** 2)).mean()

    pv_idx = torch.tensor(case.pv_buses, device=device, dtype=torch.long)
    pv_max = torch.tensor(case.pv_pmax, device=device, dtype=dtype).view(1, -1)
    loss_pv = (relu(pr[:, pv_idx] - pv_max) ** 2).mean()

    # KCL residuals are mostly construction checks because flows are recovered from subtree injections.
    kcl_p_res = []
    kcl_q_res = []
    for i in range(case.n_bus):
        if i == case.root:
            continue
        in_br = case.in_branches[i]
        out_br = case.out_branches[i]
        kcl_p = P[:, in_br].sum(dim=1, keepdim=True) - P[:, out_br].sum(dim=1, keepdim=True) + pinj[:, i:i + 1]
        kcl_q = Q[:, in_br].sum(dim=1, keepdim=True) - Q[:, out_br].sum(dim=1, keepdim=True) + qinj[:, i:i + 1]
        kcl_p_res.append(kcl_p)
        kcl_q_res.append(kcl_q)
    kcl_p_res = torch.cat(kcl_p_res, dim=1)
    kcl_q_res = torch.cat(kcl_q_res, dim=1)
    loss_kcl = (kcl_p_res ** 2).mean() + (kcl_q_res ** 2).mean()

    kcl_weight = 0.1
    loss = (
        loss_pcc_p
        + loss_pcc_q
        + loss_global_p
        + loss_global_q
        + loss_v
        + loss_line_p
        + loss_line_q
        + loss_pg
        + loss_qg
        + loss_pv
        + kcl_weight * loss_kcl
    )
    loss = alpha_phys * loss

    if return_parts:
        parts = {
            "pcc_p": loss_pcc_p.detach(),
            "pcc_q": loss_pcc_q.detach(),
            "global_p": loss_global_p.detach(),
            "global_q": loss_global_q.detach(),
            "voltage": loss_v.detach(),
            "line_p": loss_line_p.detach(),
            "line_q": loss_line_q.detach(),
            "pg": loss_pg.detach(),
            "qg": loss_qg.detach(),
            "pv": loss_pv.detach(),
            "kcl": loss_kcl.detach(),
        }
        return loss, parts
    return loss


def dispatch_supervision_loss(
    case: GridCase,
    pg_hat: torch.Tensor,
    qg_hat: torch.Tensor,
    pg_label: torch.Tensor,
    qg_label: torch.Tensor,
    normalize: bool = True,
    return_parts: bool = False,
):
    """Supervise neural Pg/Qg dispatch heads against OPF internal dispatch labels."""
    device = pg_hat.device
    dtype = pg_hat.dtype

    if normalize:
        pg_min = torch.tensor(case.pg_min, device=device, dtype=dtype).view(1, -1)
        pg_max = torch.tensor(case.pg_max, device=device, dtype=dtype).view(1, -1)
        qg_min = torch.tensor(case.qg_min, device=device, dtype=dtype).view(1, -1)
        qg_max = torch.tensor(case.qg_max, device=device, dtype=dtype).view(1, -1)
        pg_scale = torch.clamp(pg_max - pg_min, min=1e-6)
        qg_scale = torch.clamp(qg_max - qg_min, min=1e-6)
        loss_pg_sup = (((pg_hat - pg_label.to(device=device, dtype=dtype)) / pg_scale) ** 2).mean()
        loss_qg_sup = (((qg_hat - qg_label.to(device=device, dtype=dtype)) / qg_scale) ** 2).mean()
    else:
        loss_pg_sup = ((pg_hat - pg_label.to(device=device, dtype=dtype)) ** 2).mean()
        loss_qg_sup = ((qg_hat - qg_label.to(device=device, dtype=dtype)) ** 2).mean()

    loss_dispatch_sup = LAM_PG_SUP * loss_pg_sup + LAM_QG_SUP * loss_qg_sup
    if return_parts:
        return loss_dispatch_sup, {"pg_sup": loss_pg_sup.detach(), "qg_sup": loss_qg_sup.detach()}
    return loss_dispatch_sup


# =============================
# 4.1) Physical diagnostics / sanity checks
# =============================
def _kcl_residuals_np(case: GridCase, P: np.ndarray, Q: np.ndarray, pinj: np.ndarray, qinj: np.ndarray):
    max_p, max_q = 0.0, 0.0
    for i in range(case.n_bus):
        if i == case.root:
            continue
        p_res = P[case.in_branches[i]].sum() - P[case.out_branches[i]].sum() + pinj[i]
        q_res = Q[case.in_branches[i]].sum() - Q[case.out_branches[i]].sum() + qinj[i]
        max_p = max(max_p, abs(float(p_res)))
        max_q = max(max_q, abs(float(q_res)))
    return max_p, max_q


def opf_sanity_check(case: GridCase, n_samples: int = 3, seed: int = 2026):
    print("\\n=== OPF sanity check ===")
    rng = np.random.default_rng(seed)
    for k in range(n_samples):
        pd_mu, qd_mu, pr_mu, qr_mu = sample_scenario_means(case, rng)
        sol = solve_pomax_gurobi_33bus(case, pd_mu, qd_mu, pr_mu, qr_mu, return_detail=True)
        if not sol["ok"]:
            print(f"[opf-check {k}] infeasible")
            continue
        pinj = pr_mu - pd_mu
        qinj = qr_mu - qd_mu
        for g, b in enumerate(case.gen_buses):
            pinj[b] += sol["Pg"][g]
            qinj[b] += sol["Qg"][g]
        pcc_lhs = sol["P"][case.out_branches[case.root]].sum()
        qcc_lhs = sol["Q"][case.out_branches[case.root]].sum()
        kcl_p_max, kcl_q_max = _kcl_residuals_np(case, sol["P"], sol["Q"], pinj, qinj)
        print(f"[opf-check {k}] P0={sol['P0']:.4f}, sumP_root_out={pcc_lhs:.4f}, PCC_err={pcc_lhs-sol['P0']:+.2e}, sumQ_root_out={qcc_lhs:+.2e}, maxKCL(P/Q)=({kcl_p_max:.2e},{kcl_q_max:.2e})")


def dispatch_label_sanity_check(case: GridCase, n_samples: int = 3, seed: int = 2027):
    print("\n=== OPF dispatch-label sanity check ===")
    rng = np.random.default_rng(seed)
    for k in range(n_samples):
        pd_mu, qd_mu, pr_mu, qr_mu = sample_scenario_means(case, rng)
        sol = solve_pomax_gurobi_33bus(case, pd_mu, qd_mu, pr_mu, qr_mu, return_detail=True)
        if not sol["ok"]:
            print(f"[dispatch-label-check {k}] infeasible")
            continue

        pinj = pr_mu - pd_mu
        qinj = qr_mu - qd_mu
        for g, b in enumerate(case.gen_buses):
            pinj[b] += sol["Pg"][g]
            qinj[b] += sol["Qg"][g]
        pcc_p = sol["P"][case.out_branches[case.root]].sum()
        pcc_q = sol["Q"][case.out_branches[case.root]].sum()
        kcl_p_max, kcl_q_max = _kcl_residuals_np(case, sol["P"], sol["Q"], pinj, qinj)
        pg_str = np.array2string(sol["Pg"], precision=4, suppress_small=True)
        qg_str = np.array2string(sol["Qg"], precision=4, suppress_small=True)
        print(
            f"[dispatch-label-check {k}] P0={sol['P0']:.4f}, "
            f"Pg={pg_str}, Qg={qg_str}, "
            f"sumPg={sol['Pg'].sum():.4f}, sumQg={sol['Qg'].sum():.4f}, "
            f"PCC(P/Q)=({pcc_p:.4f},{pcc_q:+.2e}), "
            f"maxKCL(P/Q)=({kcl_p_max:.2e},{kcl_q_max:.2e})"
        )


def layerA_dispatch_sanity_check(case: GridCase, n_samples: int = 3, seed: int = 7):
    print("\n=== Layer-A dispatch recovery sanity check ===")
    rng = np.random.default_rng(seed)
    pg_mid = 0.5 * (case.pg_min + case.pg_max)
    qg_mid = 0.5 * (case.qg_min + case.qg_max)
    for k in range(n_samples):
        pd_mu, _, pr_mu, _ = sample_scenario_means(case, rng)
        x = make_feature_vector(case, pd_mu, pr_mu).reshape(1, -1)
        mu_p0 = np.array([[max(0.05, pd_mu.sum() - pr_mu.sum() - pg_mid.sum())]], dtype=float)
        xt = torch.tensor(x, dtype=torch.float32)
        p0t = torch.tensor(mu_p0, dtype=torch.float32)
        pgt = torch.tensor(pg_mid.reshape(1, -1), dtype=torch.float32)
        qgt = torch.tensor(qg_mid.reshape(1, -1), dtype=torch.float32)
        rec = recover_flows_from_pred_dispatch(case, xt, p0t, pgt, qgt)
        phys, parts = physics_loss_layerA(case, xt, p0t, pgt, qgt, return_parts=True)

        total_pd = float(rec["pd"].sum())
        total_pr = float(rec["pr"].sum())
        total_pg = float(rec["pg"].sum())
        total_qd = float(rec["qd"].sum())
        total_qr = float(rec["qr"].sum())
        total_qg = float(rec["qg"].sum())
        p_root = float(rec["P"][:, case.out_branches[case.root]].sum())
        q_root = float(rec["Q"][:, case.out_branches[case.root]].sum())
        v_min = float(rec["V"].min())
        print(f"[layerA-check {k}] Pbal: muP0={mu_p0[0,0]:.4f}, Pg={total_pg:.4f}, Pd={total_pd:.4f}, Pr={total_pr:.4f}, root_out={p_root:.4f}")
        print(f"                 Qbal: Qg={total_qg:.4f}, Qd={total_qd:.4f}, Qr={total_qr:.4f}, rootQ={q_root:+.2e}, Vmin^2={v_min:.4f}, phys={float(phys):.4e}")
        print("                 parts: " + ", ".join(f"{name}={float(val):.2e}" for name, val in parts.items()))

    # monotonicity debug: higher load -> lower pomax tendency (heuristic check)
    pd_mu, qd_mu, pr_mu, qr_mu = sample_scenario_means(case, rng)
    y0 = solve_pomax_gurobi_33bus(case, pd_mu, qd_mu, pr_mu, qr_mu)
    y_hi_load = solve_pomax_gurobi_33bus(case, pd_mu * 1.1, qd_mu * 1.1, pr_mu, qr_mu)
    pv_cap = np.zeros(case.n_bus, dtype=float)
    pv_cap[case.pv_buses] = case.pv_pmax
    pr_hi = np.minimum(pr_mu * 1.2, pv_cap)
    qr_hi = pr_hi * math.tan(math.acos(case.pv_pf))
    y_hi_pv = solve_pomax_gurobi_33bus(case, pd_mu, qd_mu, pr_hi, qr_hi)
    print(f"[mono-check] base={y0:.4f}, high-load={y_hi_load:.4f}, high-pv={y_hi_pv:.4f} (expected: high-load <= base, high-pv >= base, not strict)")

# =============================
# 5) GMM2 utils
# =============================
def gmm2_log_prob(y: torch.Tensor, w: torch.Tensor, mu1: torch.Tensor, s1: torch.Tensor, mu2: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    z1 = (y - mu1) / s1
    z2 = (y - mu2) / s2
    logn1 = -0.5 * z1 ** 2 - torch.log(s1) - 0.5 * math.log(2 * math.pi)
    logn2 = -0.5 * z2 ** 2 - torch.log(s2) - 0.5 * math.log(2 * math.pi)
    logw = torch.log(w.clamp(min=1e-12))
    lp = torch.logsumexp(torch.cat([logw[:, 0:1] + logn1, logw[:, 1:2] + logn2], dim=1), dim=1, keepdim=True)
    return torch.clamp(lp, min=-1e6)


def gmm2_cdf(z: np.ndarray, w: np.ndarray, mu1: float, s1: float, mu2: float, s2: float) -> np.ndarray:
    return w[0] * norm.cdf((z - mu1) / (s1 + 1e-12)) + w[1] * norm.cdf((z - mu2) / (s2 + 1e-12))


# =============================
# 6) Bayesian network
# =============================
class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0, init_rho=-5.0):
        super().__init__()
        self.prior_sigma = float(prior_sigma)
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_rho = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        nn.init.constant_(self.w_rho, init_rho)
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.b_mu, -bound, bound)
        nn.init.constant_(self.b_rho, init_rho)

    @staticmethod
    def _sigma(rho):
        return torch.nn.functional.softplus(rho) + 1e-6

    def forward(self, x, sample=True):
        if sample:
            w = self.w_mu + self._sigma(self.w_rho) * torch.randn_like(self.w_mu)
            b = self.b_mu + self._sigma(self.b_rho) * torch.randn_like(self.b_mu)
        else:
            w, b = self.w_mu, self.b_mu
        return torch.nn.functional.linear(x, w, b)

    def kl_divergence(self):
        prior_var = self.prior_sigma ** 2
        w_sigma = self._sigma(self.w_rho)
        b_sigma = self._sigma(self.b_rho)
        kl_w = (math.log(self.prior_sigma) - torch.log(w_sigma)).sum() + 0.5 * ((w_sigma ** 2 + self.w_mu ** 2) / prior_var).sum() - 0.5 * self.w_mu.numel()
        kl_b = (math.log(self.prior_sigma) - torch.log(b_sigma)).sum() + 0.5 * ((b_sigma ** 2 + self.b_mu ** 2) / prior_var).sum() - 0.5 * self.b_mu.numel()
        return kl_w + kl_b


class BayesGMM2DispatchNet(nn.Module):
    def __init__(self, in_dim: int, case: GridCase, hidden=160, depth=3, prior_sigma=1.0, init_rho=-5.0):
        super().__init__()
        self.n_gen = len(case.gen_buses)
        self.linears = nn.ModuleList()
        d = in_dim
        for _ in range(depth):
            self.linears.append(BayesLinear(d, hidden, prior_sigma=prior_sigma, init_rho=init_rho))
            d = hidden
        self.out = BayesLinear(d, 6 + 2 * self.n_gen, prior_sigma=prior_sigma, init_rho=init_rho)
        self.act = nn.ReLU()

        # Register bounds as buffers so they move automatically with .to(device).
        self.register_buffer("pg_min_t", torch.tensor(case.pg_min, dtype=torch.float32).view(1, -1))
        self.register_buffer("pg_max_t", torch.tensor(case.pg_max, dtype=torch.float32).view(1, -1))
        self.register_buffer("qg_min_t", torch.tensor(case.qg_min, dtype=torch.float32).view(1, -1))
        self.register_buffer("qg_max_t", torch.tensor(case.qg_max, dtype=torch.float32).view(1, -1))

    def forward(self, x, sample=True):
        h = x
        for layer in self.linears:
            h = self.act(layer(h, sample=sample))
        out = self.out(h, sample=sample)

        w = torch.softmax(out[:, 0:2], dim=1)
        mu1, mu2 = out[:, 2:3], out[:, 4:5]
        s1 = torch.nn.functional.softplus(out[:, 3:4]) + 1e-3
        s2 = torch.nn.functional.softplus(out[:, 5:6]) + 1e-3

        pg_raw = out[:, 6:6 + self.n_gen]
        qg_raw = out[:, 6 + self.n_gen:6 + 2 * self.n_gen]
        pg_min = self.pg_min_t.to(device=x.device, dtype=x.dtype)
        pg_max = self.pg_max_t.to(device=x.device, dtype=x.dtype)
        qg_min = self.qg_min_t.to(device=x.device, dtype=x.dtype)
        qg_max = self.qg_max_t.to(device=x.device, dtype=x.dtype)
        pg_hat = pg_min + torch.sigmoid(pg_raw) * (pg_max - pg_min)
        qg_hat = qg_min + torch.sigmoid(qg_raw) * (qg_max - qg_min)

        return w, mu1, s1, mu2, s2, pg_hat, qg_hat

    def kl_divergence(self):
        return sum(l.kl_divergence() for l in self.linears) + self.out.kl_divergence()


# =============================
# 7) train
# =============================
def train_bayes_gmm2(case: GridCase, X: np.ndarray, YP0: np.ndarray, YPG: np.ndarray, YQG: np.ndarray):
    rng = np.random.default_rng(SEED_TRAIN)
    torch.manual_seed(SEED_TRAIN)
    np.random.seed(SEED_TRAIN)

    n_gen = len(case.gen_buses)
    if X.ndim != 2:
        raise ValueError(f"X shape must be (N, d), got {X.shape}")
    if YP0.ndim != 2:
        raise ValueError(f"YP0 shape must be (N, M), got {YP0.shape}")
    if YPG.ndim != 3:
        raise ValueError(f"YPG shape must be (N, M, n_gen), got {YPG.shape}")
    if YQG.ndim != 3:
        raise ValueError(f"YQG shape must be (N, M, n_gen), got {YQG.shape}")
    if X.shape[0] == 0 or YP0.shape[0] == 0:
        raise ValueError("空数据集：无法训练。请先检查数据生成可行性。")
    if not (X.shape[0] == YP0.shape[0] == YPG.shape[0] == YQG.shape[0]):
        raise ValueError(f"X/YP0/YPG/YQG 场景数不一致: X={X.shape}, YP0={YP0.shape}, YPG={YPG.shape}, YQG={YQG.shape}")
    if not (YP0.shape[1] == YPG.shape[1] == YQG.shape[1]):
        raise ValueError(f"YP0/YPG/YQG MC 维度不一致: YP0={YP0.shape}, YPG={YPG.shape}, YQG={YQG.shape}")
    if YPG.shape[2] != n_gen:
        raise ValueError(f"YPG last dim must equal n_gen={n_gen}, got {YPG.shape}")
    if YQG.shape[2] != n_gen:
        raise ValueError(f"YQG last dim must equal n_gen={n_gen}, got {YQG.shape}")

    n_scen = X.shape[0]
    n_val = max(1, int(round(VAL_RATIO * n_scen)))
    n_train = n_scen - n_val
    if n_train <= 0:
        raise ValueError(f"样本场景数过小，无法划分 train/val。X shape={X.shape}")

    scen_idx = rng.permutation(n_scen)
    tr_idx = scen_idx[:n_train]
    va_idx = scen_idx[n_train:]
    X_tr = X[tr_idx]
    YP0_tr = YP0[tr_idx]
    YPG_tr = YPG[tr_idx]
    YQG_tr = YQG[tr_idx]
    X_va = X[va_idx]
    YP0_va = YP0[va_idx]
    YPG_va = YPG[va_idx]
    YQG_va = YQG[va_idx]

    x_mean = X_tr.mean(axis=0, keepdims=True)
    x_std = X_tr.std(axis=0, keepdims=True) + 1e-9

    def flatten_xy_dispatch(x_raw: np.ndarray, yp0_raw: np.ndarray, ypg_raw: np.ndarray, yqg_raw: np.ndarray):
        xn_local = (x_raw - x_mean) / x_std
        _, m_local = yp0_raw.shape
        x_flat = np.repeat(xn_local, m_local, axis=0)
        yp0_flat = yp0_raw.reshape(-1, 1)
        ypg_flat = ypg_raw.reshape(-1, ypg_raw.shape[-1])
        yqg_flat = yqg_raw.reshape(-1, yqg_raw.shape[-1])
        mask = (
            np.isfinite(yp0_flat[:, 0])
            & np.all(np.isfinite(ypg_flat), axis=1)
            & np.all(np.isfinite(yqg_flat), axis=1)
        )
        return x_flat[mask], yp0_flat[mask], ypg_flat[mask], yqg_flat[mask]

    x_tr_flat, yp0_tr_flat, ypg_tr_flat, yqg_tr_flat = flatten_xy_dispatch(X_tr, YP0_tr, YPG_tr, YQG_tr)
    x_va_flat, yp0_va_flat, ypg_va_flat, yqg_va_flat = flatten_xy_dispatch(X_va, YP0_va, YPG_va, YQG_va)

    xt = torch.tensor(x_tr_flat, dtype=torch.float32, device=DEVICE)
    yp0_t = torch.tensor(yp0_tr_flat, dtype=torch.float32, device=DEVICE)
    ypg_t = torch.tensor(ypg_tr_flat, dtype=torch.float32, device=DEVICE)
    yqg_t = torch.tensor(yqg_tr_flat, dtype=torch.float32, device=DEVICE)
    xv = torch.tensor(x_va_flat, dtype=torch.float32, device=DEVICE)
    yp0_v = torch.tensor(yp0_va_flat, dtype=torch.float32, device=DEVICE)
    ypg_v = torch.tensor(ypg_va_flat, dtype=torch.float32, device=DEVICE)
    yqg_v = torch.tensor(yqg_va_flat, dtype=torch.float32, device=DEVICE)
    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=DEVICE)
    x_std_t = torch.tensor(x_std, dtype=torch.float32, device=DEVICE)

    net = BayesGMM2DispatchNet(in_dim=X.shape[1], case=case, prior_sigma=PRIOR_SIGMA, init_rho=INIT_RHO).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n_data = xt.shape[0]
    if n_data == 0:
        raise ValueError("训练集展平后样本数为 0。")

    eff_batch_size = BATCH_SIZE
    if DEVICE == "cuda":
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_mem_gb < 10.0 and eff_batch_size > 1024:
            eff_batch_size = 1024
            print(f"[train] GPU memory {total_mem_gb:.1f} GB，自动降 batch_size 到 {eff_batch_size}")

    n_batches = (n_data + eff_batch_size - 1) // eff_batch_size
    total_steps = EPOCHS * n_batches
    print("=== 开始训练 33-bus VI-BPINN (v5 layer-A dispatch physics + OPF dispatch supervision) ===")
    print(f"[split] train_scenarios={X_tr.shape[0]}, val_scenarios={X_va.shape[0]}")
    print(f"[split] train_flat={xt.shape[0]}, val_flat={xv.shape[0]}")
    print(f"[train] n_data={n_data}, batch_size={eff_batch_size}, n_batches_per_epoch={n_batches}, total_update_steps={total_steps}")
    print(f"[train] LAM_DISPATCH_SUP={LAM_DISPATCH_SUP}, LAM_PG_SUP={LAM_PG_SUP}, LAM_QG_SUP={LAM_QG_SUP}, NORMALIZE_DISPATCH_SUP={NORMALIZE_DISPATCH_SUP}")

    for ep in range(EPOCHS):
        perm = rng.permutation(n_data)
        beta = BETA_KL_MAX * min(1.0, (ep + 1) / max(1, KL_WARMUP_EPOCHS))
        ep_loss, ep_nll, ep_disp_sup, ep_phys, ep_kl = 0.0, 0.0, 0.0, 0.0, 0.0

        for b in range(n_batches):
            b0 = b * eff_batch_size
            b1 = min((b + 1) * eff_batch_size, n_data)
            idx = perm[b0:b1]
            xb = xt[idx]
            yb_p0 = yp0_t[idx]
            yb_pg = ypg_t[idx]
            yb_qg = yqg_t[idx]

            nll_acc = 0.0
            disp_sup_acc = 0.0
            phys_acc = 0.0
            for _ in range(TRAIN_WEIGHT_SAMPLES):
                w, mu1, s1, mu2, s2, pg_hat, qg_hat = net(xb, sample=True)
                nll = (-gmm2_log_prob(yb_p0, w, mu1, s1, mu2, s2)).mean()
                mu_mix = w[:, 0:1] * mu1 + w[:, 1:2] * mu2
                x_raw = xb * x_std_t + x_mean_t
                phys = physics_loss_layerA(case, x_raw, mu_mix, pg_hat, qg_hat)
                dispatch_sup = dispatch_supervision_loss(
                    case,
                    pg_hat,
                    qg_hat,
                    yb_pg,
                    yb_qg,
                    normalize=NORMALIZE_DISPATCH_SUP,
                )
                nll_acc += nll
                disp_sup_acc += dispatch_sup
                phys_acc += phys

            nll = nll_acc / TRAIN_WEIGHT_SAMPLES
            dispatch_sup = disp_sup_acc / TRAIN_WEIGHT_SAMPLES
            phys = phys_acc / TRAIN_WEIGHT_SAMPLES
            kl = net.kl_divergence()
            loss = nll + LAM_DISPATCH_SUP * dispatch_sup + LAM_PHYS * phys + beta * kl / n_data

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            ep_loss += float(loss.detach().cpu())
            ep_nll += float(nll.detach().cpu())
            ep_disp_sup += float(dispatch_sup.detach().cpu())
            ep_phys += float(phys.detach().cpu())
            ep_kl += float((kl / n_data).detach().cpu())

        if ((ep + 1) % LOG_EVERY == 0) or (ep == 0) or (ep + 1 == EPOCHS):
            msg = (
                f"Epoch {ep+1:4d} | avg_loss={ep_loss/n_batches:.6f} "
                f"avg_nll={ep_nll/n_batches:.6f} avg_disp_sup={ep_disp_sup/n_batches:.6f} "
                f"avg_phys={ep_phys/n_batches:.6f} avg_kl/N={ep_kl/n_batches:.6f} "
                f"beta={beta:.4f} n_batches={n_batches}"
            )
            if xv.shape[0] > 0:
                with torch.no_grad():
                    wv, mu1v, s1v, mu2v, s2v, pg_pred_v, qg_pred_v = net(xv, sample=False)
                    val_nll = (-gmm2_log_prob(yp0_v, wv, mu1v, s1v, mu2v, s2v)).mean()
                    val_disp_sup, val_disp_parts = dispatch_supervision_loss(
                        case,
                        pg_pred_v,
                        qg_pred_v,
                        ypg_v,
                        yqg_v,
                        normalize=NORMALIZE_DISPATCH_SUP,
                        return_parts=True,
                    )
                    mu_mix_v = wv[:, 0:1] * mu1v + wv[:, 1:2] * mu2v
                    x_raw_v = xv * x_std_t + x_mean_t
                    val_phys, val_parts = physics_loss_layerA(case, x_raw_v, mu_mix_v, pg_pred_v, qg_pred_v, return_parts=True)
                    kl_now = net.kl_divergence()
                    val_total = val_nll + LAM_DISPATCH_SUP * val_disp_sup + LAM_PHYS * val_phys + beta * kl_now / n_data
                msg += (
                    f" | val_loss={val_total.item():.6f} "
                    f"val_nll={val_nll.item():.6f} val_disp_sup={val_disp_sup.item():.6f} "
                    f"val_pg_sup={val_disp_parts['pg_sup'].item():.3e} "
                    f"val_qg_sup={val_disp_parts['qg_sup'].item():.3e} "
                    f"val_phys={val_phys.item():.6f} "
                    f"val_pcc_p={val_parts['pcc_p'].item():.3e} "
                    f"val_global_p={val_parts['global_p'].item():.3e} "
                    f"val_voltage={val_parts['voltage'].item():.3e} "
                    f"val_line_p={val_parts['line_p'].item():.3e} "
                    f"val_pg={val_parts['pg'].item():.3e} "
                    f"val_qg={val_parts['qg'].item():.3e}"
                )
            print(msg)

    return net, x_mean, x_std


# =============================
# 8) eval + plot
# =============================
def draw_test_scenario(case: GridCase, seed=123):
    rng = np.random.default_rng(seed)
    return sample_scenario_means(case, rng)


def eval_and_plot(case: GridCase, net: BayesGMM2DispatchNet, x_mean: np.ndarray, x_std: np.ndarray, mc_eval=2500):
    pd_mu, qd_mu, pr_mu, qr_mu = draw_test_scenario(case, seed=SEED_EVAL)
    x = make_feature_vector(case, pd_mu, pr_mu)

    y_list = []
    np.random.seed(SEED_EVAL)
    std_pd = 0.10 * np.maximum(pd_mu, 1e-3)
    std_pr = 0.12 * np.maximum(pr_mu, 1e-3)

    for _ in range(mc_eval):
        pd = pd_mu.copy()
        pr = pr_mu.copy()
        for i in range(case.n_bus):
            if pd_mu[i] > 1e-9:
                pd[i] = sample_trunc_normal(pd_mu[i], std_pd[i], lo=0.0, hi=None)
        for k, b in enumerate(case.pv_buses):
            pr[b] = sample_trunc_normal(pr_mu[b], std_pr[b], lo=0.0, hi=float(case.pv_pmax[k]))
        qd = qd_mu * (pd / np.maximum(pd_mu, 1e-6))
        qr = pr * math.tan(math.acos(case.pv_pf))
        y = solve_pomax_gurobi_33bus(case, pd, qd, pr, qr)
        if np.isfinite(y):
            y_list.append(y)

    if len(y_list) == 0:
        raise RuntimeError("评估场景下 MC baseline 全部不可行，无法绘制 CDF。请检查 case 参数与约束。")

    y_mc = np.array(y_list, dtype=float)
    y_mc.sort()
    z_min, z_max = float(y_mc.min()), float(y_mc.max())
    z_grid = np.linspace(z_min - 0.2, z_max + 0.2, 600)
    cdf_mc = np.searchsorted(y_mc, z_grid, side="right") / y_mc.size

    x_n = (x.reshape(1, -1) - x_mean) / x_std
    xt = torch.tensor(x_n, dtype=torch.float32, device=DEVICE)

    net.eval()
    cdf_samps, q05_samps = [], []
    with torch.no_grad():
        for _ in range(EVAL_THETA_SAMPLES):
            w_t, mu1_t, s1_t, mu2_t, s2_t, _, _ = net(xt, sample=True)
            w = w_t.cpu().numpy().reshape(-1)
            mu1 = float(mu1_t.cpu().numpy().reshape(-1)[0])
            mu2 = float(mu2_t.cpu().numpy().reshape(-1)[0])
            s1 = float(s1_t.cpu().numpy().reshape(-1)[0])
            s2 = float(s2_t.cpu().numpy().reshape(-1)[0])
            cdf = gmm2_cdf(z_grid, w, mu1, s1, mu2, s2)
            cdf_samps.append(cdf)

            def f_root(z):
                return float(gmm2_cdf(np.array([z]), w, mu1, s1, mu2, s2)[0] - 0.05)
            try:
                q05_samps.append(float(brentq(f_root, z_min - 5.0, z_max + 5.0)))
            except Exception:
                q05_samps.append(float("nan"))

    cdf_samps = np.array(cdf_samps)
    cdf_mean = np.nanmean(cdf_samps, axis=0)
    cdf_std = np.nanstd(cdf_samps, axis=0)
    cdf_lo = np.clip(cdf_mean - 2.0 * cdf_std, 0.0, 1.0)
    cdf_hi = np.clip(cdf_mean + 2.0 * cdf_std, 0.0, 1.0)

    arms = 100.0 * math.sqrt(np.mean((cdf_mean - cdf_mc) ** 2))
    q05_mc = float(np.quantile(y_mc, 0.05))
    q05_arr = np.array(q05_samps, dtype=float)
    q05_arr = q05_arr[np.isfinite(q05_arr)]
    q05_mean = float(np.mean(q05_arr)) if q05_arr.size > 0 else float("nan")

    print(f"[33bus B-PINN] ARMS(CDF) = {arms:.4f}%")
    print(f"Quantile@0.05: MC={q05_mc:.4f}, BNN mean={q05_mean:.4f}")

    plt.figure(figsize=(10, 5.5), dpi=130)
    plt.fill_between(z_grid, cdf_lo, cdf_hi, alpha=0.25, label="B-PINN mean ± 2 std")
    for k in np.linspace(0, max(0, len(cdf_samps) - 1), min(15, len(cdf_samps)), dtype=int):
        plt.plot(z_grid, cdf_samps[k], color="0.5", alpha=0.12, linewidth=0.8)
    plt.plot(z_grid, cdf_mc, "k", linewidth=2.5, label="MC baseline CDF")
    plt.plot(z_grid, cdf_mean, "--", color="#d95f02", linewidth=2.5, label=f"B-PINN mean (ARMS={arms:.3f}%)")
    plt.axhline(0.05, linestyle=":", color="k", linewidth=1.8, alpha=0.6)
    plt.scatter([q05_mc], [0.05], color="k", s=50)
    if np.isfinite(q05_mean):
        plt.scatter([q05_mean], [0.05], color="#d95f02", s=60, marker="x")
    plt.xlabel("P0 max (MW)")
    plt.ylabel("CDF")
    plt.ylim(-0.02, 1.02)
    plt.title("IEEE 33-bus Pomax CDF — v5 layer-A dispatch supervision")
    plt.legend(loc="upper left")
    plt.grid(False)
    plt.tight_layout()
    out_png = "Pomax_CDF_33bus_v5_layerA_dispatch_supervised.png"
    plt.savefig(out_png, dpi=280, bbox_inches="tight")
    plt.show()
    return arms


def eval_multiple_test_scenarios(
    case: GridCase,
    net: BayesGMM2DispatchNet,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    n_scenarios: int = N_TEST_SCENARIOS,
    mc_eval_multi: int = MC_EVAL_MULTI,
):
    rng = np.random.default_rng(SEED_EVAL + 1000)
    arms_list = []
    net.eval()

    for s in range(n_scenarios):
        pd_mu, qd_mu, pr_mu, qr_mu = sample_scenario_means(case, rng)
        x = make_feature_vector(case, pd_mu, pr_mu)

        y_list = []
        std_pd = 0.10 * np.maximum(pd_mu, 1e-3)
        std_pr = 0.12 * np.maximum(pr_mu, 1e-3)
        for _ in range(mc_eval_multi):
            pd = pd_mu.copy()
            pr = pr_mu.copy()
            for i in range(case.n_bus):
                if pd_mu[i] > 1e-9:
                    pd[i] = sample_trunc_normal(pd_mu[i], std_pd[i], lo=0.0, hi=None)
            for k, b in enumerate(case.pv_buses):
                pr[b] = sample_trunc_normal(pr_mu[b], std_pr[b], lo=0.0, hi=float(case.pv_pmax[k]))
            qd = qd_mu * (pd / np.maximum(pd_mu, 1e-6))
            qr = pr * math.tan(math.acos(case.pv_pf))
            y = solve_pomax_gurobi_33bus(case, pd, qd, pr, qr)
            if np.isfinite(y):
                y_list.append(y)

        if len(y_list) < 30:
            print(f"[multi-eval] 场景{s+1}/{n_scenarios} 可行 MC 过少，跳过。")
            continue

        y_mc = np.array(y_list, dtype=float)
        y_mc.sort()
        z_min, z_max = float(y_mc.min()), float(y_mc.max())
        z_grid = np.linspace(z_min - 0.2, z_max + 0.2, 500)
        cdf_mc = np.searchsorted(y_mc, z_grid, side="right") / y_mc.size

        x_n = (x.reshape(1, -1) - x_mean) / x_std
        xt = torch.tensor(x_n, dtype=torch.float32, device=DEVICE)
        cdf_samps = []
        with torch.no_grad():
            for _ in range(EVAL_THETA_SAMPLES):
                w_t, mu1_t, s1_t, mu2_t, s2_t, _, _ = net(xt, sample=True)
                w = w_t.cpu().numpy().reshape(-1)
                mu1 = float(mu1_t.cpu().numpy().reshape(-1)[0])
                mu2 = float(mu2_t.cpu().numpy().reshape(-1)[0])
                s1 = float(s1_t.cpu().numpy().reshape(-1)[0])
                s2 = float(s2_t.cpu().numpy().reshape(-1)[0])
                cdf_samps.append(gmm2_cdf(z_grid, w, mu1, s1, mu2, s2))

        cdf_mean = np.mean(np.array(cdf_samps), axis=0)
        arms = 100.0 * math.sqrt(np.mean((cdf_mean - cdf_mc) ** 2))
        arms_list.append(arms)
        print(f"[multi-eval] scenario={s+1:02d}/{n_scenarios}, ARMS={arms:.4f}%")

    if len(arms_list) == 0:
        print("[multi-eval] 无可用场景用于统计 ARMS。")
        return

    arms_arr = np.array(arms_list, dtype=float)
    print("[multi-eval] ARMS summary over test scenarios:")
    print(f"  mean   = {arms_arr.mean():.4f}%")
    print(f"  median = {np.median(arms_arr):.4f}%")
    print(f"  min    = {arms_arr.min():.4f}%")
    print(f"  max    = {arms_arr.max():.4f}%")
    print(f"  q90    = {np.quantile(arms_arr, 0.90):.4f}%")


def main():
    case = build_ieee33_case()
    if RUN_SANITY_CHECKS:
        opf_sanity_check(case, n_samples=3, seed=SEED_DATA + 11)
        layerA_dispatch_sanity_check(case, n_samples=3, seed=SEED_DATA + 12)
        dispatch_label_sanity_check(case, n_samples=3, seed=SEED_DATA + 13)
    print("生成 33 节点训练数据...")
    print(f"[config] FAST_MODE={FAST_MODE}")
    print(f"[config] NUM_SCENARIOS={NUM_SCENARIOS}, MC_PER_SCENARIO={MC_PER_SCENARIO}, total_labels={NUM_SCENARIOS * MC_PER_SCENARIO}")
    X, YP0, YPG, YQG = generate_dataset(case, NUM_SCENARIOS, MC_PER_SCENARIO, seed=SEED_DATA)
    print(f"Dataset: X={X.shape}, YP0={YP0.shape}, YPG={YPG.shape}, YQG={YQG.shape}")
    print(f"Flattened samples before split = {X.shape[0] * YP0.shape[1]}")

    net, x_mean, x_std = train_bayes_gmm2(case, X, YP0, YPG, YQG)
    eval_and_plot(case, net, x_mean, x_std, mc_eval=2500)
    if RUN_MULTI_TEST:
        eval_multiple_test_scenarios(
            case,
            net,
            x_mean,
            x_std,
            n_scenarios=N_TEST_SCENARIOS,
            mc_eval_multi=MC_EVAL_MULTI,
        )


if __name__ == "__main__":
    main()
