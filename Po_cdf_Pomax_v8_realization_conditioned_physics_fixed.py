# -*- coding: utf-8 -*-
"""
v8 realization-conditioned physics:
IEEE 33-bus single-period VI-BPINN with:
- Bayesian GMM-2 output for P0max conditional distribution: x_mu -> p(P0 | x_mu)
- realization-conditioned dispatch recovery: (x_mu, omega, P0) -> Pg/Qg
- physics loss evaluated on actual MC source-load realization omega, not on scenario mean x_mu
- empirical quantile diagnostic based on matched MC realization
- active constraint pattern diagnostics inherited from v7
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
FIXED_QUANTILE_TAUS = [0.05, 0.25, 0.50, 0.75, 0.95]
QUANTILE_TAUS = FIXED_QUANTILE_TAUS  # backward compatibility
USE_RANDOM_PHYS_TAUS = True
N_RANDOM_PHYS_TAUS = 4
RANDOM_TAU_LOW = 0.02
RANDOM_TAU_HIGH = 0.98
LAM_PHYS_REALIZATION = 0.08
LAM_PHYS_EMP_QUANTILE = 0.04
USE_EMPIRICAL_QUANTILE_PHYS = True
EMP_QUANTILE_TAUS = [0.05, 0.25, 0.50, 0.75, 0.95]
USE_RANDOM_EMP_QUANTILE_TAUS = True
N_RANDOM_EMP_QUANTILE_TAUS = 4
TAIL_WEIGHTED_PHYS = False
DETACH_QUANTILE_Z = False
ACTIVE_TOL_PG = 1e-4
ACTIVE_TOL_QG = 1e-4
ACTIVE_TOL_V = 1e-4
ACTIVE_TOL_LINE = 1e-4
ACTIVE_TOP_K = 10
SAVE_ACTIVE_PATTERN_CSV = True

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


def get_active_constraint_signature(
    case,
    sol,
    tol_pg=ACTIVE_TOL_PG,
    tol_qg=ACTIVE_TOL_QG,
    tol_v=ACTIVE_TOL_V,
    tol_line=ACTIVE_TOL_LINE,
):
    pg = np.asarray(sol["Pg"], dtype=float)
    qg = np.asarray(sol["Qg"], dtype=float)
    v = np.asarray(sol["V"], dtype=float)
    p = np.asarray(sol["P"], dtype=float)
    q = np.asarray(sol["Q"], dtype=float)
    all_names, bits = [], []
    for g in range(len(case.gen_buses)):
        all_names.append(f"Pg_min_g{g}")
        bits.append(1 if pg[g] <= case.pg_min[g] + tol_pg else 0)
    for g in range(len(case.gen_buses)):
        all_names.append(f"Pg_max_g{g}")
        bits.append(1 if pg[g] >= case.pg_max[g] - tol_pg else 0)
    for g in range(len(case.gen_buses)):
        all_names.append(f"Qg_min_g{g}")
        bits.append(1 if qg[g] <= case.qg_min[g] + tol_qg else 0)
    for g in range(len(case.gen_buses)):
        all_names.append(f"Qg_max_g{g}")
        bits.append(1 if qg[g] >= case.qg_max[g] - tol_qg else 0)
    vmin2, vmax2 = case.vmin ** 2, case.vmax ** 2
    for i in range(case.n_bus):
        all_names.append(f"Vmin_bus{i+1:02d}")
        bits.append(1 if v[i] <= vmin2 + tol_v else 0)
    for i in range(case.n_bus):
        all_names.append(f"Vmax_bus{i+1:02d}")
        bits.append(1 if v[i] >= vmax2 - tol_v else 0)
    for l in range(case.from_bus.size):
        all_names.append(f"Pline_pos_l{l:02d}")
        bits.append(1 if p[l] >= case.fmax_p[l] - tol_line else 0)
    for l in range(case.from_bus.size):
        all_names.append(f"Pline_neg_l{l:02d}")
        bits.append(1 if p[l] <= -case.fmax_p[l] + tol_line else 0)
    for l in range(case.from_bus.size):
        all_names.append(f"Qline_pos_l{l:02d}")
        bits.append(1 if q[l] >= case.fmax_q[l] - tol_line else 0)
    for l in range(case.from_bus.size):
        all_names.append(f"Qline_neg_l{l:02d}")
        bits.append(1 if q[l] <= -case.fmax_q[l] + tol_line else 0)
    signature_tuple = tuple(int(b) for b in bits)
    active_names = [n for n, b in zip(all_names, bits) if b == 1]
    return signature_tuple, active_names, all_names


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
    xmu_list, xreal_list, yp0s, ypgs, yqgs = [], [], [], [], []
    active_records = []
    all_active_names = None
    n_gen = len(case.gen_buses)
    dropped_scenarios = 0

    for sidx in range(num_scenarios):
        pd_mu, qd_mu, pr_mu, qr_mu = sample_scenario_means(case, rng)
        x_mu = make_feature_vector(case, pd_mu, pr_mu)
        y_p0_row, y_pg_row, y_qg_row, x_real_row, active_row = [], [], [], [], []
        std_pd = 0.10 * np.maximum(pd_mu, 1e-3)
        std_pr = 0.12 * np.maximum(pr_mu, 1e-3)

        for _ in range(mc_per_scenario):
            pd = pd_mu.copy(); pr = pr_mu.copy()
            for i in range(case.n_bus):
                if pd_mu[i] > 1e-9:
                    pd[i] = sample_trunc_normal(pd_mu[i], std_pd[i], lo=0.0, hi=None)
            for k, b in enumerate(case.pv_buses):
                pr[b] = sample_trunc_normal(pr_mu[b], std_pr[b], lo=0.0, hi=float(case.pv_pmax[k]))
            qd = qd_mu * (pd / np.maximum(pd_mu, 1e-6))
            qr = pr * math.tan(math.acos(case.pv_pf))
            sol = solve_pomax_gurobi_33bus(case, pd, qd, pr, qr, return_detail=True)
            if sol["ok"]:
                y_p0_row.append(float(sol["P0"]))
                y_pg_row.append(np.asarray(sol["Pg"], dtype=float).reshape(-1))
                y_qg_row.append(np.asarray(sol["Qg"], dtype=float).reshape(-1))
                x_real_row.append(make_feature_vector(case, pd, pr))
                sig, an, alln = get_active_constraint_signature(case, sol)
                all_active_names = alln
                active_row.append({"signature": sig, "active_names": an})

        if len(y_p0_row) == 0:
            sol_fb = solve_pomax_gurobi_33bus(case, pd_mu, qd_mu, pr_mu, qr_mu, return_detail=True)
            if sol_fb["ok"]:
                xr_fb = make_feature_vector(case, pd_mu, pr_mu)
                sig, an, alln = get_active_constraint_signature(case, sol_fb)
                all_active_names = alln
                y_p0_row = [float(sol_fb["P0"]) for _ in range(mc_per_scenario)]
                y_pg_row = [np.asarray(sol_fb["Pg"], dtype=float).copy() for _ in range(mc_per_scenario)]
                y_qg_row = [np.asarray(sol_fb["Qg"], dtype=float).copy() for _ in range(mc_per_scenario)]
                x_real_row = [xr_fb.copy() for _ in range(mc_per_scenario)]
                active_row = [{"signature": sig, "active_names": list(an)} for _ in range(mc_per_scenario)]
            else:
                for _ in range(5):
                    pd_try = pd_mu * rng.uniform(0.85, 1.15, size=case.n_bus)
                    pr_try = pr_mu.copy()
                    for k, b in enumerate(case.pv_buses):
                        pr_try[b] = np.clip(pr_try[b] * rng.uniform(0.85, 1.15), 0.0, case.pv_pmax[k])
                    qd_try = qd_mu * (pd_try / np.maximum(pd_mu, 1e-6))
                    qr_try = pr_try * math.tan(math.acos(case.pv_pf))
                    sol_fb2 = solve_pomax_gurobi_33bus(case, pd_try, qd_try, pr_try, qr_try, return_detail=True)
                    if sol_fb2["ok"]:
                        xr_fb2 = make_feature_vector(case, pd_try, pr_try)
                        sig, an, alln = get_active_constraint_signature(case, sol_fb2)
                        all_active_names = alln
                        y_p0_row = [float(sol_fb2["P0"]) for _ in range(mc_per_scenario)]
                        y_pg_row = [np.asarray(sol_fb2["Pg"], dtype=float).copy() for _ in range(mc_per_scenario)]
                        y_qg_row = [np.asarray(sol_fb2["Qg"], dtype=float).copy() for _ in range(mc_per_scenario)]
                        x_real_row = [xr_fb2.copy() for _ in range(mc_per_scenario)]
                        active_row = [{"signature": sig, "active_names": list(an)} for _ in range(mc_per_scenario)]
                        break
                if len(y_p0_row) == 0:
                    dropped_scenarios += 1
                    continue
        if len(y_p0_row) < mc_per_scenario:
            fill_idx = rng.choice(len(y_p0_row), size=mc_per_scenario-len(y_p0_row), replace=True)
            y_p0_row += [y_p0_row[int(i)] for i in fill_idx]
            y_pg_row += [y_pg_row[int(i)].copy() for i in fill_idx]
            y_qg_row += [y_qg_row[int(i)].copy() for i in fill_idx]
            x_real_row += [x_real_row[int(i)].copy() for i in fill_idx]
            active_row += [{"signature": active_row[int(i)]["signature"], "active_names": list(active_row[int(i)]["active_names"])} for i in fill_idx]

        xmu_list.append(x_mu)
        xreal_list.append(np.stack(x_real_row[:mc_per_scenario], axis=0).astype(float))
        yp0s.append(np.array(y_p0_row[:mc_per_scenario], dtype=float))
        ypgs.append(np.stack(y_pg_row[:mc_per_scenario], axis=0).astype(float))
        yqgs.append(np.stack(y_qg_row[:mc_per_scenario], axis=0).astype(float))
        active_records.extend(active_row[:mc_per_scenario])

    XMU = np.array(xmu_list, dtype=float)
    XREAL = np.stack(xreal_list, axis=0).astype(float)
    YP0 = np.stack(yp0s, axis=0).astype(float)
    YPG = np.stack(ypgs, axis=0).astype(float)
    YQG = np.stack(yqgs, axis=0).astype(float)
    print(f"[dataset] XMU shape={XMU.shape}, XREAL shape={XREAL.shape}, YP0 shape={YP0.shape}, YPG shape={YPG.shape}, YQG shape={YQG.shape}")
    flattened_samples = XREAL.shape[0] * XREAL.shape[1]
    active_match = (len(active_records) == flattened_samples)
    print(f"[dataset] flattened samples = {flattened_samples}")
    print(f"[dataset] active_records = {len(active_records)}")
    print(f"[dataset] active_records_match = {active_match}")
    if not active_match:
        print("[dataset] WARNING: active_records count mismatch with flattened labels.")
    print("[dataset] Pg label range:", np.nanmin(YPG), np.nanmax(YPG))
    print("[dataset] Qg label range:", np.nanmin(YQG), np.nanmax(YQG))
    return XMU, XREAL, YP0, YPG, YQG, active_records, all_active_names


def summarize_active_patterns(active_records, all_names, top_k=ACTIVE_TOP_K):
    from collections import Counter
    if not active_records:
        print("[active-pattern] WARNING: active_records is empty.")
        return
    n = len(active_records)
    counter = Counter([r["signature"] for r in active_records])
    print(f"[active-pattern] total_samples={n}, unique_patterns={len(counter)}")
    top = counter.most_common(top_k)
    rows_pat = []
    for i, (sig, cnt) in enumerate(top, start=1):
        active_names = [all_names[j] for j, b in enumerate(sig) if b == 1] if all_names else []
        rows_pat.append((i, cnt, 100.0 * cnt / n, ";".join(active_names)))
        print(f"[active-pattern] top{i:02d}: count={cnt}, pct={100.0*cnt/n:.2f}%, active={active_names}")
    single = np.zeros(len(all_names or []), dtype=int)
    for r in active_records:
        single += np.array(r["signature"], dtype=int)
    ord_idx = np.argsort(-single)[:top_k]
    rows_single = []
    for j in ord_idx:
        rows_single.append((all_names[j], int(single[j]), float(single[j] / n)))
        print(f"[active-single] {all_names[j]}: count={int(single[j])}, active_rate={float(single[j] / n):.6f}")
    for pref in ["Pg_min_", "Pg_max_", "Qg_min_", "Qg_max_", "Vmin_", "Vmax_", "Pline_pos_", "Pline_neg_", "Qline_pos_", "Qline_neg_"]:
        idx = [i for i, nm in enumerate(all_names or []) if nm.startswith(pref)]
        if idx:
            c = int(single[idx].sum())
            denom = len(idx) * n
            print(f"[active-group] {pref}: count={c}, rate={c/max(1,denom):.6f}")
    import csv
    if SAVE_ACTIVE_PATTERN_CSV:
        with open("active_constraint_patterns_v8.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["pattern_id", "count", "percentage", "active_names"])
            for r in rows_pat:
                w.writerow([r[0], r[1], f"{r[2]:.6f}", r[3]])
        with open("active_constraint_single_rates_v8.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["constraint_name", "count", "active_rate"])
            for r in rows_single:
                w.writerow([r[0], r[1], f"{r[2]:.8f}"])



# =============================
# 4) Layer-A dispatch physics loss
# =============================
def recover_flows_from_dispatch_batch(
    case: GridCase,
    x_raw: torch.Tensor,
    z_p0: torch.Tensor,
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
    _ = z_p0
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


def _gmm_quantile_weights(device, dtype, K=None):
    if K is None:
        K = len(FIXED_QUANTILE_TAUS)
    if TAIL_WEIGHTED_PHYS and K == len(FIXED_QUANTILE_TAUS):
        w = torch.tensor([0.30, 0.15, 0.10, 0.15, 0.30], device=device, dtype=dtype)
    else:
        w = torch.full((K,), 1.0 / K, device=device, dtype=dtype)
    return w / w.sum()


def physics_loss_quantile(
    case: GridCase,
    x_raw: torch.Tensor,
    z_quantiles: torch.Tensor,
    pg_quantiles: torch.Tensor,
    qg_quantiles: torch.Tensor,
    tau_weights: torch.Tensor = None,
    alpha_phys: float = 1.0,
    return_parts: bool = False,
):
    B, K = z_quantiles.shape
    x_rep = x_raw.unsqueeze(1).repeat(1, K, 1).reshape(B * K, -1)
    z_flat = z_quantiles.reshape(B * K, 1)
    pg_flat = pg_quantiles.reshape(B * K, -1)
    qg_flat = qg_quantiles.reshape(B * K, -1)
    rec = recover_flows_from_dispatch_batch(case, x_rep, z_flat, pg_flat, q_flat := qg_flat)
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
    loss_pcc_p_all = ((pcc_out - z_flat) ** 2).reshape(B, K)
    loss_pcc_q = (qcc_out ** 2).mean()

    loss_global_p_all = ((z_flat + pinj.sum(dim=1, keepdim=True)) ** 2).reshape(B, K)
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
    per_sample = (
        loss_pcc_p_all.reshape(B, K)
        + qcc_out.pow(2).reshape(B, K)
        + loss_global_p_all.reshape(B, K)
        + (qinj.sum(dim=1, keepdim=True) ** 2).reshape(B, K)
        + (((relu(vmin2 - V) ** 2) + (relu(V - vmax2) ** 2)).mean(dim=1, keepdim=True)).reshape(B, K)
        + ((relu(torch.abs(P) - fmax_p) ** 2).mean(dim=1, keepdim=True)).reshape(B, K)
        + ((relu(torch.abs(Q) - fmax_q) ** 2).mean(dim=1, keepdim=True)).reshape(B, K)
        + ((((relu(pg_min - pg) ** 2) + (relu(pg - pg_max) ** 2)).mean(dim=1, keepdim=True))).reshape(B, K)
        + ((((relu(qg_min - qg) ** 2) + (relu(qg - qg_max) ** 2)).mean(dim=1, keepdim=True))).reshape(B, K)
        + ((relu(pr[:, pv_idx] - pv_max) ** 2).mean(dim=1, keepdim=True)).reshape(B, K)
        + kcl_weight * (((kcl_p_res ** 2).mean(dim=1, keepdim=True) + (kcl_q_res ** 2).mean(dim=1, keepdim=True)).reshape(B, K))
    )
    if tau_weights is None:
        tau_weights = _gmm_quantile_weights(device, dtype, K=K)
    tau_weights = tau_weights.view(1, K)
    loss = (per_sample * tau_weights).sum(dim=1).mean() * alpha_phys

    # aggregated diagnostics
    loss_pcc_p = (loss_pcc_p_all * tau_weights).sum(dim=1).mean()
    loss_global_p = (loss_global_p_all * tau_weights).sum(dim=1).mean()
    loss_pcc_q = ((qcc_out ** 2).reshape(B, K) * tau_weights).sum(dim=1).mean()
    loss_global_q = (((qinj.sum(dim=1, keepdim=True) ** 2).reshape(B, K)) * tau_weights).sum(dim=1).mean()
    loss_v = ((((relu(vmin2 - V) ** 2) + (relu(V - vmax2) ** 2)).mean(dim=1).reshape(B, K)) * tau_weights).sum(dim=1).mean()
    loss_line_p = (((relu(torch.abs(P) - fmax_p) ** 2).mean(dim=1).reshape(B, K)) * tau_weights).sum(dim=1).mean()
    loss_line_q = (((relu(torch.abs(Q) - fmax_q) ** 2).mean(dim=1).reshape(B, K)) * tau_weights).sum(dim=1).mean()
    loss_pg = (((((relu(pg_min - pg) ** 2) + (relu(pg - pg_max) ** 2)).mean(dim=1).reshape(B, K)) * tau_weights).sum(dim=1).mean())
    loss_qg = (((((relu(qg_min - qg) ** 2) + (relu(qg - qg_max) ** 2)).mean(dim=1).reshape(B, K)) * tau_weights).sum(dim=1).mean())
    loss_pv = (((relu(pr[:, pv_idx] - pv_max) ** 2).mean(dim=1).reshape(B, K) * tau_weights).sum(dim=1).mean())
    loss_kcl = ((((kcl_p_res ** 2).mean(dim=1).reshape(B, K) + (kcl_q_res ** 2).mean(dim=1).reshape(B, K)) * tau_weights).sum(dim=1).mean())

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




def physics_loss_realization(case: GridCase, x_real_raw: torch.Tensor, p0_label: torch.Tensor, pg_hat: torch.Tensor, qg_hat: torch.Tensor, return_parts: bool = False):
    return physics_loss_quantile(case, x_real_raw, p0_label, pg_hat.unsqueeze(1), qg_hat.unsqueeze(1), tau_weights=torch.ones((1,), device=x_real_raw.device, dtype=x_real_raw.dtype), return_parts=return_parts)


def make_empirical_taus(training: bool, device=None, dtype=None):
    fixed = torch.tensor(EMP_QUANTILE_TAUS, device=device, dtype=dtype)
    if training and USE_RANDOM_EMP_QUANTILE_TAUS and N_RANDOM_EMP_QUANTILE_TAUS > 0:
        rand = torch.rand(N_RANDOM_EMP_QUANTILE_TAUS, device=device, dtype=dtype)
        rand = RANDOM_TAU_LOW + (RANDOM_TAU_HIGH - RANDOM_TAU_LOW) * rand
        return torch.cat([fixed, torch.sort(rand).values], dim=0)
    return fixed


def sample_empirical_quantile_batch(XMU_scen, XREAL_scen, YP0_scen, YPG_scen, YQG_scen, scenario_indices, taus):
    xmu_q, xreal_q, p0_q, pg_q, qg_q = [], [], [], [], []
    taus_np = np.asarray(taus, dtype=float).reshape(-1)
    M = YP0_scen.shape[1]
    for i in scenario_indices:
        order = np.argsort(YP0_scen[i])
        for tau in taus_np:
            q_idx = int(round(float(tau) * (M - 1)))
            mc_idx = int(order[q_idx])
            xmu_q.append(XMU_scen[i])
            xreal_q.append(XREAL_scen[i, mc_idx])
            p0_q.append(YP0_scen[i, mc_idx])
            pg_q.append(YPG_scen[i, mc_idx])
            qg_q.append(YQG_scen[i, mc_idx])
    return np.array(xmu_q), np.array(xreal_q), np.array(p0_q).reshape(-1,1), np.array(pg_q), np.array(qg_q)
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


def realization_quantile_sanity_check(case: GridCase, net, x_mu_mean, x_mu_std, x_real_mean, x_real_std, p0_mean, p0_std, n_samples: int = 3, mc_eval_sanity: int = 300, seed: int = 99):
    print("\n=== realization-matched quantile sanity check ===")
    rng = np.random.default_rng(seed)
    sanity_taus = [0.05, 0.25, 0.50, 0.75, 0.95]
    net.eval()
    for s in range(n_samples):
        pd_mu, qd_mu, pr_mu, qr_mu = sample_scenario_means(case, rng)
        x_mu_raw = make_feature_vector(case, pd_mu, pr_mu).reshape(1, -1)
        feasible = []
        std_pd = 0.10 * np.maximum(pd_mu, 1e-3)
        std_pr = 0.12 * np.maximum(pr_mu, 1e-3)
        for _ in range(mc_eval_sanity):
            pd = pd_mu.copy(); pr = pr_mu.copy()
            for i in range(case.n_bus):
                if pd_mu[i] > 1e-9:
                    pd[i] = sample_trunc_normal(pd_mu[i], std_pd[i], lo=0.0, hi=None)
            for k, b in enumerate(case.pv_buses):
                pr[b] = sample_trunc_normal(pr_mu[b], std_pr[b], lo=0.0, hi=float(case.pv_pmax[k]))
            qd = qd_mu * (pd / np.maximum(pd_mu, 1e-6))
            qr = pr * math.tan(math.acos(case.pv_pf))
            sol = solve_pomax_gurobi_33bus(case, pd, qd, pr, qr, return_detail=True)
            if sol["ok"]:
                feasible.append((make_feature_vector(case, pd, pr), float(sol["P0"]), np.asarray(sol["Pg"], dtype=float), np.asarray(sol["Qg"], dtype=float)))
        if len(feasible) < 20:
            print(f"[real-q-check s{s}] warning: feasible samples={len(feasible)} < 20, skip.")
            continue
        p0_arr = np.array([f[1] for f in feasible], dtype=float)
        ord_idx = np.argsort(p0_arr)
        pick = [ord_idx[int(round(t * (len(ord_idx) - 1)))] for t in sanity_taus]
        x_real_tau_raw = np.stack([feasible[i][0] for i in pick], axis=0).astype(float)
        p0_tau = np.array([[feasible[i][1]] for i in pick], dtype=float)
        pg_opf = np.stack([feasible[i][2] for i in pick], axis=0).astype(float)
        qg_opf = np.stack([feasible[i][3] for i in pick], axis=0).astype(float)
        with torch.no_grad():
            xmu_n = (x_mu_raw - x_mu_mean) / x_mu_std
            xmu_t = torch.tensor(xmu_n, dtype=torch.float32, device=DEVICE)
            hmu = net.encode(xmu_t, sample=False)
            w, mu1, s1, mu2, s2 = net.gmm_head(hmu, sample=False)
            zgmm = gmm2_quantile_torch(w, mu1, s1, mu2, s2, sanity_taus).reshape(-1).detach().cpu().numpy()
            K = len(sanity_taus)
            h_rep = hmu.repeat(K, 1)
            xr_n = (x_real_tau_raw - x_real_mean) / x_real_std
            xr_t = torch.tensor(xr_n, dtype=torch.float32, device=DEVICE)
            xr_raw_t = torch.tensor(x_real_tau_raw, dtype=torch.float32, device=DEVICE)
            p0_t = torch.tensor(p0_tau, dtype=torch.float32, device=DEVICE)
            pg_pred, qg_pred = net.recover_dispatch_from_h_omega_z(h_rep, xr_t, p0_t, torch.tensor([[p0_mean]], dtype=torch.float32, device=DEVICE), torch.tensor([[p0_std]], dtype=torch.float32, device=DEVICE), sample=False)
            rec = recover_flows_from_dispatch_batch(case, xr_raw_t, p0_t, pg_pred, qg_pred)
            pg_prev, qg_prev = None, None
            for k, tau in enumerate(sanity_taus):
                pgv = pg_pred[k].detach().cpu().numpy(); qgv = qg_pred[k].detach().cpu().numpy()
                if k == 0:
                    dpg, dqg = np.nan, np.nan
                else:
                    dpg = np.linalg.norm(pgv - pg_prev, ord=1); dqg = np.linalg.norm(qgv - qg_prev, ord=1)
                pd_sum = float(x_real_tau_raw[k, :case.n_bus].sum())
                pr_sum = float(x_real_tau_raw[k, case.n_bus:case.n_bus + len(case.pv_buses)].sum())
                root_out_p = float(rec["P"][k, case.out_branches[case.root]].sum().cpu())
                pcc_p_res = root_out_p - float(p0_tau[k, 0])
                global_p_res = float((p0_t[k, 0] + rec["pinj"][k].sum()).cpu())
                l1_pg_err = float(np.linalg.norm(pgv - pg_opf[k], ord=1))
                l1_qg_err = float(np.linalg.norm(qgv - qg_opf[k], ord=1))
                print(
                    f"[real-q-check s{s} tau={tau:.2f}] P0_label={p0_tau[k,0]:.4f}, GMM_z={zgmm[k]:.4f}, "
                    f"sumPd_real={pd_sum:.4f}, sumPr_real={pr_sum:.4f}, "
                    f"Pg_pred={np.array2string(pgv, precision=4, suppress_small=True)}, "
                    f"Qg_pred={np.array2string(qgv, precision=4, suppress_small=True)}, "
                    f"Pg_OPF={np.array2string(pg_opf[k], precision=4, suppress_small=True)}, "
                    f"Qg_OPF={np.array2string(qg_opf[k], precision=4, suppress_small=True)}, "
                    f"sumPg_pred={pgv.sum():.4f}, sumQg_pred={qgv.sum():.4f}, sumPg_OPF={pg_opf[k].sum():.4f}, sumQg_OPF={qg_opf[k].sum():.4f}, "
                    f"root_out_P={root_out_p:.4f}, pcc_p_res={pcc_p_res:+.2e}, global_p_res={global_p_res:+.2e}, "
                    f"minV={float(rec['V'][k].min()):.4f}, max|P|={float(rec['P'][k].abs().max()):.4f}, max|Q|={float(rec['Q'][k].abs().max()):.4f}, "
                    f"L1_Pg_err={l1_pg_err:.4e}, L1_Qg_err={l1_qg_err:.4e}, delta_from_prev: L1_dPg={dpg:.4e}, L1_dQg={dqg:.4e}"
                )
                pg_prev, qg_prev = pgv, qgv

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

def _normal_cdf_torch(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gmm2_cdf_torch(z, w, mu1, s1, mu2, s2):
    c1 = _normal_cdf_torch((z - mu1) / (s1 + 1e-12))
    c2 = _normal_cdf_torch((z - mu2) / (s2 + 1e-12))
    return w[:,0:1] * c1 + w[:,1:2] * c2

def _taus_to_tensor(taus, device, dtype):
    if torch.is_tensor(taus):
        return taus.to(device=device, dtype=dtype).view(1, -1)
    return torch.tensor(taus, device=device, dtype=dtype).view(1, -1)


def gmm2_quantile_torch(w, mu1, s1, mu2, s2, taus, n_iter=50):
    B = w.shape[0]
    taus_1 = _taus_to_tensor(taus, w.device, w.dtype)
    K = taus_1.shape[1]
    taus_t = taus_1.repeat(B, 1)
    lower = torch.minimum(mu1 - 8.0*s1, mu2 - 8.0*s2).repeat(1, K)
    upper = torch.maximum(mu1 + 8.0*s1, mu2 + 8.0*s2).repeat(1, K)
    for _ in range(n_iter):
        mid=(lower+upper)/2.0
        cdf_mid = gmm2_cdf_torch(mid, w, mu1, s1, mu2, s2)
        go_right = cdf_mid < taus_t
        lower = torch.where(go_right, mid, lower)
        upper = torch.where(go_right, upper, mid)
    z = 0.5*(lower+upper)
    if DETACH_QUANTILE_Z:
        # stability fallback: stop gradient from quantile solver to GMM parameters
        z = z.detach()
    return z


def make_physics_taus(training: bool, device=None, dtype=None):
    fixed = torch.tensor(FIXED_QUANTILE_TAUS, device=device, dtype=dtype)
    if training and USE_RANDOM_PHYS_TAUS and N_RANDOM_PHYS_TAUS > 0:
        rand = torch.rand(N_RANDOM_PHYS_TAUS, device=device, dtype=dtype)
        rand = RANDOM_TAU_LOW + (RANDOM_TAU_HIGH - RANDOM_TAU_LOW) * rand
        rand = torch.sort(rand).values
        return torch.cat([fixed, rand], dim=0)
    return fixed


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


class BayesGMM2QuantileDispatchNet(nn.Module):
    def __init__(self, in_dim: int, case: GridCase, hidden=160, depth=3, prior_sigma=1.0, init_rho=-5.0):
        super().__init__()
        self.n_gen = len(case.gen_buses)
        self.linears = nn.ModuleList()
        d = in_dim
        for _ in range(depth):
            self.linears.append(BayesLinear(d, hidden, prior_sigma=prior_sigma, init_rho=init_rho))
            d = hidden
        self.gmm_out = BayesLinear(d, 6, prior_sigma=prior_sigma, init_rho=init_rho)
        self.x_dim = in_dim
        self.dispatch_out = BayesLinear(d + self.x_dim + 1, 2 * self.n_gen, prior_sigma=prior_sigma, init_rho=init_rho)
        self.act = nn.ReLU()

        # Register bounds as buffers so they move automatically with .to(device).
        self.register_buffer("pg_min_t", torch.tensor(case.pg_min, dtype=torch.float32).view(1, -1))
        self.register_buffer("pg_max_t", torch.tensor(case.pg_max, dtype=torch.float32).view(1, -1))
        self.register_buffer("qg_min_t", torch.tensor(case.qg_min, dtype=torch.float32).view(1, -1))
        self.register_buffer("qg_max_t", torch.tensor(case.qg_max, dtype=torch.float32).view(1, -1))

    def encode(self, x, sample=True):
        h = x
        for layer in self.linears:
            h = self.act(layer(h, sample=sample))
        return h

    def gmm_head(self, h, sample=True):
        out = self.gmm_out(h, sample=sample)
        w = torch.softmax(out[:, 0:2], dim=1)
        mu1, mu2 = out[:, 2:3], out[:, 4:5]
        s1 = torch.nn.functional.softplus(out[:, 3:4]) + 1e-3
        s2 = torch.nn.functional.softplus(out[:, 5:6]) + 1e-3
        return w, mu1, s1, mu2, s2

    def recover_dispatch_from_h_omega_z(self, h, x_real_norm, z, p0_mean, p0_std, sample=True):
        z_norm = (z - p0_mean) / (p0_std + 1e-9)
        hz = torch.cat([h, x_real_norm, z_norm], dim=1)
        out = self.dispatch_out(hz, sample=sample)
        pg_raw = out[:, :self.n_gen]
        qg_raw = out[:, self.n_gen:]
        pg_min = self.pg_min_t.to(device=h.device, dtype=h.dtype)
        pg_max = self.pg_max_t.to(device=h.device, dtype=h.dtype)
        qg_min = self.qg_min_t.to(device=h.device, dtype=h.dtype)
        qg_max = self.qg_max_t.to(device=h.device, dtype=h.dtype)
        pg_hat = pg_min + torch.sigmoid(pg_raw) * (pg_max - pg_min)
        qg_hat = qg_min + torch.sigmoid(qg_raw) * (qg_max - qg_min)
        return pg_hat, qg_hat

    def forward_gmm(self, x, sample=True):
        h = self.encode(x, sample=sample)
        return self.gmm_head(h, sample=sample)

    def forward_recovery(self, x_mu_norm, x_real_norm, z, p0_mean, p0_std, sample=True):
        h = self.encode(x_mu_norm, sample=sample)
        return self.recover_dispatch_from_h_omega_z(h, x_real_norm, z, p0_mean, p0_std, sample=sample)

    def kl_divergence(self):
        return sum(l.kl_divergence() for l in self.linears) + self.gmm_out.kl_divergence() + self.dispatch_out.kl_divergence()


# =============================
# 7) train
# =============================
def train_bayes_gmm2(case: GridCase, XMU: np.ndarray, XREAL: np.ndarray, YP0: np.ndarray, YPG: np.ndarray, YQG: np.ndarray):
    rng = np.random.default_rng(SEED_TRAIN)
    torch.manual_seed(SEED_TRAIN); np.random.seed(SEED_TRAIN)
    n_scen = XMU.shape[0]; n_gen = len(case.gen_buses)
    n_val = max(1, int(round(VAL_RATIO * n_scen))); n_train = n_scen - n_val
    idx = rng.permutation(n_scen); tr_idx, va_idx = idx[:n_train], idx[n_train:]
    XMU_tr, XREAL_tr, YP0_tr, YPG_tr, YQG_tr = XMU[tr_idx], XREAL[tr_idx], YP0[tr_idx], YPG[tr_idx], YQG[tr_idx]
    XMU_va, XREAL_va, YP0_va, YPG_va, YQG_va = XMU[va_idx], XREAL[va_idx], YP0[va_idx], YPG[va_idx], YQG[va_idx]
    x_mu_mean = XMU_tr.mean(0, keepdims=True); x_mu_std = XMU_tr.std(0, keepdims=True) + 1e-9
    xr_flat = XREAL_tr.reshape(-1, XREAL_tr.shape[-1]); x_real_mean = xr_flat.mean(0, keepdims=True); x_real_std = xr_flat.std(0, keepdims=True)+1e-9

    def flatten(xmu, xreal, yp0, ypg, yqg):
        N,M,_ = xreal.shape
        xmu_n = (xmu - x_mu_mean)/x_mu_std
        xreal_n = (xreal - x_real_mean)/x_real_std
        xmu_f = np.repeat(xmu_n, M, axis=0)
        xreal_f = xreal_n.reshape(-1, xreal_n.shape[-1])
        xreal_raw_f = xreal.reshape(-1, xreal.shape[-1])
        yp0_f = yp0.reshape(-1,1); ypg_f = ypg.reshape(-1, ypg.shape[-1]); yqg_f = yqg.reshape(-1, yqg.shape[-1])
        return xmu_f, xreal_f, xreal_raw_f, yp0_f, ypg_f, yqg_f

    xt_mu, xt_real, xt_real_raw, yp0_t, ypg_t, yqg_t = flatten(XMU_tr, XREAL_tr, YP0_tr, YPG_tr, YQG_tr)
    xv_mu, xv_real, xv_real_raw, yp0_v, ypg_v, yqg_v = flatten(XMU_va, XREAL_va, YP0_va, YPG_va, YQG_va)

    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    xt_mu, xt_real, xt_real_raw, yp0_t, ypg_t, yqg_t = map(t, [xt_mu, xt_real, xt_real_raw, yp0_t, ypg_t, yqg_t])
    xv_mu, xv_real, xv_real_raw, yp0_v, ypg_v, yqg_v = map(t, [xv_mu, xv_real, xv_real_raw, yp0_v, ypg_v, yqg_v])
    p0_mean_t = torch.tensor([[float(YP0_tr.mean())]], dtype=torch.float32, device=DEVICE)
    p0_std_t = torch.tensor([[float(YP0_tr.std()+1e-9)]], dtype=torch.float32, device=DEVICE)

    net = BayesGMM2QuantileDispatchNet(in_dim=XMU.shape[1], case=case, prior_sigma=PRIOR_SIGMA, init_rho=INIT_RHO).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n_data = xt_mu.shape[0]; n_batches = (n_data + BATCH_SIZE - 1)//BATCH_SIZE
    print("=== 开始训练 33-bus VI-BPINN (v8 realization-conditioned physics) ===")

    for ep in range(EPOCHS):
        perm = rng.permutation(n_data); beta = BETA_KL_MAX * min(1.0, (ep+1)/max(1,KL_WARMUP_EPOCHS))
        ep_loss=ep_nll=ep_disp=ep_phys=ep_emp=ep_kl=0.0; ep_n_emp_taus=0
        for b in range(n_batches):
            ii = perm[b*BATCH_SIZE:min((b+1)*BATCH_SIZE,n_data)]
            xb_mu, xb_real, xb_real_raw, yb_p0, yb_pg, yb_qg = xt_mu[ii], xt_real[ii], xt_real_raw[ii], yp0_t[ii], ypg_t[ii], yqg_t[ii]
            h = net.encode(xb_mu, sample=True)
            w, mu1, s1, mu2, s2 = net.gmm_head(h, sample=True)
            nll = (-gmm2_log_prob(yb_p0, w, mu1, s1, mu2, s2)).mean()
            pg_hat, qg_hat = net.recover_dispatch_from_h_omega_z(h, xb_real, yb_p0, p0_mean_t, p0_std_t, sample=True)
            disp = dispatch_supervision_loss(case, pg_hat, qg_hat, yb_pg, yb_qg, normalize=NORMALIZE_DISPATCH_SUP)
            phys_real = physics_loss_realization(case, xb_real_raw, yb_p0, pg_hat, qg_hat)
            phys_emp = torch.tensor(0.0, device=DEVICE)
            if USE_EMPIRICAL_QUANTILE_PHYS:
                taus_emp = make_empirical_taus(True, device=xb_mu.device, dtype=xb_mu.dtype)
                ep_n_emp_taus = int(taus_emp.numel())
                scen_pick = rng.choice(XMU_tr.shape[0], size=min(64, XMU_tr.shape[0]), replace=False)
                xmu_q, xr_q, p0_q, pg_q_lab, qg_q_lab = sample_empirical_quantile_batch(XMU_tr, XREAL_tr, YP0_tr, YPG_tr, YQG_tr, scen_pick, taus_emp.detach().cpu().numpy())
                xmu_qn = t((xmu_q - x_mu_mean)/x_mu_std); xr_qn = t((xr_q - x_real_mean)/x_real_std); xr_qraw = t(xr_q); p0_qt = t(p0_q)
                hq = net.encode(xmu_qn, sample=True)
                pg_qh, qg_qh = net.recover_dispatch_from_h_omega_z(hq, xr_qn, p0_qt, p0_mean_t, p0_std_t, sample=True)
                phys_emp = physics_loss_realization(case, xr_qraw, p0_qt, pg_qh, qg_qh)
            kl = net.kl_divergence()
            loss = nll + LAM_DISPATCH_SUP*disp + LAM_PHYS_REALIZATION*phys_real + LAM_PHYS_EMP_QUANTILE*phys_emp + beta*kl/n_data
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
            ep_loss += float(loss.detach().cpu()); ep_nll += float(nll.detach().cpu()); ep_disp += float(disp.detach().cpu()); ep_phys += float(phys_real.detach().cpu()); ep_emp += float(phys_emp.detach().cpu()); ep_kl += float((kl/n_data).detach().cpu())

        if ep==0 or (ep+1)%LOG_EVERY==0 or ep+1==EPOCHS:
            with torch.no_grad():
                hv = net.encode(xv_mu, sample=False); wv,mu1v,s1v,mu2v,s2v = net.gmm_head(hv, sample=False)
                val_nll = (-gmm2_log_prob(yp0_v,wv,mu1v,s1v,mu2v,s2v)).mean()
                pgv,qgv = net.recover_dispatch_from_h_omega_z(hv, xv_real, yp0_v, p0_mean_t, p0_std_t, sample=False)
                val_disp, _ = dispatch_supervision_loss(case, pgv, qgv, ypg_v, yqg_v, normalize=NORMALIZE_DISPATCH_SUP, return_parts=True)
                val_phys_real, val_parts = physics_loss_realization(case, xv_real_raw, yp0_v, pgv, qgv, return_parts=True)
                taus_eval = make_empirical_taus(False, device=xv_mu.device, dtype=xv_mu.dtype)
                scen_pick = np.arange(XMU_va.shape[0])
                xmu_q, xr_q, p0_q, _, _ = sample_empirical_quantile_batch(XMU_va, XREAL_va, YP0_va, YPG_va, YQG_va, scen_pick, taus_eval.detach().cpu().numpy())
                xmu_qn = t((xmu_q - x_mu_mean)/x_mu_std); xr_qn = t((xr_q - x_real_mean)/x_real_std); xr_qraw = t(xr_q); p0_qt = t(p0_q)
                hq = net.encode(xmu_qn, sample=False); pg_qh, qg_qh = net.recover_dispatch_from_h_omega_z(hq, xr_qn, p0_qt, p0_mean_t, p0_std_t, sample=False)
                val_emp = physics_loss_realization(case, xr_qraw, p0_qt, pg_qh, qg_qh)
                val_total = val_nll + LAM_DISPATCH_SUP*val_disp + LAM_PHYS_REALIZATION*val_phys_real + LAM_PHYS_EMP_QUANTILE*val_emp + beta*net.kl_divergence()/n_data
            print(f"Epoch {ep+1:4d} | avg_loss={ep_loss/n_batches:.6f} avg_nll={ep_nll/n_batches:.6f} avg_disp_sup={ep_disp/n_batches:.6f} avg_phys_real={ep_phys/n_batches:.6f} avg_phys_emp_quantile={ep_emp/n_batches:.6f} avg_kl/N={ep_kl/n_batches:.6f} beta={beta:.4f} n_batches={n_batches} n_emp_taus={ep_n_emp_taus} | val_loss={val_total.item():.6f} val_nll={val_nll.item():.6f} val_disp_sup={val_disp.item():.6f} val_phys_real={val_phys_real.item():.6f} val_emp_quantile_phys={val_emp.item():.6f} val_pcc_p={val_parts['pcc_p'].item():.3e} val_global_p={val_parts['global_p'].item():.3e} val_voltage={val_parts['voltage'].item():.3e} val_line_p={val_parts['line_p'].item():.3e} val_pg={val_parts['pg'].item():.3e} val_qg={val_parts['qg'].item():.3e}")

    return net, x_mu_mean, x_mu_std, x_real_mean, x_real_std, float(p0_mean_t.item()), float(p0_std_t.item())



# =============================
# 8) eval + plot
# =============================
def draw_test_scenario(case: GridCase, seed=123):
    rng = np.random.default_rng(seed)
    return sample_scenario_means(case, rng)


def eval_and_plot(case: GridCase, net: BayesGMM2QuantileDispatchNet, x_mean: np.ndarray, x_std: np.ndarray, mc_eval=2500):
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
            w_t, mu1_t, s1_t, mu2_t, s2_t = net.forward_gmm(xt, sample=True)
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
    cdf_lo = np.quantile(cdf_samps, 0.025, axis=0)
    cdf_hi = np.quantile(cdf_samps, 0.975, axis=0)

    arms = 100.0 * math.sqrt(np.mean((cdf_mean - cdf_mc) ** 2))
    q05_mc = float(np.quantile(y_mc, 0.05))
    q05_arr = np.array(q05_samps, dtype=float)
    q05_arr = q05_arr[np.isfinite(q05_arr)]
    q05_mean = float(np.mean(q05_arr)) if q05_arr.size > 0 else float("nan")

    print(f"[33bus B-PINN] ARMS(CDF) = {arms:.4f}%")
    print(f"Quantile@0.05: MC={q05_mc:.4f}, BNN mean={q05_mean:.4f}")

    plt.figure(figsize=(10, 5.5), dpi=130)
    plt.fill_between(z_grid, cdf_lo, cdf_hi, alpha=0.25, label="B-PINN posterior CDF 2.5%-97.5% band")
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
    plt.title("IEEE 33-bus Pomax CDF — v8 realization-conditioned physics")
    plt.legend(loc="upper left")
    plt.grid(False)
    plt.tight_layout()
    out_png = "Pomax_CDF_33bus_v8_realization_conditioned_physics.png"
    plt.savefig(out_png, dpi=280, bbox_inches="tight")
    plt.show()
    return arms


def eval_multiple_test_scenarios(
    case: GridCase,
    net: BayesGMM2QuantileDispatchNet,
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
                w_t, mu1_t, s1_t, mu2_t, s2_t = net.forward_gmm(xt, sample=True)
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
        dispatch_label_sanity_check(case, n_samples=3, seed=SEED_DATA + 13)
    print("生成 33 节点训练数据...")
    print(f"[config] FAST_MODE={FAST_MODE}")
    print(f"[config] NUM_SCENARIOS={NUM_SCENARIOS}, MC_PER_SCENARIO={MC_PER_SCENARIO}, total_labels={NUM_SCENARIOS * MC_PER_SCENARIO}")
    XMU, XREAL, YP0, YPG, YQG, active_records, all_active_names = generate_dataset(case, NUM_SCENARIOS, MC_PER_SCENARIO, seed=SEED_DATA)
    summarize_active_patterns(active_records, all_active_names, top_k=ACTIVE_TOP_K)
    print(f"Dataset: XMU={XMU.shape}, XREAL={XREAL.shape}, YP0={YP0.shape}, YPG={YPG.shape}, YQG={YQG.shape}")
    print(f"Flattened samples before split = {XMU.shape[0] * YP0.shape[1]}")

    net, x_mu_mean, x_mu_std, x_real_mean, x_real_std, p0_mean, p0_std = train_bayes_gmm2(case, XMU, XREAL, YP0, YPG, YQG)
    if RUN_SANITY_CHECKS:
        realization_quantile_sanity_check(case, net, x_mu_mean, x_mu_std, x_real_mean, x_real_std, p0_mean, p0_std, n_samples=3, seed=SEED_DATA + 21)
    eval_and_plot(case, net, x_mu_mean, x_mu_std, mc_eval=2500)
    if RUN_MULTI_TEST:
        eval_multiple_test_scenarios(
            case,
            net,
            x_mu_mean,
            x_mu_std,
            n_scenarios=N_TEST_SCENARIOS,
            mc_eval_multi=MC_EVAL_MULTI,
        )


if __name__ == "__main__":
    main()
