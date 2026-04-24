# -*- coding: utf-8 -*-
"""
Stage-1 extension: 3-bus VI-BPINN -> IEEE 33-bus single-period VI-BPINN.

Core kept from original script:
- Bayesian NN (VI) + GMM-2 output
- ELBO style training objective: NLL + lambda_phys * physics_loss + beta * KL/N
- output Pomax conditional distribution (PCC active power exchange max)
- evaluate with MC baseline CDF, ARMS, q0.05 comparison

What is new in this stage:
- General radial-network LinDistFlow model for 33-bus
- Generic OPF label generator (Gurobi) for Pomax
- Generic mean-only physics loss with tree-based flow/voltage recovery
- Dataset generation expanded to multi-node load/PV features

IMPORTANT (stage-1 approximation):
- physics loss uses "mean-level surrogate dispatch" recovered from
  (scenario mean inputs + predicted mu_mix of Pomax)
- this is intentionally a simple and explainable surrogate for large systems
- can be replaced later by decision-recovery nets or differentiable OPF layer
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
NUM_SCENARIOS = 180
MC_PER_SCENARIO = 60

EPOCHS = 1200
BATCH_SIZE = 1024
LR = 1e-3
LAM_PHYS = 0.08

PRIOR_SIGMA = 1.0
BETA_KL_MAX = 1.0
KL_WARMUP_EPOCHS = 500
INIT_RHO = -5.0
TRAIN_WEIGHT_SAMPLES = 1
EVAL_THETA_SAMPLES = 50
RUN_SANITY_CHECKS = True

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
    part_p: np.ndarray
    part_q: np.ndarray
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

    # stage-1 participation factors (switchable/configurable)
    part_p = pg_max / (pg_max.sum() + 1e-12)
    q_span = (qg_max - qg_min)
    part_q = q_span / (q_span.sum() + 1e-12)

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
        part_p=part_p, part_q=part_q,
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

    feats, ys = [], []
    dropped_scenarios = 0
    for s in range(num_scenarios):
        pd_mu, qd_mu, pr_mu, qr_mu = sample_scenario_means(case, rng)
        x = make_feature_vector(case, pd_mu, pr_mu)

        std_pd = 0.10 * np.maximum(pd_mu, 1e-3)
        std_pr = 0.12 * np.maximum(pr_mu, 1e-3)

        y_row = []
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

            y = solve_pomax_gurobi_33bus(case, pd, qd, pr, qr)
            if np.isfinite(y):
                y_row.append(y)

        if len(y_row) < max(5, mc_per_scenario // 8):
            # fallback-1: deterministic mean point
            y_fb = solve_pomax_gurobi_33bus(case, pd_mu, qd_mu, pr_mu, qr_mu)
            if np.isfinite(y_fb):
                y_row = [y_fb for _ in range(mc_per_scenario)]
            else:
                # fallback-2: progressively relax the sampled operating point (loads down, PV clipped)
                for scale in [0.95, 0.90, 0.85, 0.80]:
                    pd_try = pd_mu * scale
                    qd_try = qd_mu * scale
                    pr_cap = np.zeros(case.n_bus, dtype=float)
                    pr_cap[case.pv_buses] = case.pv_pmax
                    pr_try = np.minimum(pr_mu, pr_cap) * scale
                    qr_try = pr_try * math.tan(math.acos(case.pv_pf))
                    y_fb2 = solve_pomax_gurobi_33bus(case, pd_try, qd_try, pr_try, qr_try)
                    if np.isfinite(y_fb2):
                        y_row = [y_fb2 for _ in range(mc_per_scenario)]
                        break

        if len(y_row) > 0:
            if len(y_row) < mc_per_scenario:
                fill = rng.choice(y_row, size=mc_per_scenario - len(y_row), replace=True)
                y_row = y_row + fill.tolist()
            feats.append(x)
            ys.append(np.array(y_row[:mc_per_scenario], dtype=float))
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
    Y = np.stack(ys, axis=0).astype(float)
    print(f"[dataset] feasible_scenarios={len(feats)}/{num_scenarios}, dropped={dropped_scenarios}")
    return X, Y


# =============================
# 4) Generic mean-only physics loss
# =============================
def recover_surrogate_dispatch_and_flows(case: GridCase, x_raw: torch.Tensor, mu_p0: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Recover mean-level surrogate dispatch from:
    - scenario mean inputs (x_raw)
    - predicted mu_mix of Pomax (mu_p0)

    This is a stage-1 surrogate recovery, not exact OPF recovery.
    """
    B = x_raw.shape[0]
    nb = case.n_bus
    nl = case.from_bus.size
    n_pv = len(case.pv_buses)

    pd = x_raw[:, :nb]
    pr_pv = x_raw[:, nb:nb + n_pv]
    pr = torch.zeros((B, nb), device=x_raw.device, dtype=x_raw.dtype)
    pr[:, torch.tensor(case.pv_buses, device=x_raw.device)] = pr_pv

    tan_pv = math.tan(math.acos(case.pv_pf))
    qr = pr * tan_pv

    # load Q recovered from fixed node PF implied by base data
    ratio_qp = np.divide(case.qd_base, np.maximum(case.pd_base, 1e-6))
    qd = pd * torch.tensor(ratio_qp, device=x_raw.device, dtype=x_raw.dtype).view(1, -1)

    # surrogate generator dispatch via participation factors + balance target from mu_p0
    # Unified convention: P0 is import at PCC, so global active balance is:
    #   P0 + sum_i(pinj_i) = 0  =>  sum(Pg) = sum(Pd)-sum(Pr)-P0
    total_pd = pd.sum(dim=1, keepdim=True)
    total_pr = pr.sum(dim=1, keepdim=True)
    total_pg_need = total_pd - total_pr - mu_p0

    total_qd = qd.sum(dim=1, keepdim=True)
    total_qr = qr.sum(dim=1, keepdim=True)
    total_qg_need = total_qd - total_qr  # PCC Q fixed to 0 in this stage: sum(qinj)=0

    part_p = torch.tensor(case.part_p, device=x_raw.device, dtype=x_raw.dtype).view(1, -1)
    part_q = torch.tensor(case.part_q, device=x_raw.device, dtype=x_raw.dtype).view(1, -1)

    pg = total_pg_need * part_p
    qg = total_qg_need * part_q

    # nodal net injection (generation + RE - load)
    pinj = pr - pd
    qinj = qr - qd
    gen_b = torch.tensor(case.gen_buses, device=x_raw.device)
    pinj[:, gen_b] += pg
    qinj[:, gen_b] += qg

    # radial flow recovery: post-order accumulate subtree injections.
    # For branch p->j with positive direction p->j:
    #   P_pj = - (sum injections in subtree rooted at j)
    P = torch.zeros((B, nl), device=x_raw.device, dtype=x_raw.dtype)
    Q = torch.zeros((B, nl), device=x_raw.device, dtype=x_raw.dtype)
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

    # voltage recovery: forward sweep from root V^2=1
    V = torch.zeros((B, nb), device=x_raw.device, dtype=x_raw.dtype)
    V[:, case.root] = 1.0
    r_t = torch.tensor(case.r, device=x_raw.device, dtype=x_raw.dtype)
    x_t = torch.tensor(case.x, device=x_raw.device, dtype=x_raw.dtype)

    bus_pos = {b: k for k, b in enumerate(case.topo_order)}
    for j in case.topo_order:
        if j == case.root:
            continue
        l = int(case.parent_branch[j])
        p = int(case.parent_bus[j])
        _ = bus_pos[j]
        V[:, j] = V[:, p] - 2.0 * (r_t[l] * P[:, l] + x_t[l] * Q[:, l])

    return {"pd": pd, "qd": qd, "pr": pr, "qr": qr, "pg": pg, "qg": qg, "pinj": pinj, "qinj": qinj, "P": P, "Q": Q, "V": V}


def physics_loss_meanonly(case: GridCase, x_raw: torch.Tensor, mu_p0: torch.Tensor, alpha_phys: float = 1.0) -> torch.Tensor:
    rec = recover_surrogate_dispatch_and_flows(case, x_raw, mu_p0)
    relu = torch.relu

    loss = 0.0
    V = rec["V"]
    P = rec["P"]
    Q = rec["Q"]
    pg = rec["pg"]
    qg = rec["qg"]
    pr = rec["pr"]
    pinj = rec["pinj"]
    qinj = rec["qinj"]

    vmin2, vmax2 = case.vmin ** 2, case.vmax ** 2
    loss = loss + (relu(vmin2 - V) ** 2).mean() + (relu(V - vmax2) ** 2).mean()

    fmax_p = torch.tensor(case.fmax_p, device=x_raw.device, dtype=x_raw.dtype).view(1, -1)
    fmax_q = torch.tensor(case.fmax_q, device=x_raw.device, dtype=x_raw.dtype).view(1, -1)
    loss = loss + (relu(torch.abs(P) - fmax_p) ** 2).mean()
    loss = loss + (relu(torch.abs(Q) - fmax_q) ** 2).mean()

    pg_min = torch.tensor(case.pg_min, device=x_raw.device, dtype=x_raw.dtype).view(1, -1)
    pg_max = torch.tensor(case.pg_max, device=x_raw.device, dtype=x_raw.dtype).view(1, -1)
    qg_min = torch.tensor(case.qg_min, device=x_raw.device, dtype=x_raw.dtype).view(1, -1)
    qg_max = torch.tensor(case.qg_max, device=x_raw.device, dtype=x_raw.dtype).view(1, -1)
    loss = loss + (relu(pg_min - pg) ** 2).mean() + (relu(pg - pg_max) ** 2).mean()
    loss = loss + (relu(qg_min - qg) ** 2).mean() + (relu(qg - qg_max) ** 2).mean()

    # renewable upper bounds (mean-level)
    pv_idx = torch.tensor(case.pv_buses, device=x_raw.device)
    pv_max = torch.tensor(case.pv_pmax, device=x_raw.device, dtype=x_raw.dtype).view(1, -1)
    loss = loss + (relu(rec["pr"][:, pv_idx] - pv_max) ** 2).mean()

    # PCC consistency under unified sign: sum(root outgoing) == P0(import)
    root_out = case.out_branches[case.root]
    pcc_out = rec["P"][:, root_out].sum(dim=1, keepdim=True)
    qcc_out = rec["Q"][:, root_out].sum(dim=1, keepdim=True)
    loss = loss + ((pcc_out - mu_p0) ** 2).mean()

    # Global balance consistency residuals
    loss = loss + ((mu_p0 + pinj.sum(dim=1, keepdim=True)) ** 2).mean()
    loss = loss + ((qinj.sum(dim=1, keepdim=True)) ** 2).mean()
    loss = loss + (qcc_out ** 2).mean()

    # Nodal KCL residual consistency (non-root)
    kcl_p_res = []
    kcl_q_res = []
    for i in range(case.n_bus):
        if i == case.root:
            continue
        in_br = case.in_branches[i]
        out_br = case.out_branches[i]
        kcl_p = rec["P"][:, in_br].sum(dim=1, keepdim=True) - rec["P"][:, out_br].sum(dim=1, keepdim=True) + pinj[:, i:i + 1]
        kcl_q = rec["Q"][:, in_br].sum(dim=1, keepdim=True) - rec["Q"][:, out_br].sum(dim=1, keepdim=True) + qinj[:, i:i + 1]
        kcl_p_res.append(kcl_p)
        kcl_q_res.append(kcl_q)
    kcl_p_res = torch.cat(kcl_p_res, dim=1)
    kcl_q_res = torch.cat(kcl_q_res, dim=1)
    loss = loss + (kcl_p_res ** 2).mean() + (kcl_q_res ** 2).mean()

    return alpha_phys * loss


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


def surrogate_sanity_check(case: GridCase, n_samples: int = 3, seed: int = 7):
    print("\\n=== Surrogate recovery sanity check ===")
    rng = np.random.default_rng(seed)
    for k in range(n_samples):
        pd_mu, _, pr_mu, _ = sample_scenario_means(case, rng)
        x = make_feature_vector(case, pd_mu, pr_mu).reshape(1, -1)
        # heuristic mu_p0 proxy (positive import)
        mu_p0 = np.array([[max(0.05, pd_mu.sum() - pr_mu.sum() - 0.4)]], dtype=float)
        xt = torch.tensor(x, dtype=torch.float32)
        p0t = torch.tensor(mu_p0, dtype=torch.float32)
        rec = recover_surrogate_dispatch_and_flows(case, xt, p0t)

        total_pd = float(rec["pd"].sum())
        total_pr = float(rec["pr"].sum())
        total_pg = float(rec["pg"].sum())
        total_qd = float(rec["qd"].sum())
        total_qr = float(rec["qr"].sum())
        total_qg = float(rec["qg"].sum())
        p_root = float(rec["P"][:, case.out_branches[case.root]].sum())
        q_root = float(rec["Q"][:, case.out_branches[case.root]].sum())
        v_min = float(rec["V"].min())
        print(f"[sur-check {k}] Pbal: P0={mu_p0[0,0]:.4f}, Pg={total_pg:.4f}, Pd={total_pd:.4f}, Pr={total_pr:.4f}, root_out={p_root:.4f}")
        print(f"             Qbal: Qg={total_qg:.4f}, Qd={total_qd:.4f}, Qr={total_qr:.4f}, rootQ={q_root:+.2e}, Vmin^2={v_min:.4f}")

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


class BayesGMM2Net(nn.Module):
    def __init__(self, in_dim: int, hidden=160, depth=3, prior_sigma=1.0, init_rho=-5.0):
        super().__init__()
        self.linears = nn.ModuleList()
        d = in_dim
        for _ in range(depth):
            self.linears.append(BayesLinear(d, hidden, prior_sigma=prior_sigma, init_rho=init_rho))
            d = hidden
        self.out = BayesLinear(d, 6, prior_sigma=prior_sigma, init_rho=init_rho)
        self.act = nn.ReLU()

    def forward(self, x, sample=True):
        h = x
        for layer in self.linears:
            h = self.act(layer(h, sample=sample))
        out = self.out(h, sample=sample)
        w = torch.softmax(out[:, 0:2], dim=1)
        mu1, mu2 = out[:, 2:3], out[:, 4:5]
        s1 = torch.nn.functional.softplus(out[:, 3:4]) + 1e-3
        s2 = torch.nn.functional.softplus(out[:, 5:6]) + 1e-3
        return w, mu1, s1, mu2, s2

    def kl_divergence(self):
        return sum(l.kl_divergence() for l in self.linears) + self.out.kl_divergence()


# =============================
# 7) train
# =============================
def train_bayes_gmm2(case: GridCase, X: np.ndarray, Y: np.ndarray):
    rng = np.random.default_rng(SEED_TRAIN)
    torch.manual_seed(SEED_TRAIN)
    np.random.seed(SEED_TRAIN)

    if X.ndim != 2:
        raise ValueError(f"X shape must be (N, d), got {X.shape}")
    if Y.ndim != 2:
        raise ValueError(
            f"Y shape must be (N, M), got {Y.shape}. "
            "这通常表示数据集阶段没有成功生成每个场景的 MC Pomax 样本。"
        )
    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError("空数据集：无法训练。请先检查数据生成可行性。")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X/Y 场景数不一致: X={X.shape}, Y={Y.shape}")

    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True) + 1e-9
    xn = (X - x_mean) / x_std

    N, M = Y.shape
    x_flat = np.repeat(xn, M, axis=0)
    y_flat = Y.reshape(-1, 1)
    mask = np.isfinite(y_flat[:, 0])
    x_flat, y_flat = x_flat[mask], y_flat[mask]

    xt = torch.tensor(x_flat, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_flat, dtype=torch.float32, device=DEVICE)
    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=DEVICE)
    x_std_t = torch.tensor(x_std, dtype=torch.float32, device=DEVICE)

    net = BayesGMM2Net(in_dim=X.shape[1], prior_sigma=PRIOR_SIGMA, init_rho=INIT_RHO).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n_data = xt.shape[0]

    print("=== 开始训练 33-bus VI-BPINN ===")
    for ep in range(EPOCHS):
        idx = rng.integers(0, n_data, size=BATCH_SIZE)
        xb, yb = xt[idx], yt[idx]
        beta = BETA_KL_MAX * min(1.0, (ep + 1) / max(1, KL_WARMUP_EPOCHS))

        nll_acc = 0.0
        phys_acc = 0.0
        for _ in range(TRAIN_WEIGHT_SAMPLES):
            w, mu1, s1, mu2, s2 = net(xb, sample=True)
            nll = (-gmm2_log_prob(yb, w, mu1, s1, mu2, s2)).mean()
            mu_mix = w[:, 0:1] * mu1 + w[:, 1:2] * mu2
            x_raw = xb * x_std_t + x_mean_t
            phys = physics_loss_meanonly(case, x_raw, mu_mix)
            nll_acc += nll
            phys_acc += phys

        nll = nll_acc / TRAIN_WEIGHT_SAMPLES
        phys = phys_acc / TRAIN_WEIGHT_SAMPLES
        kl = net.kl_divergence()
        loss = nll + LAM_PHYS * phys + beta * kl / n_data

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if (ep + 1) % 100 == 0:
            print(f"Epoch {ep+1:4d} | loss={loss.item():.6f} nll={nll.item():.6f} phys={phys.item():.6f} kl/N={(kl/n_data).item():.6f}")

    return net, x_mean, x_std


# =============================
# 8) eval + plot
# =============================
def draw_test_scenario(case: GridCase, seed=123):
    rng = np.random.default_rng(seed)
    return sample_scenario_means(case, rng)


def eval_and_plot(case: GridCase, net: BayesGMM2Net, x_mean: np.ndarray, x_std: np.ndarray, mc_eval=2500):
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
            w_t, mu1_t, s1_t, mu2_t, s2_t = net(xt, sample=True)
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
    plt.title("IEEE 33-bus Pomax CDF (single-period, VI-BPINN GMM2)")
    plt.legend(loc="upper left")
    plt.grid(False)
    plt.tight_layout()
    out_png = "Pomax_CDF_33bus_stage1_VI_BPINN.png"
    plt.savefig(out_png, dpi=280, bbox_inches="tight")
    plt.show()


def main():
    case = build_ieee33_case()
    if RUN_SANITY_CHECKS:
        opf_sanity_check(case, n_samples=3, seed=SEED_DATA + 11)
        surrogate_sanity_check(case, n_samples=3, seed=SEED_DATA + 12)
    print("生成 33 节点训练数据...")
    X, Y = generate_dataset(case, NUM_SCENARIOS, MC_PER_SCENARIO, seed=SEED_DATA)
    print(f"Dataset: X={X.shape}, Y={Y.shape}")

    net, x_mean, x_std = train_bayes_gmm2(case, X, Y)
    eval_and_plot(case, net, x_mean, x_std, mc_eval=2500)


if __name__ == "__main__":
    main()
