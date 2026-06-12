"""v7.4.6.5 polygon vertex OPF feasibility checks.

This version is built on the v7.4.6.4 polygon vertex physical-residual
workflow, but it does not use the NN recovery head as a feasibility certificate.
For every selected polygon vertex, it fixes P0/Q0 and solves the original
LinDistFlow OPF feasibility problem directly. If the fixed point is infeasible,
it optionally solves a nearest feasible P0/Q0 projection OPF.

Windows PowerShell:
    py -m py_compile Po_flex_domain_v7_4_6_5_vertex_opf_feasibility.py
    py Po_flex_domain_v7_4_6_5_vertex_opf_feasibility.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import Po_flex_domain_v7_4_6_4_polygon_vertex_phys_residual as phys64
    v7463 = phys64.v7463
    base = phys64.base
    gp = base.gp
    GRB = base.GRB
    GUROBI_IMPORT_AVAILABLE = True
    GUROBI_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001 - manifest must be writable when gurobipy/base import fails.
    phys64 = None
    v7463 = None
    base = None
    gp = None
    GRB = None
    GUROBI_IMPORT_AVAILABLE = False
    GUROBI_IMPORT_ERROR = repr(exc)

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_SCRIPT_NAME = Path(__file__).name
PREVIOUS_PHYS_RESIDUAL_FILE = "Po_flex_domain_v7_4_6_4_polygon_vertex_phys_residual.py"
BASE_FILE_NAME = getattr(phys64, "BASE_SCRIPT_NAME", "Po_flex_domain_v7_4_6_3_no_pcc_qcal_selected_seed_flex_domain_plots_with_cdf_bands.py")

# =========================
# OPF vertex feasibility config
# =========================
RUN_VERTEX_OPF_FEASIBILITY = True
OPF_VALIDATE_POSTERIOR_Q50_VERTICES = True
OPF_VALIDATE_DETERMINISTIC_BPINN_Q50_VERTICES = True
OPF_VALIDATE_MC_Q50_VERTICES_IF_AVAILABLE = True

OPF_VALIDATE_MODE = "all"
OPF_MAX_VERTICES_PER_SEED = None

OPF_TIME_LIMIT_SEC = 30
OPF_OUTPUT_FLAG = 0
OPF_FEASIBILITY_TOL = 1e-6
OPF_OPTIMALITY_TOL = 1e-6

RUN_PROJECTION_IF_INFEASIBLE = True
PROJECTION_OBJECTIVE = "l1_pq"
PROJECTION_P_SCALE = 1.0
PROJECTION_Q_SCALE = 1.0

SAVE_OPF_DETAIL = True
SAVE_OPF_SUMMARY = True
PLOT_OPF_FEASIBILITY_RESULTS = True
SAVE_OPF_SOLUTIONS_NPZ = True

SOURCE_LOAD_QUANTILE_FOR_OPF = 0.50
SELECTED_EXTERNAL_SEEDS = list(getattr(phys64, "SELECTED_EXTERNAL_SEEDS", [8])) if phys64 is not None else [8]
N_THETA = int(getattr(phys64, "N_THETA", 12)) if phys64 is not None else 12
THETA_VALUES = np.asarray(getattr(phys64, "THETA_VALUES", np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)), dtype=float)
SAVE_PNG = bool(getattr(phys64, "SAVE_PNG", True)) if phys64 is not None else True
SAVE_PDF = bool(getattr(phys64, "SAVE_PDF", True)) if phys64 is not None else True
DPI = int(getattr(phys64, "DPI", 300)) if phys64 is not None else 300

DETAIL_COLUMNS = [
    "seed", "source_polygon", "sample_idx", "vertex_idx", "P0", "Q0", "polygon_area", "max_halfspace_violation", "n_vertices",
    "opf_status", "opf_status_code", "opf_status_category", "opf_feasible", "opf_solve_time_sec", "opf_obj_value", "opf_max_constraint_residual",
    "opf_pcc_p_residual", "opf_pcc_q_residual", "opf_global_p_residual", "opf_global_q_residual",
    "opf_voltage_upper_violation_max", "opf_voltage_lower_violation_max", "opf_line_p_violation_max", "opf_line_q_violation_max",
    "opf_pg_upper_violation_max", "opf_pg_lower_violation_max", "opf_qg_upper_violation_max", "opf_qg_lower_violation_max", "opf_pv_upper_violation_max",
    "projection_status", "projection_feasible", "projection_solve_time_sec", "p0_projected", "q0_projected", "delta_p0_projected", "delta_q0_projected", "projection_l1_distance", "projection_l2_distance",
    "nn_max_phys_violation", "nn_t_abs_error", "nn_p0_forced_recovery_abs_error", "nn_q0_forced_recovery_abs_error", "nn_check_mode",
    "failure_reason", "warning_message",
]

_manifest_artifacts: List[Dict[str, object]] = []


def _seed_tag(seed: int) -> str:
    return f"{int(seed):04d}"


def _abs_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else SCRIPT_DIR / p


def _selected_output_dir() -> Path:
    if v7463 is not None:
        return _abs_path(v7463.SELECTED_FLEX_DOMAIN_OUT_DIR)
    return SCRIPT_DIR / "v746_selected_seed_flex_domain_plots"


def _seed_dir(seed: int) -> Path:
    if phys64 is not None:
        return phys64._seed_dir(seed)
    d = _selected_output_dir() / f"seed_{_seed_tag(seed)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_to_csv(df: pd.DataFrame, path: str | Path, *, description: str) -> Path:
    path = _abs_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[opf-save-csv] {description} -> {path}", flush=True)
    df.to_csv(path, index=False)
    if not path.exists():
        raise RuntimeError(f"CSV save failed: {path}")
    return path


def _record_artifact(seed: int | str, artifact_type: str, path: str | Path, description: str) -> None:
    path = _abs_path(path)
    _manifest_artifacts.append({"seed": seed, "artifact_type": artifact_type, "filename": path.name, "path": str(path), "description": description})
    print(f"[opf-artifact] {artifact_type} seed={seed} path={path}", flush=True)


def _save_fig(base_path_without_ext: Path, seed: int, description: str) -> None:
    base_path_without_ext = _abs_path(base_path_without_ext)
    base_path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        png = base_path_without_ext.with_suffix(".png")
        print(f"[opf-save-fig] -> {png}", flush=True)
        plt.savefig(png, dpi=DPI)
        _record_artifact(seed, "figure_png", png, description)
    if SAVE_PDF:
        pdf = base_path_without_ext.with_suffix(".pdf")
        print(f"[opf-save-fig] -> {pdf}", flush=True)
        plt.savefig(pdf)
        _record_artifact(seed, "figure_pdf", pdf, description)


def _status_name(status_code: int) -> str:
    if GRB is None:
        return "GUROBI_UNAVAILABLE"
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
    }
    return mapping.get(status_code, f"STATUS_{status_code}")


def _status_category(status_code: int, feasible: bool) -> str:
    if GRB is None:
        return "skipped_gurobi_unavailable"
    if status_code == GRB.OPTIMAL and feasible:
        return "optimal_feasible"
    if status_code == GRB.OPTIMAL and not feasible:
        return "optimal_residual_violation"
    if status_code in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return "infeasible"
    if status_code == GRB.TIME_LIMIT:
        return "unknown_time_limit"
    return "unknown_other"


def _build_pcc_opf_model(case, pd_mu, qd_mu, pr_mu, qr_mu, *, output_flag=0, time_limit_sec=30, feasibility_tol=1e-6, optimality_tol=1e-6):
    """Build the same LinDistFlow/no-PCC-limit OPF constraints used by base support OPF."""
    nb, nl = case.n_bus, case.from_bus.size
    m = gp.Model()
    m.Params.OutputFlag = int(output_flag)
    m.Params.TimeLimit = float(time_limit_sec)
    m.Params.FeasibilityTol = float(feasibility_tol)
    m.Params.OptimalityTol = float(optimality_tol)
    P = m.addVars(nl, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P")
    Q = m.addVars(nl, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Q")
    V = m.addVars(nb, lb=case.vmin**2, ub=case.vmax**2, name="V")
    P0 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P0")
    Q0 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Q0")
    Pg = m.addVars(len(case.gen_buses), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Pg")
    Qg = m.addVars(len(case.gen_buses), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Qg")

    for g in range(len(case.gen_buses)):
        m.addConstr(Pg[g] >= float(case.pg_min[g]), name=f"pg_min_{g}")
        m.addConstr(Pg[g] <= float(case.pg_max[g]), name=f"pg_max_{g}")
        m.addConstr(Qg[g] >= float(case.qg_min[g]), name=f"qg_min_{g}")
        m.addConstr(Qg[g] <= float(case.qg_max[g]), name=f"qg_max_{g}")
    for l in range(nl):
        is_pcc_branch = bool(case.pcc_branch_mask[l])
        if case.line_limit_mode == "internal_only_no_pcc" and is_pcc_branch:
            continue
        m.addConstr(P[l] <= float(case.fmax_p[l]), name=f"pmax_{l}")
        m.addConstr(P[l] >= -float(case.fmax_p[l]), name=f"pmin_{l}")
        m.addConstr(Q[l] <= float(case.fmax_q[l]), name=f"qmax_{l}")
        m.addConstr(Q[l] >= -float(case.fmax_q[l]), name=f"qmin_{l}")
    root_out = case.out_branches[case.root]
    m.addConstr(gp.quicksum(P[l] for l in root_out) == P0, name="pcc_p_def")
    m.addConstr(gp.quicksum(Q[l] for l in root_out) == Q0, name="pcc_q_def")
    if case.use_explicit_pcc_limits:
        m.addConstr(P0 >= float(case.pcc_p_min), name="pcc_p_min")
        m.addConstr(P0 <= float(case.pcc_p_max), name="pcc_p_max")
        m.addConstr(Q0 >= float(case.pcc_q_min), name="pcc_q_min")
        m.addConstr(Q0 <= float(case.pcc_q_max), name="pcc_q_max")
    bus_to_gen = {int(b): g for g, b in enumerate(case.gen_buses)}
    for i in range(case.n_bus):
        if i == case.root:
            continue
        inb, outb = case.in_branches[i], case.out_branches[i]
        pg_i = Pg[bus_to_gen[i]] if i in bus_to_gen else 0.0
        qg_i = Qg[bus_to_gen[i]] if i in bus_to_gen else 0.0
        pinj = float(pr_mu[i]) - float(pd_mu[i]) + pg_i
        qinj = float(qr_mu[i]) - float(qd_mu[i]) + qg_i
        m.addConstr(gp.quicksum(P[l] for l in inb) - gp.quicksum(P[l] for l in outb) + pinj == 0, name=f"kcl_p_{i}")
        m.addConstr(gp.quicksum(Q[l] for l in inb) - gp.quicksum(Q[l] for l in outb) + qinj == 0, name=f"kcl_q_{i}")
    m.addConstr(V[case.root] == 1.0, name="v_root")
    for l in range(nl):
        i, j = int(case.from_bus[l]), int(case.to_bus[l])
        m.addConstr(V[j] == V[i] - 2.0 * (float(case.r[l]) * P[l] + float(case.x[l]) * Q[l]), name=f"lindistflow_v_{l}")
    return m, {"P": P, "Q": Q, "V": V, "P0": P0, "Q0": Q0, "Pg": Pg, "Qg": Qg}


def _extract_solution(case, var) -> Dict[str, np.ndarray | float]:
    nl = case.from_bus.size
    return {
        "P0": float(var["P0"].X),
        "Q0": float(var["Q0"].X),
        "P": np.array([var["P"][l].X for l in range(nl)], dtype=float),
        "Q": np.array([var["Q"][l].X for l in range(nl)], dtype=float),
        "V": np.array([var["V"][i].X for i in range(case.n_bus)], dtype=float),
        "Pg": np.array([var["Pg"][g].X for g in range(len(case.gen_buses))], dtype=float),
        "Qg": np.array([var["Qg"][g].X for g in range(len(case.gen_buses))], dtype=float),
    }


def compute_opf_solution_residuals(case, pd_mu, qd_mu, pr_mu, qr_mu, sol: Dict[str, np.ndarray | float], p0_target: Optional[float] = None, q0_target: Optional[float] = None) -> Dict[str, float]:
    P = np.asarray(sol["P"], dtype=float)
    Q = np.asarray(sol["Q"], dtype=float)
    V = np.asarray(sol["V"], dtype=float)
    Pg = np.asarray(sol["Pg"], dtype=float)
    Qg = np.asarray(sol["Qg"], dtype=float)
    p0 = float(sol["P0"])
    q0 = float(sol["Q0"])
    pinj = np.asarray(pr_mu, dtype=float) - np.asarray(pd_mu, dtype=float)
    qinj = np.asarray(qr_mu, dtype=float) - np.asarray(qd_mu, dtype=float)
    for g, bus in enumerate(case.gen_buses):
        pinj[int(bus)] += Pg[g]
        qinj[int(bus)] += Qg[g]
    root_out = np.asarray(case.out_branches[case.root], dtype=int)
    kcl_p, kcl_q = [], []
    for i in range(case.n_bus):
        if i == case.root:
            continue
        inb = np.asarray(case.in_branches[i], dtype=int)
        outb = np.asarray(case.out_branches[i], dtype=int)
        kcl_p.append(float(np.sum(P[inb]) - np.sum(P[outb]) + pinj[i]))
        kcl_q.append(float(np.sum(Q[inb]) - np.sum(Q[outb]) + qinj[i]))
    if case.line_limit_mode == "internal_only_no_pcc":
        line_mask = np.asarray(case.internal_branch_mask, dtype=bool)
    else:
        line_mask = np.ones_like(P, dtype=bool)
    vals = {
        "opf_pcc_p_residual": abs(float(np.sum(P[root_out]) - p0)) if p0_target is None else abs(float(p0 - p0_target)),
        "opf_pcc_q_residual": abs(float(np.sum(Q[root_out]) - q0)) if q0_target is None else abs(float(q0 - q0_target)),
        "opf_global_p_residual": abs(float(p0 + np.sum(pinj))),
        "opf_global_q_residual": abs(float(q0 + np.sum(qinj))),
        "opf_voltage_upper_violation_max": float(np.max(np.maximum(V - case.vmax**2, 0.0))) if V.size else 0.0,
        "opf_voltage_lower_violation_max": float(np.max(np.maximum(case.vmin**2 - V, 0.0))) if V.size else 0.0,
        "opf_line_p_violation_max": float(np.max(np.maximum(np.abs(P[line_mask]) - np.asarray(case.fmax_p)[line_mask], 0.0))) if np.any(line_mask) else 0.0,
        "opf_line_q_violation_max": float(np.max(np.maximum(np.abs(Q[line_mask]) - np.asarray(case.fmax_q)[line_mask], 0.0))) if np.any(line_mask) else 0.0,
        "opf_pg_upper_violation_max": float(np.max(np.maximum(Pg - np.asarray(case.pg_max), 0.0))) if Pg.size else 0.0,
        "opf_pg_lower_violation_max": float(np.max(np.maximum(np.asarray(case.pg_min) - Pg, 0.0))) if Pg.size else 0.0,
        "opf_qg_upper_violation_max": float(np.max(np.maximum(Qg - np.asarray(case.qg_max), 0.0))) if Qg.size else 0.0,
        "opf_qg_lower_violation_max": float(np.max(np.maximum(np.asarray(case.qg_min) - Qg, 0.0))) if Qg.size else 0.0,
        "opf_pv_upper_violation_max": float(np.max(np.maximum(np.asarray(pr_mu)[np.asarray(case.pv_buses, dtype=int)] - np.asarray(case.pv_pmax), 0.0))) if len(case.pv_buses) else 0.0,
        "opf_kcl_p_max_abs_residual": float(np.max(np.abs(kcl_p))) if kcl_p else 0.0,
        "opf_kcl_q_max_abs_residual": float(np.max(np.abs(kcl_q))) if kcl_q else 0.0,
    }
    vals["opf_max_constraint_residual"] = float(np.nanmax(list(vals.values())))
    return vals


def _optimize_with_inf_or_unbd_retry(m) -> int:
    m.optimize()
    if m.Status == GRB.INF_OR_UNBD:
        print("[opf] status=INF_OR_UNBD; retrying with DualReductions=0", flush=True)
        m.Params.DualReductions = 0
        m.reset()
        m.optimize()
    return int(m.Status)


def solve_pcc_fixed_opf_feasibility(case, pd_mu, qd_mu, pr_mu, qr_mu, p0_target, q0_target, *, time_limit_sec=30, feasibility_tol=1e-6, output_flag=0) -> Dict[str, object]:
    if not GUROBI_IMPORT_AVAILABLE:
        return {"opf_status": "GUROBI_UNAVAILABLE", "opf_status_code": -1, "opf_status_category": "skipped_gurobi_unavailable", "opf_feasible": False, "failure_reason": GUROBI_IMPORT_ERROR}
    t0 = time.perf_counter()
    try:
        m, var = _build_pcc_opf_model(case, pd_mu, qd_mu, pr_mu, qr_mu, output_flag=output_flag, time_limit_sec=time_limit_sec, feasibility_tol=feasibility_tol, optimality_tol=OPF_OPTIMALITY_TOL)
        m.addConstr(var["P0"] == float(p0_target), name="fixed_p0_target")
        m.addConstr(var["Q0"] == float(q0_target), name="fixed_q0_target")
        m.setObjective(0.0, GRB.MINIMIZE)
        status = _optimize_with_inf_or_unbd_retry(m)
        elapsed = time.perf_counter() - t0
        result = {"opf_status": _status_name(status), "opf_status_code": status, "opf_solve_time_sec": elapsed, "opf_obj_value": np.nan, "failure_reason": "", "warning_message": ""}
        feasible = False
        sol = None
        if status == GRB.OPTIMAL:
            result["opf_obj_value"] = float(m.ObjVal)
            sol = _extract_solution(case, var)
            residuals = compute_opf_solution_residuals(case, pd_mu, qd_mu, pr_mu, qr_mu, sol, p0_target, q0_target)
            feasible = bool(residuals["opf_max_constraint_residual"] <= max(10.0 * feasibility_tol, feasibility_tol))
            result.update(residuals)
        else:
            result.update({k: np.nan for k in DETAIL_COLUMNS if k.startswith("opf_") and k not in result})
        result["opf_feasible"] = feasible
        result["opf_status_category"] = _status_category(status, feasible)
        result["solution"] = sol
        return result
    except Exception as exc:  # noqa: BLE001
        return {"opf_status": "EXCEPTION", "opf_status_code": -2, "opf_status_category": "exception", "opf_feasible": False, "opf_solve_time_sec": time.perf_counter() - t0, "failure_reason": repr(exc), "warning_message": "strict feasibility solver exception"}


def solve_nearest_feasible_pq_projection(case, pd_mu, qd_mu, pr_mu, qr_mu, p0_target, q0_target, *, objective="l1_pq") -> Dict[str, object]:
    if not GUROBI_IMPORT_AVAILABLE:
        return {"projection_status": "GUROBI_UNAVAILABLE", "projection_feasible": False, "projection_solve_time_sec": 0.0}
    t0 = time.perf_counter()
    try:
        m, var = _build_pcc_opf_model(case, pd_mu, qd_mu, pr_mu, qr_mu, output_flag=OPF_OUTPUT_FLAG, time_limit_sec=OPF_TIME_LIMIT_SEC, feasibility_tol=OPF_FEASIBILITY_TOL, optimality_tol=OPF_OPTIMALITY_TOL)
        dP_pos = m.addVar(lb=0.0, name="dP_pos")
        dP_neg = m.addVar(lb=0.0, name="dP_neg")
        dQ_pos = m.addVar(lb=0.0, name="dQ_pos")
        dQ_neg = m.addVar(lb=0.0, name="dQ_neg")
        m.addConstr(var["P0"] - float(p0_target) == dP_pos - dP_neg, name="projection_p_abs")
        m.addConstr(var["Q0"] - float(q0_target) == dQ_pos - dQ_neg, name="projection_q_abs")
        if objective != "l1_pq":
            print(f"[opf-warning] unsupported PROJECTION_OBJECTIVE={objective!r}; using l1_pq", flush=True)
        m.setObjective((dP_pos + dP_neg) / max(PROJECTION_P_SCALE, 1e-12) + (dQ_pos + dQ_neg) / max(PROJECTION_Q_SCALE, 1e-12), GRB.MINIMIZE)
        status = _optimize_with_inf_or_unbd_retry(m)
        elapsed = time.perf_counter() - t0
        out = {"projection_status": _status_name(status), "projection_solve_time_sec": elapsed, "projection_feasible": False, "p0_projected": np.nan, "q0_projected": np.nan, "delta_p0_projected": np.nan, "delta_q0_projected": np.nan, "projection_l1_distance": np.nan, "projection_l2_distance": np.nan, "projection_solution": None}
        if status == GRB.OPTIMAL:
            sol = _extract_solution(case, var)
            p0p, q0p = float(sol["P0"]), float(sol["Q0"])
            dp, dq = p0p - float(p0_target), q0p - float(q0_target)
            out.update({"projection_feasible": True, "p0_projected": p0p, "q0_projected": q0p, "delta_p0_projected": dp, "delta_q0_projected": dq, "projection_l1_distance": abs(dp) / max(PROJECTION_P_SCALE, 1e-12) + abs(dq) / max(PROJECTION_Q_SCALE, 1e-12), "projection_l2_distance": math.sqrt(dp * dp + dq * dq), "projection_solution": sol})
        return out
    except Exception as exc:  # noqa: BLE001
        return {"projection_status": "EXCEPTION", "projection_feasible": False, "projection_solve_time_sec": time.perf_counter() - t0, "failure_reason": repr(exc)}


def load_source_load_context(seed: int, case) -> Tuple[Dict[str, object], str]:
    seed_dir = _seed_dir(seed)
    cache_path = seed_dir / f"selected_seed_{_seed_tag(seed)}_mc_supports.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        keys = set(z.files)
        if {"x_mu", "pd_mu", "qd_mu", "pr_mu", "qr_mu"}.issubset(keys):
            print(f"[opf] seed={seed:04d} loaded source/load context from cache {cache_path}", flush=True)
            return {k: z[k] for k in z.files}, "loaded_support_cache"
    print(f"[opf-warning] seed={seed:04d} support cache missing/incomplete; drawing fixed q50 source/load context without MC-OPF", flush=True)
    pd_mu, qd_mu, pr_mu, qr_mu = base.draw_external_scenario_by_seed(case, seed)
    x_mu = base.make_feature_vector(case, pd_mu, pr_mu)
    return {"seed": np.array(seed), "x_mu": x_mu, "pd_mu": pd_mu, "qd_mu": qd_mu, "pr_mu": pr_mu, "qr_mu": qr_mu}, "drawn_from_seed_no_mc_opf"


def aggregate_nn_detail(seed: int) -> pd.DataFrame:
    path = _seed_dir(seed) / f"selected_seed_{_seed_tag(seed)}_posterior_q50_vertex_phys_residual_detail.csv"
    if not path.exists():
        print(f"[opf-warning] seed={seed:04d} NN residual detail missing: {path}", flush=True)
        return pd.DataFrame(columns=["sample_idx", "vertex_idx", "nn_max_phys_violation", "nn_t_abs_error", "nn_p0_forced_recovery_abs_error", "nn_q0_forced_recovery_abs_error", "nn_check_mode"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["sample_idx", "vertex_idx", "nn_max_phys_violation", "nn_t_abs_error", "nn_p0_forced_recovery_abs_error", "nn_q0_forced_recovery_abs_error", "nn_check_mode"])
    rows = []
    for (sample_idx, vertex_idx), g in df.groupby(["sample_idx", "vertex_idx"]):
        rows.append({
            "sample_idx": int(sample_idx),
            "vertex_idx": int(vertex_idx),
            "nn_max_phys_violation": float(pd.to_numeric(g.get("max_phys_violation"), errors="coerce").max()),
            "nn_t_abs_error": float(pd.to_numeric(g.get("t_abs_error"), errors="coerce").max()),
            "nn_p0_forced_recovery_abs_error": float(pd.to_numeric(g.get("p0_forced_recovery_abs_error"), errors="coerce").max()),
            "nn_q0_forced_recovery_abs_error": float(pd.to_numeric(g.get("q0_forced_recovery_abs_error"), errors="coerce").max()),
            "nn_check_mode": ";".join(sorted(set(str(x) for x in g.get("check_mode", pd.Series(dtype=str)).dropna().values))),
        })
    return pd.DataFrame(rows)


def _polygon_rows_from_poly(seed: int, source_polygon: str, sample_idx: int, poly: np.ndarray, polygon_area: float, max_halfspace_violation: float) -> List[dict]:
    rows = []
    for vertex_idx, (p0, q0) in enumerate(np.asarray(poly, dtype=float)):
        rows.append({"seed": seed, "source_polygon": source_polygon, "sample_idx": sample_idx, "vertex_idx": vertex_idx, "P0": float(p0), "Q0": float(q0), "polygon_area": polygon_area, "max_halfspace_violation": max_halfspace_violation, "n_vertices": int(len(poly))})
    return rows


def load_or_generate_posterior_vertices(seed: int, case, context: Dict[str, object], model_map: Dict[int, dict]) -> Tuple[pd.DataFrame, str]:
    path = _seed_dir(seed) / f"selected_seed_{_seed_tag(seed)}_posterior_q50_polygon_vertices.csv"
    if path.exists() and OPF_VALIDATE_POSTERIOR_Q50_VERTICES:
        df = pd.read_csv(path)
        if not df.empty:
            df = df.copy()
            df.insert(1, "source_polygon", "posterior_q50_bpinN")
            print(f"[opf] seed={seed:04d} loaded posterior vertices {path} rows={len(df)}", flush=True)
            return df, "loaded_v7464_vertices"
    if not OPF_VALIDATE_POSTERIOR_Q50_VERTICES:
        return pd.DataFrame(), "disabled"
    missing = [j for j in range(N_THETA) if model_map.get(j, {}).get("net") is None or model_map.get(j, {}).get("norm") is None]
    if missing:
        return pd.DataFrame(), f"posterior vertices missing and all-theta models unavailable: {missing}"
    rows = []
    x_mu = np.asarray(context["x_mu"], dtype=float).reshape(1, -1)
    for sample_idx in range(int(getattr(phys64, "N_BNN_DOMAIN_PHYS_SAMPLES", 50))):
        h_values, _ = phys64._posterior_sample_q50_supports(seed, sample_idx, x_mu, model_map)
        poly, info = phys64.strict_halfspace_polygon_vertices(THETA_VALUES, h_values)
        if info.get("status") != "ok":
            print(f"[opf-warning] seed={seed:04d} regenerated posterior sample={sample_idx:03d} polygon failed: {info}", flush=True)
            continue
        rows.extend(_polygon_rows_from_poly(seed, "posterior_q50_bpinN", sample_idx, poly, phys64.polygon_area(poly), float(info.get("max_halfspace_violation", np.nan))))
    return pd.DataFrame(rows), "regenerated_from_existing_models"


def build_deterministic_vertices(seed: int, context: Dict[str, object], model_map: Dict[int, dict]) -> Tuple[pd.DataFrame, str]:
    if not OPF_VALIDATE_DETERMINISTIC_BPINN_Q50_VERTICES:
        return pd.DataFrame(), "disabled"
    missing = [j for j in range(N_THETA) if model_map.get(j, {}).get("net") is None or model_map.get(j, {}).get("norm") is None]
    if missing:
        return pd.DataFrame(), f"all-theta models unavailable: {missing}"
    h = phys64._predict_deterministic_q50_for_seed(seed, np.asarray(context["x_mu"], dtype=float).reshape(1, -1), model_map)
    poly, info = phys64.strict_halfspace_polygon_vertices(THETA_VALUES, h)
    if info.get("status") != "ok":
        return pd.DataFrame(), f"strict deterministic polygon failed: {info}"
    return pd.DataFrame(_polygon_rows_from_poly(seed, "deterministic_q50_bpinN", -1, poly, phys64.polygon_area(poly), float(info.get("max_halfspace_violation", np.nan)))), "ok"


def build_mc_q50_vertices(seed: int, context: Dict[str, object]) -> Tuple[pd.DataFrame, str, str]:
    if not OPF_VALIDATE_MC_Q50_VERTICES_IF_AVAILABLE:
        return pd.DataFrame(), "disabled", "disabled by config"
    if "H_MC" not in context:
        return pd.DataFrame(), "skipped", "mc support cache missing or keys not found"
    H = np.asarray(context["H_MC"], dtype=float)
    if H.ndim != 2 or H.shape[1] < N_THETA:
        return pd.DataFrame(), "skipped", "mc support cache missing or keys not found"
    if "success_mask" in context:
        success = np.asarray(context["success_mask"], dtype=bool)
        Hq = np.where(success, H, np.nan)
    else:
        Hq = H
    h = np.nanquantile(Hq[:, :N_THETA], SOURCE_LOAD_QUANTILE_FOR_OPF, axis=0)
    poly, info = phys64.strict_halfspace_polygon_vertices(THETA_VALUES, h)
    if info.get("status") != "ok":
        return pd.DataFrame(), "skipped", f"strict MC q50 polygon failed: {info}"
    return pd.DataFrame(_polygon_rows_from_poly(seed, "mc_q50_reference", -1, poly, phys64.polygon_area(poly), float(info.get("max_halfspace_violation", np.nan)))), "ok", ""


def _q(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanquantile(arr, q)) if arr.size else np.nan


def _corr(x, y) -> float:
    xx = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    yy = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    if int(mask.sum()) < 2:
        return np.nan
    return float(np.corrcoef(xx[mask], yy[mask])[0, 1])


def summarize_detail(seed: int, detail_df: pd.DataFrame, mc_status: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if detail_df.empty:
        sample_df = pd.DataFrame(columns=["seed", "source_polygon", "sample_idx", "n_vertices", "n_opf_feasible", "opf_feasible_rate", "all_vertices_opf_feasible", "max_projection_l1_distance", "q50_projection_l1_distance", "q95_projection_l1_distance", "max_projection_l2_distance", "q50_projection_l2_distance", "q95_projection_l2_distance", "mean_nn_max_phys_violation", "max_nn_max_phys_violation"])
        polygon_df = pd.DataFrame(columns=["seed", "source_polygon", "n_polygons_or_samples", "n_vertices", "n_opf_feasible", "opf_feasible_rate", "n_all_vertices_feasible_samples", "all_vertices_feasible_sample_rate", "max_projection_l1_distance", "q50_projection_l1_distance", "q95_projection_l1_distance", "max_projection_l2_distance", "q50_projection_l2_distance", "q95_projection_l2_distance"])
        summary_df = pd.DataFrame([{"seed": seed, "n_total_vertices_checked": 0, "n_total_opf_feasible": 0, "overall_opf_feasible_rate": np.nan, "posterior_q50_n_vertices": 0, "posterior_q50_opf_feasible_rate": np.nan, "posterior_q50_all_vertices_feasible_sample_rate": np.nan, "deterministic_q50_n_vertices": 0, "deterministic_q50_opf_feasible_rate": np.nan, "mc_q50_reference_n_vertices": 0, "mc_q50_reference_opf_feasible_rate": np.nan, "mc_q50_reference_status": mc_status, "worst_source_polygon": "", "worst_sample_idx": -1, "worst_vertex_idx": -1, "worst_P0": np.nan, "worst_Q0": np.nan, "worst_projection_l1_distance": np.nan, "worst_projection_l2_distance": np.nan, "nn_residual_vs_opf_projection_corr": np.nan, "nn_t_error_vs_opf_projection_corr": np.nan}])
        return sample_df, polygon_df, summary_df
    sample_rows = []
    for (src, sid), g in detail_df.groupby(["source_polygon", "sample_idx"]):
        l1 = pd.to_numeric(g["projection_l1_distance"], errors="coerce")
        l2 = pd.to_numeric(g["projection_l2_distance"], errors="coerce")
        feasible = g["opf_feasible"].astype(bool)
        sample_rows.append({"seed": seed, "source_polygon": src, "sample_idx": int(sid), "n_vertices": int(len(g)), "n_opf_feasible": int(feasible.sum()), "opf_feasible_rate": float(feasible.mean()), "all_vertices_opf_feasible": bool(feasible.all()), "max_projection_l1_distance": _q(l1, 1.0), "q50_projection_l1_distance": _q(l1, 0.50), "q95_projection_l1_distance": _q(l1, 0.95), "max_projection_l2_distance": _q(l2, 1.0), "q50_projection_l2_distance": _q(l2, 0.50), "q95_projection_l2_distance": _q(l2, 0.95), "mean_nn_max_phys_violation": _q(pd.to_numeric(g["nn_max_phys_violation"], errors="coerce"), 0.50), "max_nn_max_phys_violation": _q(pd.to_numeric(g["nn_max_phys_violation"], errors="coerce"), 1.0)})
    sample_df = pd.DataFrame(sample_rows)
    polygon_rows = []
    for src, g in detail_df.groupby("source_polygon"):
        feasible = g["opf_feasible"].astype(bool)
        sdf = sample_df[sample_df["source_polygon"] == src]
        polygon_rows.append({"seed": seed, "source_polygon": src, "n_polygons_or_samples": int(g["sample_idx"].nunique()), "n_vertices": int(len(g)), "n_opf_feasible": int(feasible.sum()), "opf_feasible_rate": float(feasible.mean()), "n_all_vertices_feasible_samples": int(sdf["all_vertices_opf_feasible"].sum()) if not sdf.empty else 0, "all_vertices_feasible_sample_rate": float(sdf["all_vertices_opf_feasible"].mean()) if not sdf.empty else np.nan, "max_projection_l1_distance": _q(g["projection_l1_distance"], 1.0), "q50_projection_l1_distance": _q(g["projection_l1_distance"], 0.50), "q95_projection_l1_distance": _q(g["projection_l1_distance"], 0.95), "max_projection_l2_distance": _q(g["projection_l2_distance"], 1.0), "q50_projection_l2_distance": _q(g["projection_l2_distance"], 0.50), "q95_projection_l2_distance": _q(g["projection_l2_distance"], 0.95)})
    polygon_df = pd.DataFrame(polygon_rows)
    worst_metric = pd.to_numeric(detail_df["projection_l1_distance"], errors="coerce").fillna(-np.inf)
    worst_idx = int(worst_metric.idxmax()) if len(worst_metric) else None
    worst = detail_df.loc[worst_idx] if worst_idx is not None and np.isfinite(worst_metric.loc[worst_idx]) else pd.Series(dtype=object)
    def rate(src):
        g = detail_df[detail_df["source_polygon"] == src]
        return float(g["opf_feasible"].astype(bool).mean()) if not g.empty else np.nan
    def nverts(src):
        return int((detail_df["source_polygon"] == src).sum())
    post_samples = sample_df[sample_df["source_polygon"] == "posterior_q50_bpinN"]
    summary = {"seed": seed, "n_total_vertices_checked": int(len(detail_df)), "n_total_opf_feasible": int(detail_df["opf_feasible"].astype(bool).sum()), "overall_opf_feasible_rate": float(detail_df["opf_feasible"].astype(bool).mean()), "posterior_q50_n_vertices": nverts("posterior_q50_bpinN"), "posterior_q50_opf_feasible_rate": rate("posterior_q50_bpinN"), "posterior_q50_all_vertices_feasible_sample_rate": float(post_samples["all_vertices_opf_feasible"].mean()) if not post_samples.empty else np.nan, "deterministic_q50_n_vertices": nverts("deterministic_q50_bpinN"), "deterministic_q50_opf_feasible_rate": rate("deterministic_q50_bpinN"), "mc_q50_reference_n_vertices": nverts("mc_q50_reference"), "mc_q50_reference_opf_feasible_rate": rate("mc_q50_reference"), "mc_q50_reference_status": mc_status, "worst_source_polygon": str(worst.get("source_polygon", "")), "worst_sample_idx": int(worst.get("sample_idx", -1)) if not worst.empty else -1, "worst_vertex_idx": int(worst.get("vertex_idx", -1)) if not worst.empty else -1, "worst_P0": float(worst.get("P0", np.nan)) if not worst.empty else np.nan, "worst_Q0": float(worst.get("Q0", np.nan)) if not worst.empty else np.nan, "worst_projection_l1_distance": float(worst.get("projection_l1_distance", np.nan)) if not worst.empty else np.nan, "worst_projection_l2_distance": float(worst.get("projection_l2_distance", np.nan)) if not worst.empty else np.nan, "nn_residual_vs_opf_projection_corr": _corr(detail_df["nn_max_phys_violation"], detail_df["projection_l1_distance"]), "nn_t_error_vs_opf_projection_corr": _corr(detail_df["nn_t_abs_error"], detail_df["projection_l1_distance"])}
    return sample_df, polygon_df, pd.DataFrame([summary])


def plot_results(seed: int, detail_df: pd.DataFrame, all_vertices_df: pd.DataFrame) -> None:
    seed_dir = _seed_dir(seed)
    tag = _seed_tag(seed)
    plt.figure(figsize=(7.4, 6.5))
    for (src, sid), g in all_vertices_df.groupby(["source_polygon", "sample_idx"]):
        if len(g) < 3:
            continue
        pts = g.sort_values("vertex_idx")[["P0", "Q0"]].to_numpy(float)
        pts = np.vstack([pts, pts[0]])
        if src == "posterior_q50_bpinN":
            plt.plot(pts[:, 0], pts[:, 1], color="#94a3b8", linewidth=0.6, alpha=0.25, label="posterior q50 BPINN polygons" if sid == all_vertices_df[all_vertices_df.source_polygon == src]["sample_idx"].min() else None)
        elif src == "deterministic_q50_bpinN":
            plt.plot(pts[:, 0], pts[:, 1], color="#f97316", linewidth=2.0, linestyle="--", label="deterministic q50 BPINN polygon")
        elif src == "mc_q50_reference":
            plt.plot(pts[:, 0], pts[:, 1], color="#2563eb", linewidth=2.0, label="MC q50 reference polygon")
    if not detail_df.empty:
        feas = detail_df[detail_df["opf_feasible"].astype(bool)]
        infeas = detail_df[~detail_df["opf_feasible"].astype(bool)]
        if not feas.empty:
            plt.scatter(feas["P0"], feas["Q0"], marker="o", s=28, color="#16a34a", alpha=0.75, label="OPF feasible vertex")
        if not infeas.empty:
            l1 = pd.to_numeric(infeas["projection_l1_distance"], errors="coerce").fillna(0.0).to_numpy(float)
            sizes = 35.0 + 120.0 * (l1 / max(float(np.nanmax(l1)) if l1.size else 0.0, 1e-12))
            plt.scatter(infeas["P0"], infeas["Q0"], marker="x", s=sizes, color="#dc2626", alpha=0.85, label="OPF infeasible vertex")
            worst = infeas.iloc[int(np.nanargmax(l1))] if l1.size else infeas.iloc[0]
            plt.scatter([worst["P0"]], [worst["Q0"]], marker="*", s=240, color="#7f1d1d", edgecolors="black", linewidths=0.5, label="worst projection vertex")
    plt.xlabel("P_0")
    plt.ylabel("Q_0")
    plt.title(f"Seed {seed}: vertex OPF feasibility")
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    _save_fig(seed_dir / f"selected_seed_{tag}_vertex_opf_feasibility_overlay", seed, "Vertex OPF feasibility overlay")
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    vals = pd.to_numeric(detail_df.get("projection_l1_distance", pd.Series(dtype=float)), errors="coerce").dropna()
    vals = vals[vals > 0]
    if len(vals):
        plt.hist(vals, bins=min(30, max(5, int(np.sqrt(len(vals))))), color="#2563eb", alpha=0.8)
        plt.xlabel("projection_l1_distance")
    else:
        plt.text(0.5, 0.5, "No infeasible/projection distances", ha="center", va="center", transform=plt.gca().transAxes)
    plt.title(f"Seed {seed}: OPF projection distance distribution")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    _save_fig(seed_dir / f"selected_seed_{tag}_vertex_opf_projection_distance_distribution", seed, "Projection distance distribution")
    plt.close()

    plt.figure(figsize=(6.2, 5.0))
    x = pd.to_numeric(detail_df.get("nn_max_phys_violation", pd.Series(dtype=float)), errors="coerce")
    y = pd.to_numeric(detail_df.get("projection_l1_distance", pd.Series(dtype=float)), errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.any():
        plt.scatter(x[mask], y[mask], color="#7c3aed", alpha=0.7, s=28)
        plt.xlabel("nn_max_phys_violation")
        plt.ylabel("projection_l1_distance")
    else:
        plt.text(0.5, 0.5, "No paired NN residual / OPF projection data", ha="center", va="center", transform=plt.gca().transAxes)
    plt.title(f"Seed {seed}: NN residual vs OPF projection")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    _save_fig(seed_dir / f"selected_seed_{tag}_nn_residual_vs_opf_projection", seed, "NN residual versus OPF projection scatter")
    plt.close()


def write_manifest(seed: int, manifest: Dict[str, object]) -> Path:
    path = _seed_dir(seed) / f"selected_seed_{_seed_tag(seed)}_vertex_opf_feasibility_manifest.json"
    print(f"[opf-save-json] manifest -> {path}", flush=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _record_artifact(seed, "json", path, "Vertex OPF feasibility manifest")
    return path


def write_empty_seed_outputs(seed: int, reason: str) -> Dict[str, object]:
    seed_dir = _seed_dir(seed)
    tag = _seed_tag(seed)
    detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)
    sample_df, polygon_df, summary_df = summarize_detail(seed, detail_df, "skipped")
    for df, name, desc in [
        (detail_df, f"selected_seed_{tag}_vertex_opf_feasibility_detail.csv", "empty OPF detail"),
        (sample_df, f"selected_seed_{tag}_vertex_opf_feasibility_by_sample.csv", "empty OPF sample summary"),
        (polygon_df, f"selected_seed_{tag}_vertex_opf_feasibility_by_polygon_type.csv", "empty OPF polygon-type summary"),
        (summary_df, f"selected_seed_{tag}_vertex_opf_feasibility_summary.csv", "empty OPF global summary"),
    ]:
        p = seed_dir / name
        _safe_to_csv(df, p, description=f"{desc}: {reason}")
        _record_artifact(seed, "csv", p, desc)
    manifest = base_manifest(seed)
    manifest.update({"gurobi_available": GUROBI_IMPORT_AVAILABLE, "gurobi_import_error": GUROBI_IMPORT_ERROR, "skip_reason": reason, "outputs": _manifest_artifacts})
    write_manifest(seed, manifest)
    return summary_df.iloc[0].to_dict()


def base_manifest(seed: int) -> Dict[str, object]:
    return {
        "new_file_name": NEW_SCRIPT_NAME,
        "base_file_name": BASE_FILE_NAME,
        "previous_phys_residual_file": PREVIOUS_PHYS_RESIDUAL_FILE,
        "selected_seeds": SELECTED_EXTERNAL_SEEDS,
        "RUN_VERTEX_OPF_FEASIBILITY": RUN_VERTEX_OPF_FEASIBILITY,
        "OPF_VALIDATE_POSTERIOR_Q50_VERTICES": OPF_VALIDATE_POSTERIOR_Q50_VERTICES,
        "OPF_VALIDATE_DETERMINISTIC_BPINN_Q50_VERTICES": OPF_VALIDATE_DETERMINISTIC_BPINN_Q50_VERTICES,
        "OPF_VALIDATE_MC_Q50_VERTICES_IF_AVAILABLE": OPF_VALIDATE_MC_Q50_VERTICES_IF_AVAILABLE,
        "OPF_VALIDATE_MODE": OPF_VALIDATE_MODE,
        "OPF_MAX_VERTICES_PER_SEED": OPF_MAX_VERTICES_PER_SEED,
        "OPF_TIME_LIMIT_SEC": OPF_TIME_LIMIT_SEC,
        "OPF_FEASIBILITY_TOL": OPF_FEASIBILITY_TOL,
        "RUN_PROJECTION_IF_INFEASIBLE": RUN_PROJECTION_IF_INFEASIBLE,
        "PROJECTION_OBJECTIVE": PROJECTION_OBJECTIVE,
        "retrained_model": False,
        "rebuilt_dataset": False,
        "reran_alltheta_mc2500": False,
        "reran_mc_opf_supports": False,
        "used_nn_recovery_as_feasibility_certificate": False,
        "opf_feasibility_check": True,
        "projection_check": bool(RUN_PROJECTION_IF_INFEASIBLE),
        "gurobi_available": GUROBI_IMPORT_AVAILABLE,
    }


def run_seed(seed: int, case, model_map: Dict[int, dict]) -> Dict[str, object]:
    seed_dir = _seed_dir(seed)
    tag = _seed_tag(seed)
    context, context_status = load_source_load_context(seed, case)
    pd_mu, qd_mu, pr_mu, qr_mu = [np.asarray(context[k], dtype=float) for k in ["pd_mu", "qd_mu", "pr_mu", "qr_mu"]]
    nn_df = aggregate_nn_detail(seed)

    posterior_df, posterior_status = load_or_generate_posterior_vertices(seed, case, context, model_map)
    det_df, det_status = build_deterministic_vertices(seed, context, model_map)
    mc_df, mc_status, mc_skip_reason = build_mc_q50_vertices(seed, context)
    all_vertices = pd.concat([df for df in [posterior_df, det_df, mc_df] if df is not None and not df.empty], ignore_index=True) if any(not df.empty for df in [posterior_df, det_df, mc_df]) else pd.DataFrame()
    if all_vertices.empty:
        return write_empty_seed_outputs(seed, f"no vertices to validate; posterior_status={posterior_status}; deterministic_status={det_status}; mc_status={mc_status}; mc_skip_reason={mc_skip_reason}")
    all_vertices = all_vertices.drop_duplicates(subset=["source_polygon", "sample_idx", "vertex_idx"]).reset_index(drop=True)
    if OPF_MAX_VERTICES_PER_SEED is not None:
        all_vertices = all_vertices.head(int(OPF_MAX_VERTICES_PER_SEED)).copy()

    if not nn_df.empty:
        all_vertices = all_vertices.merge(nn_df, on=["sample_idx", "vertex_idx"], how="left")
    else:
        for c in ["nn_max_phys_violation", "nn_t_abs_error", "nn_p0_forced_recovery_abs_error", "nn_q0_forced_recovery_abs_error", "nn_check_mode"]:
            all_vertices[c] = np.nan if c != "nn_check_mode" else ""

    detail_rows: List[dict] = []
    solutions: List[dict] = []
    total = len(all_vertices)
    for k, row in all_vertices.iterrows():
        p0, q0 = float(row["P0"]), float(row["Q0"])
        print(f"[opf] seed={tag} source={row['source_polygon']} vertex={k+1:03d}/{total:03d} P0={p0:.6f} Q0={q0:.6f}", flush=True)
        strict = solve_pcc_fixed_opf_feasibility(case, pd_mu, qd_mu, pr_mu, qr_mu, p0, q0, time_limit_sec=OPF_TIME_LIMIT_SEC, feasibility_tol=OPF_FEASIBILITY_TOL, output_flag=OPF_OUTPUT_FLAG)
        print(f"[opf] strict feasibility status={strict.get('opf_status')} feasible={strict.get('opf_feasible')} solve_time={strict.get('opf_solve_time_sec', np.nan):.3f}", flush=True)
        proj = {"projection_status": "not_run", "projection_feasible": False, "projection_solve_time_sec": np.nan, "p0_projected": np.nan, "q0_projected": np.nan, "delta_p0_projected": np.nan, "delta_q0_projected": np.nan, "projection_l1_distance": 0.0 if strict.get("opf_feasible") else np.nan, "projection_l2_distance": 0.0 if strict.get("opf_feasible") else np.nan}
        if (not bool(strict.get("opf_feasible", False))) and RUN_PROJECTION_IF_INFEASIBLE and GUROBI_IMPORT_AVAILABLE:
            print("[opf] infeasible; running nearest feasible projection...", flush=True)
            proj = solve_nearest_feasible_pq_projection(case, pd_mu, qd_mu, pr_mu, qr_mu, p0, q0, objective=PROJECTION_OBJECTIVE)
            print(f"[opf] projection distance l1={proj.get('projection_l1_distance', np.nan)} l2={proj.get('projection_l2_distance', np.nan)}", flush=True)
        out = {col: row.get(col, np.nan) for col in ["seed", "source_polygon", "sample_idx", "vertex_idx", "P0", "Q0", "polygon_area", "max_halfspace_violation", "n_vertices", "nn_max_phys_violation", "nn_t_abs_error", "nn_p0_forced_recovery_abs_error", "nn_q0_forced_recovery_abs_error", "nn_check_mode"]}
        out.update({c: np.nan for c in DETAIL_COLUMNS if c not in out})
        out.update({kk: vv for kk, vv in strict.items() if kk != "solution"})
        out.update({kk: vv for kk, vv in proj.items() if kk != "projection_solution"})
        out["failure_reason"] = "; ".join(str(x) for x in [strict.get("failure_reason", ""), proj.get("failure_reason", "")] if str(x))
        detail_rows.append(out)
        sol = strict.get("solution") if strict.get("opf_feasible") else proj.get("projection_solution")
        if sol is not None:
            solutions.append({"solution_index": len(solutions), "source_polygon": str(row["source_polygon"]), "sample_idx": int(row["sample_idx"]), "vertex_idx": int(row["vertex_idx"]), "P0": p0, "Q0": q0, "PG": sol["Pg"], "QG": sol["Qg"], "P": sol["P"], "Q": sol["Q"], "V": sol["V"]})
    detail_df = pd.DataFrame(detail_rows)
    for col in DETAIL_COLUMNS:
        if col not in detail_df.columns:
            detail_df[col] = np.nan
    detail_df = detail_df[DETAIL_COLUMNS]
    sample_df, polygon_df, summary_df = summarize_detail(seed, detail_df, mc_status)

    outputs = [
        (detail_df, seed_dir / f"selected_seed_{tag}_vertex_opf_feasibility_detail.csv", "OPF vertex feasibility detail"),
        (sample_df, seed_dir / f"selected_seed_{tag}_vertex_opf_feasibility_by_sample.csv", "OPF feasibility summary by posterior sample"),
        (polygon_df, seed_dir / f"selected_seed_{tag}_vertex_opf_feasibility_by_polygon_type.csv", "OPF feasibility summary by polygon type"),
        (summary_df, seed_dir / f"selected_seed_{tag}_vertex_opf_feasibility_summary.csv", "OPF feasibility global summary"),
    ]
    if SAVE_OPF_DETAIL or SAVE_OPF_SUMMARY:
        for df, path, desc in outputs:
            _safe_to_csv(df, path, description=desc)
            _record_artifact(seed, "csv", path, desc)
    if SAVE_OPF_SOLUTIONS_NPZ:
        sol_path = seed_dir / f"selected_seed_{tag}_vertex_opf_feasibility_solutions.npz"
        print(f"[opf-save-npz] -> {sol_path}", flush=True)
        np.savez_compressed(sol_path, solutions=np.array(solutions, dtype=object), allow_pickle=True)
        _record_artifact(seed, "npz", sol_path, "Feasible/projected OPF dispatch solutions; object array of dicts")
    if PLOT_OPF_FEASIBILITY_RESULTS:
        plot_results(seed, detail_df, all_vertices)
    counts = detail_df.groupby("source_polygon").size().to_dict() if not detail_df.empty else {}
    manifest = base_manifest(seed)
    manifest.update({"source_load_context_status": context_status, "posterior_vertices_status": posterior_status, "deterministic_q50_status": det_status, "mc_q50_reference_status": mc_status, "mc_q50_reference_skipped_reason": mc_skip_reason, "number_of_vertices_checked_by_source_polygon": {str(k): int(v) for k, v in counts.items()}, "outputs": _manifest_artifacts, "solutions_npz_note": "Stored as an object array of per-vertex dictionaries when SAVE_OPF_SOLUTIONS_NPZ=True."})
    write_manifest(seed, manifest)
    for _, srow in sample_df.iterrows():
        print(f"[opf] sample={int(srow['sample_idx']):03d} source={srow['source_polygon']} feasible_rate={srow['opf_feasible_rate']:.4f} max_projection_l1={srow['max_projection_l1_distance']}", flush=True)
    for _, prow in polygon_df.iterrows():
        print(f"[opf] source_polygon={prow['source_polygon']} feasible_rate={prow['opf_feasible_rate']:.4f}", flush=True)
    return summary_df.iloc[0].to_dict()


def main() -> None:
    print("[v7.4.6.5-vertex-opf-feasibility] start", flush=True)
    print(f"[opf-config] RUN_VERTEX_OPF_FEASIBILITY={RUN_VERTEX_OPF_FEASIBILITY}", flush=True)
    print(f"[opf-config] OPF_VALIDATE_POSTERIOR_Q50_VERTICES={OPF_VALIDATE_POSTERIOR_Q50_VERTICES}", flush=True)
    print(f"[opf-config] OPF_VALIDATE_DETERMINISTIC_BPINN_Q50_VERTICES={OPF_VALIDATE_DETERMINISTIC_BPINN_Q50_VERTICES}", flush=True)
    print(f"[opf-config] OPF_VALIDATE_MC_Q50_VERTICES_IF_AVAILABLE={OPF_VALIDATE_MC_Q50_VERTICES_IF_AVAILABLE}", flush=True)
    print(f"[opf-config] OPF_TIME_LIMIT_SEC={OPF_TIME_LIMIT_SEC} OPF_FEASIBILITY_TOL={OPF_FEASIBILITY_TOL}", flush=True)
    print(f"[opf-config] RUN_PROJECTION_IF_INFEASIBLE={RUN_PROJECTION_IF_INFEASIBLE} PROJECTION_OBJECTIVE={PROJECTION_OBJECTIVE}", flush=True)
    if not RUN_VERTEX_OPF_FEASIBILITY:
        print("[opf] disabled; exiting", flush=True)
        return
    if not GUROBI_IMPORT_AVAILABLE:
        print(f"[opf-warning] gurobipy/base unavailable; writing skip manifests. error={GUROBI_IMPORT_ERROR}", flush=True)
        summaries = [write_empty_seed_outputs(seed, "skipped because gurobipy unavailable or base import failed") for seed in SELECTED_EXTERNAL_SEEDS]
    else:
        out_dir = _selected_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_ranking = phys64.safe_load_seed_ranking()
        selected_seeds = v7463.select_external_seeds(seed_ranking)
        print("[opf-setup] building IEEE-33 no-PCC case", flush=True)
        case = base.build_ieee33_case()
        base.ALL_THETA_LIST = list(range(N_THETA))
        base.ALL_THETA_RESULT_DIR = v7463.ALL_THETA_RESULT_DIR
        print("[opf-setup] loading existing all-theta models only (no training)", flush=True)
        model_map = base.load_existing_all_theta_models(case)
        summaries = [run_seed(seed, case, model_map) for seed in selected_seeds]
    all_summary = pd.DataFrame(summaries)
    all_path = _selected_output_dir() / "vertex_opf_feasibility_all_selected_seed_summary.csv"
    _safe_to_csv(all_summary, all_path, description="All selected seed vertex OPF feasibility summary")
    _record_artifact("all", "csv", all_path, "All selected seed vertex OPF feasibility summary")
    print("[opf-explain] 本版本对 polygon 顶点固定 P0、Q0，并通过原始 OPF 约束直接验证可行性。", flush=True)
    print("[opf-explain] 该结果用于区分：1) NN recovery head 无法恢复，但真实 OPF 可行；2) polygon 顶点真实 OPF 不可行；3) MC/OPF q50 支撑线有限方向交集本身导致外近似不可行。", flush=True)
    print("[opf-explain] 本版本仍不等价于完整概率可行域证明；若要得到调度可用边界，后续需要基于 infeasible vertex 的 projection distance 做边界收缩。", flush=True)
    if not all_summary.empty:
        print("[opf-done] summary:", flush=True)
        print(all_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
