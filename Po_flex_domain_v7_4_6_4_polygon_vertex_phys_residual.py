"""v7.4.6.4 selected-seed posterior q50 polygon vertex physical residual diagnostics.

This script intentionally does not overwrite the v7.4.6.3 visualization script. It
reuses v7.4.6.3 selected-seed/model-loading/support-polygon utilities and the
v7.4.4 base module for network structures, normalization, grid constants, and
physics/recovery helpers.

Windows PowerShell:
    py -m py_compile Po_flex_domain_v7_4_6_4_polygon_vertex_phys_residual.py
    py Po_flex_domain_v7_4_6_4_polygon_vertex_phys_residual.py

Scope of this version:
- fixed source/load uncertainty quantile at q50 only;
- Bayesian posterior samples represent BNN parameter uncertainty only;
- no model training, no dataset rebuild, and no all-theta MC2500 re-evaluation;
- the physical residual is a neural-network recovery soft-feasibility diagnostic,
  not an OPF feasibility certificate.
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

import Po_flex_domain_v7_4_6_3_no_pcc_qcal_selected_seed_flex_domain_plots_with_cdf_bands as v7463

base = v7463.base
SCRIPT_DIR = Path(__file__).resolve().parent
NEW_SCRIPT_NAME = Path(__file__).name
BASE_SCRIPT_NAME = Path(v7463.__file__).name

# =========================
# Polygon vertex physical residual config
# =========================
PLOT_POLYGON_VERTEX_PHYS_RESIDUAL = True
N_BNN_DOMAIN_PHYS_SAMPLES = 50
SOURCE_LOAD_QUANTILE_FOR_PHYS = 0.50
PHYS_OK_TOL = 1e-4
SAVE_VERTEX_PHYS_DETAIL = True
SAVE_VERTEX_PHYS_SUMMARY = True
PLOT_VERTEX_PHYS_RESIDUAL = True

# Reuse selected seed configuration from v7.4.6.3 by default.
SELECTED_EXTERNAL_SEEDS = list(v7463.SELECTED_EXTERNAL_SEEDS)
N_THETA = int(v7463.N_THETA)
THETA_VALUES = np.asarray(v7463.THETA_VALUES, dtype=float)
ACTIVE_THETA_TOL = 1e-5
NEAREST_THETA_FALLBACK = True
POLYGON_COORD_MAX = float(v7463.POLYGON_COORD_MAX)
SUPPORT_EPS = float(v7463.SUPPORT_EPS)
SAVE_PNG = bool(v7463.SAVE_PNG)
SAVE_PDF = bool(v7463.SAVE_PDF)
DPI = int(v7463.DPI)

RESIDUAL_COLUMNS = [
    "pcc_p_abs_residual",
    "pcc_q_abs_residual",
    "global_p_abs_residual",
    "global_q_abs_residual",
    "kcl_p_max_abs_residual",
    "kcl_q_max_abs_residual",
    "voltage_upper_violation_max",
    "voltage_lower_violation_max",
    "line_p_violation_max",
    "line_q_violation_max",
    "line_s_violation_max",
    "pg_upper_violation_max",
    "pg_lower_violation_max",
    "qg_upper_violation_max",
    "qg_lower_violation_max",
    "pv_upper_violation_max",
    "max_eq_residual",
    "max_ineq_violation",
    "max_phys_violation",
    "phys_ok",
    "t_recovered",
    "t_forced",
    "t_abs_error",
    "p0_recovered_from_t",
    "q0_recovered_from_t",
    "p0_forced_recovery_abs_error",
    "q0_forced_recovery_abs_error",
    "support_residual_abs",
    "pcc_p_abs_residual_per_unit_raw",
    "pcc_q_abs_residual_per_unit_raw",
]

_manifest_artifacts: List[Dict[str, object]] = []
_nan_reasons: Dict[str, str] = {"line_s_violation_max": "Base v7.4.4 physics model enforces separate P/Q branch limits; no explicit apparent-power S branch limit is available in the reused constraints."}
_fallback_modes_seen: set[str] = set()


def _seed_tag(seed: int) -> str:
    return f"{int(seed):04d}"


def _abs_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else SCRIPT_DIR / p


def _seed_dir(seed: int) -> Path:
    return v7463._seed_dir(seed)


def _record_artifact(seed: int | str, artifact_type: str, path: str | Path, description: str) -> None:
    path = _abs_path(path)
    _manifest_artifacts.append(
        {
            "seed": seed,
            "artifact_type": artifact_type,
            "filename": path.name,
            "path": str(path),
            "description": description,
        }
    )
    print(f"[phys-artifact] {artifact_type} seed={seed} path={path}", flush=True)


def _safe_to_csv(df: pd.DataFrame, path: str | Path, *, description: str) -> Path:
    path = _abs_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[phys-save-csv] {description} -> {path}", flush=True)
    df.to_csv(path, index=False)
    if not path.exists():
        raise RuntimeError(f"CSV save failed: {path}")
    return path


def _save_fig(base_path_without_ext: Path, seed: int, description: str) -> None:
    base_path_without_ext = _abs_path(base_path_without_ext)
    base_path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        png = base_path_without_ext.with_suffix(".png")
        print(f"[phys-save-fig] -> {png}", flush=True)
        plt.savefig(png, dpi=DPI)
        _record_artifact(seed, "figure_png", png, description)
    if SAVE_PDF:
        pdf = base_path_without_ext.with_suffix(".pdf")
        print(f"[phys-save-fig] -> {pdf}", flush=True)
        plt.savefig(pdf)
        _record_artifact(seed, "figure_pdf", pdf, description)


def polygon_area(poly: np.ndarray) -> float:
    return v7463.polygon_area(poly)


def _closed(poly: np.ndarray) -> np.ndarray:
    return v7463._closed(poly)


def strict_halfspace_polygon_vertices(
    theta_values: Iterable[float],
    h_values: Iterable[float],
    *,
    tol: float = 1e-7,
    coord_max: float = POLYGON_COORD_MAX,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Extract true vertices of alpha*P0+beta*Q0 <= h by all-pair line intersections."""
    theta = np.asarray(theta_values, dtype=float).reshape(-1)
    h = np.asarray(h_values, dtype=float).reshape(-1)
    finite = np.isfinite(theta) & np.isfinite(h)
    theta = theta[finite]
    h = h[finite]
    alpha = np.cos(theta)
    beta = np.sin(theta)
    n = len(h)
    vertices: List[np.ndarray] = []
    max_violation = np.nan
    fail_reason = ""
    if n < 3:
        return np.empty((0, 2), dtype=float), {"status": "failed", "fail_reason": "fewer_than_3_halfspaces", "max_halfspace_violation": np.nan, "n_vertices": 0}
    for i in range(n):
        for j in range(i + 1, n):
            A = np.array([[alpha[i], beta[i]], [alpha[j], beta[j]]], dtype=float)
            det = float(np.linalg.det(A))
            if abs(det) < 1e-10:
                continue
            try:
                v = np.linalg.solve(A, np.array([h[i], h[j]], dtype=float))
            except np.linalg.LinAlgError:
                continue
            if (not np.all(np.isfinite(v))) or float(np.max(np.abs(v))) > coord_max:
                continue
            residual = alpha * v[0] + beta * v[1] - h
            local_max = float(np.nanmax(residual))
            max_violation = local_max if not np.isfinite(max_violation) else max(max_violation, local_max)
            if local_max <= tol:
                vertices.append(v)
    if not vertices:
        fail_reason = "no_pairwise_intersection_satisfies_all_halfspaces"
        return np.empty((0, 2), dtype=float), {"status": "failed", "fail_reason": fail_reason, "max_halfspace_violation": max_violation, "n_vertices": 0}
    # Deduplicate nearby intersections.
    unique: List[np.ndarray] = []
    for v in vertices:
        if not any(np.linalg.norm(v - u) <= 1e-6 for u in unique):
            unique.append(v)
    poly = np.asarray(unique, dtype=float)
    if poly.shape[0] < 3:
        return np.empty((0, 2), dtype=float), {"status": "failed", "fail_reason": "fewer_than_3_unique_vertices", "max_halfspace_violation": max_violation, "n_vertices": int(poly.shape[0])}
    center = np.mean(poly, axis=0)
    order = np.argsort(np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0]))
    poly = poly[order]
    final_res = np.matmul(np.column_stack([alpha, beta]), poly.T).T - h.reshape(1, -1)
    final_max = float(max(0.0, np.nanmax(final_res)))
    return poly, {"status": "ok", "fail_reason": "", "max_halfspace_violation": final_max, "n_vertices": int(poly.shape[0])}


def _recover_tensor_scalar(x) -> float:
    if hasattr(x, "detach"):
        return float(x.detach().cpu().numpy().reshape(-1)[0])
    return float(np.asarray(x).reshape(-1)[0])


def _max_or_zero(arr) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmax(arr))


def compute_physics_residual_components_for_vertex(
    *,
    case,
    net,
    norm: Dict[str, np.ndarray],
    x_mu: np.ndarray,
    theta_idx: int,
    h_value: float,
    t_value: Optional[float],
    forced_P0: float,
    forced_Q0: float,
    sample: bool = True,
    h_feature=None,
    device=None,
) -> Dict[str, object]:
    """Recover dispatch at a forced vertex support value and compute raw physical residual components."""
    device = base.DEVICE if device is None else device
    theta = float(THETA_VALUES[int(theta_idx)])
    alpha = float(np.cos(theta))
    beta = float(np.sin(theta))
    check_mode = "vertex_h_forced"
    with base.torch.no_grad():
        x_raw_np = np.asarray(x_mu, dtype=float).reshape(1, -1)
        x_norm_np = (x_raw_np - norm["x_mu_mean"]) / norm["x_mu_std"]
        xt = base.torch.tensor(x_norm_np, dtype=base.torch.float32, device=device)
        if h_feature is None:
            h_feature, _, _, _ = net.forward_gmm_single(xt, sample=sample)
        h_t = base.torch.tensor([[float(h_value)]], dtype=base.torch.float32, device=device)
        hm = base.torch.tensor(norm["h_mean"], dtype=base.torch.float32, device=device)
        hs = base.torch.tensor(norm["h_std"], dtype=base.torch.float32, device=device)
        tm = base.torch.tensor(norm["t_mean"], dtype=base.torch.float32, device=device)
        ts = base.torch.tensor(norm["t_std"], dtype=base.torch.float32, device=device)
        t_hat, pg_hat, qg_hat = net.recover_dispatch_from_h(h_feature, h_t, hm, hs, tm, ts, sample=sample)
        p0_forced_t = base.torch.tensor([[float(forced_P0)]], dtype=base.torch.float32, device=device)
        q0_forced_t = base.torch.tensor([[float(forced_Q0)]], dtype=base.torch.float32, device=device)
        x_raw_t = base.torch.tensor(x_raw_np, dtype=base.torch.float32, device=device)
        rec = base.recover_flows_from_flex_dispatch_batch(case, x_raw_t, p0_forced_t, q0_forced_t, pg_hat, qg_hat)

    P = rec["P"].detach().cpu().numpy()[0]
    Q = rec["Q"].detach().cpu().numpy()[0]
    V = rec["V"].detach().cpu().numpy()[0]
    pinj = rec["pinj"].detach().cpu().numpy()[0]
    qinj = rec["qinj"].detach().cpu().numpy()[0]
    pr = rec["pr"].detach().cpu().numpy()[0]
    pg = pg_hat.detach().cpu().numpy()[0]
    qg = qg_hat.detach().cpu().numpy()[0]
    root_out = np.asarray(case.out_branches[case.root], dtype=int)

    pcc_p_abs = abs(float(np.sum(P[root_out]) - forced_P0))
    pcc_q_abs = abs(float(np.sum(Q[root_out]) - forced_Q0))
    global_p_abs = abs(float(forced_P0 + np.sum(pinj)))
    global_q_abs = abs(float(forced_Q0 + np.sum(qinj)))

    kcl_p = []
    kcl_q = []
    for bus in range(case.n_bus):
        if bus == case.root:
            continue
        in_br = np.asarray(case.in_branches[bus], dtype=int)
        out_br = np.asarray(case.out_branches[bus], dtype=int)
        kcl_p.append(float(np.sum(P[in_br]) - np.sum(P[out_br]) + pinj[bus]))
        kcl_q.append(float(np.sum(Q[in_br]) - np.sum(Q[out_br]) + qinj[bus]))

    v_upper = np.maximum(V - case.vmax**2, 0.0)
    v_lower = np.maximum(case.vmin**2 - V, 0.0)
    if case.line_limit_mode == "internal_only_no_pcc":
        line_mask = np.asarray(case.internal_branch_mask, dtype=bool)
    else:
        line_mask = np.ones_like(P, dtype=bool)
    line_p = np.maximum(np.abs(P[line_mask]) - np.asarray(case.fmax_p)[line_mask], 0.0)
    line_q = np.maximum(np.abs(Q[line_mask]) - np.asarray(case.fmax_q)[line_mask], 0.0)
    pg_upper = np.maximum(pg - np.asarray(case.pg_max), 0.0)
    pg_lower = np.maximum(np.asarray(case.pg_min) - pg, 0.0)
    qg_upper = np.maximum(qg - np.asarray(case.qg_max), 0.0)
    qg_lower = np.maximum(np.asarray(case.qg_min) - qg, 0.0)
    pv_upper = np.maximum(pr[np.asarray(case.pv_buses, dtype=int)] - np.asarray(case.pv_pmax), 0.0)

    t_recovered = _recover_tensor_scalar(t_hat)
    t_forced = float(t_value) if t_value is not None and np.isfinite(t_value) else float(-beta * forced_P0 + alpha * forced_Q0)
    p0_recovered = float(alpha * h_value - beta * t_recovered)
    q0_recovered = float(beta * h_value + alpha * t_recovered)

    eq_vals = [pcc_p_abs, pcc_q_abs, global_p_abs, global_q_abs, _max_or_zero(np.abs(kcl_p)), _max_or_zero(np.abs(kcl_q))]
    ineq_vals = [
        _max_or_zero(v_upper),
        _max_or_zero(v_lower),
        _max_or_zero(line_p),
        _max_or_zero(line_q),
        _max_or_zero(pg_upper),
        _max_or_zero(pg_lower),
        _max_or_zero(qg_upper),
        _max_or_zero(qg_lower),
        _max_or_zero(pv_upper),
    ]
    max_eq = float(np.nanmax(eq_vals))
    max_ineq = float(np.nanmax(ineq_vals))
    max_phys = float(max(max_eq, max_ineq))
    return {
        "check_mode": check_mode,
        "pcc_p_abs_residual": pcc_p_abs,
        "pcc_q_abs_residual": pcc_q_abs,
        "global_p_abs_residual": global_p_abs,
        "global_q_abs_residual": global_q_abs,
        "kcl_p_max_abs_residual": _max_or_zero(np.abs(kcl_p)),
        "kcl_q_max_abs_residual": _max_or_zero(np.abs(kcl_q)),
        "voltage_upper_violation_max": _max_or_zero(v_upper),
        "voltage_lower_violation_max": _max_or_zero(v_lower),
        "line_p_violation_max": _max_or_zero(line_p),
        "line_q_violation_max": _max_or_zero(line_q),
        "line_s_violation_max": np.nan,
        "pg_upper_violation_max": _max_or_zero(pg_upper),
        "pg_lower_violation_max": _max_or_zero(pg_lower),
        "qg_upper_violation_max": _max_or_zero(qg_upper),
        "qg_lower_violation_max": _max_or_zero(qg_lower),
        "pv_upper_violation_max": _max_or_zero(pv_upper),
        "max_eq_residual": max_eq,
        "max_ineq_violation": max_ineq,
        "max_phys_violation": max_phys,
        "phys_ok": bool(max_phys <= PHYS_OK_TOL),
        "t_recovered": t_recovered,
        "t_forced": t_forced,
        "t_abs_error": abs(t_recovered - t_forced),
        "p0_recovered_from_t": p0_recovered,
        "q0_recovered_from_t": q0_recovered,
        "p0_forced_recovery_abs_error": abs(p0_recovered - forced_P0),
        "q0_forced_recovery_abs_error": abs(q0_recovered - forced_Q0),
        "support_residual_abs": abs(float(alpha * forced_P0 + beta * forced_Q0 - h_value)),
        "pcc_p_abs_residual_per_unit_raw": pcc_p_abs,
        "pcc_q_abs_residual_per_unit_raw": pcc_q_abs,
    }


def _predict_deterministic_q50_for_seed(seed: int, x_mu: np.ndarray, model_map: Dict[int, dict]) -> np.ndarray:
    h = np.full(N_THETA, np.nan, dtype=float)
    x_mu = np.asarray(x_mu, dtype=float).reshape(1, -1)
    for theta_idx in range(N_THETA):
        entry = model_map.get(theta_idx, {})
        if entry.get("net") is None or entry.get("norm") is None:
            print(f"[phys-warning] seed={seed:04d} theta={theta_idx:02d} missing deterministic model", flush=True)
            continue
        net = entry["net"]
        norm = entry["norm"]
        xt = base.torch.tensor((x_mu - norm["x_mu_mean"]) / norm["x_mu_std"], dtype=base.torch.float32, device=base.DEVICE)
        with base.torch.no_grad():
            _, w, mu, sigma = net.forward_gmm_single(xt, sample=False)
        q_norm = np.asarray(base.gmm_quantile([SOURCE_LOAD_QUANTILE_FOR_PHYS], w.detach().cpu().numpy().reshape(-1), mu.detach().cpu().numpy().reshape(-1), sigma.detach().cpu().numpy().reshape(-1))).reshape(-1)[0]
        h[theta_idx] = float(norm["h_mean"][0, 0] + norm["h_std"][0, 0] * q_norm)
    return h


def _posterior_sample_q50_supports(seed: int, sample_idx: int, x_mu: np.ndarray, model_map: Dict[int, dict]) -> Tuple[np.ndarray, Dict[int, object]]:
    h_values = np.full(N_THETA, np.nan, dtype=float)
    h_features: Dict[int, object] = {}
    x_mu = np.asarray(x_mu, dtype=float).reshape(1, -1)
    for theta_idx in range(N_THETA):
        entry = model_map.get(theta_idx, {})
        if entry.get("net") is None or entry.get("norm") is None:
            print(f"[phys-warning] seed={seed:04d} sample={sample_idx:03d} theta={theta_idx:02d} missing model", flush=True)
            continue
        net = entry["net"]
        norm = entry["norm"]
        xt = base.torch.tensor((x_mu - norm["x_mu_mean"]) / norm["x_mu_std"], dtype=base.torch.float32, device=base.DEVICE)
        with base.torch.no_grad():
            hf, w, mu, sigma = net.forward_gmm_single(xt, sample=True)
        q_norm = np.asarray(base.gmm_quantile([SOURCE_LOAD_QUANTILE_FOR_PHYS], w.detach().cpu().numpy().reshape(-1), mu.detach().cpu().numpy().reshape(-1), sigma.detach().cpu().numpy().reshape(-1))).reshape(-1)[0]
        h_values[theta_idx] = float(norm["h_mean"][0, 0] + norm["h_std"][0, 0] * q_norm)
        h_features[theta_idx] = hf
    return h_values, h_features


def _active_theta_indices(p0: float, q0: float, h_values: np.ndarray) -> Tuple[List[int], np.ndarray]:
    alpha = np.cos(THETA_VALUES)
    beta = np.sin(THETA_VALUES)
    support_res = alpha * p0 + beta * q0 - h_values
    active = np.where(np.isfinite(support_res) & (np.abs(support_res) <= ACTIVE_THETA_TOL))[0].astype(int).tolist()
    if not active and NEAREST_THETA_FALLBACK:
        finite = np.where(np.isfinite(support_res))[0]
        if finite.size:
            nearest = int(finite[np.argmin(np.abs(support_res[finite]))])
            active = [nearest]
            _fallback_modes_seen.add("nearest_theta_fallback")
    return active, support_res


def _quantile_or_nan(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanquantile(arr, q))


def _write_empty_outputs_for_failed_sample(seed: int, sample_idx: int, reason: str) -> dict:
    print(f"[phys-warning] seed={seed:04d} sample={sample_idx:03d} polygon failed: {reason}", flush=True)
    return {
        "seed": seed,
        "sample_idx": sample_idx,
        "n_vertices": 0,
        "n_checked_vertex_theta_pairs": 0,
        "n_phys_ok": 0,
        "phys_ok_rate": np.nan,
        "max_phys_violation": np.nan,
        "q50_phys_violation": np.nan,
        "q95_phys_violation": np.nan,
        "max_eq_residual": np.nan,
        "max_ineq_violation": np.nan,
        "polygon_area": np.nan,
        "max_halfspace_violation": np.nan,
        "status": "failed",
        "fail_reason": reason,
    }



def safe_load_seed_ranking() -> pd.DataFrame:
    """Use v7.4.6.3 ranking when present; otherwise keep manual selected seeds runnable."""
    try:
        return v7463.load_seed_ranking(v7463.COMBINED_EXTERNAL_METRICS_CSV)
    except FileNotFoundError as exc:
        print(f"[phys-warning] combined metrics CSV missing; using manual selected seeds without ranking: {exc}", flush=True)
        rows = [{"seed": int(seed), "mean_arms_percent": np.nan, "max_arms_percent": np.nan, "median_arms_percent": np.nan, "q90_arms_percent": np.nan, "n_theta": N_THETA, "n_theta_gt_5_percent": 0} for seed in SELECTED_EXTERNAL_SEEDS]
        return pd.DataFrame(rows)


def safe_get_support_context_for_seed(seed: int, case) -> Dict[str, object]:
    """Load the selected-seed support cache if available; otherwise build only q50 x_mu context.

    The vertex residual diagnostic does not need MC support clouds. Avoiding a cache-miss
    MC-OPF recomputation keeps this version aligned with the no-new-OPF requirement.
    """
    seed_dir = _seed_dir(seed)
    cache_path = seed_dir / f"selected_seed_{_seed_tag(seed)}_mc_supports.npz"
    if cache_path.exists():
        print(f"[phys] seed={seed:04d} loading existing v7.4.6.3 support cache {cache_path}", flush=True)
        z = np.load(cache_path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    print(f"[phys-warning] seed={seed:04d} support cache missing; not rerunning MC-OPF in v7.4.6.4, drawing fixed q50 source/load context only", flush=True)
    pd_mu, qd_mu, pr_mu, qr_mu = base.draw_external_scenario_by_seed(case, seed)
    x_mu = base.make_feature_vector(case, pd_mu, pr_mu)
    return {"seed": np.array(seed), "x_mu": x_mu, "pd_mu": pd_mu, "qd_mu": qd_mu, "pr_mu": pr_mu, "qr_mu": qr_mu, "mc_opf_recomputed_in_v7464": np.array(False)}


def write_missing_model_outputs(seed: int, missing_theta: List[int]) -> Dict[str, object]:
    """Write version-specific empty diagnostics when required model artifacts are unavailable."""
    seed_dir = _seed_dir(seed)
    seed_tag = _seed_tag(seed)
    print(f"[phys-warning] seed={seed_tag} missing theta models={missing_theta}; writing empty diagnostics and skipping residual checks", flush=True)
    vertices_df = pd.DataFrame(columns=["seed", "sample_idx", "vertex_idx", "P0", "Q0", "polygon_area", "max_halfspace_violation", "n_vertices"])
    detail_df = pd.DataFrame(columns=["seed", "sample_idx", "vertex_idx", "P0", "Q0", "active_theta_idx", "theta", "alpha", "beta", "h_vertex", "t_vertex", "check_mode", *RESIDUAL_COLUMNS])
    sample_df = pd.DataFrame(columns=["seed", "sample_idx", "n_vertices", "n_checked_vertex_theta_pairs", "n_phys_ok", "phys_ok_rate", "max_phys_violation", "q50_phys_violation", "q95_phys_violation", "max_eq_residual", "max_ineq_violation", "polygon_area", "max_halfspace_violation"])
    theta_df = pd.DataFrame(columns=["seed", "theta_idx", "theta", "n_checks", "phys_ok_rate", "max_phys_violation", "q50_phys_violation", "q95_phys_violation", "mean_phys_violation"])
    summary = {"seed": seed, "n_bnn_samples": N_BNN_DOMAIN_PHYS_SAMPLES, "source_load_quantile": SOURCE_LOAD_QUANTILE_FOR_PHYS, "phys_ok_tol": PHYS_OK_TOL, "total_vertices": 0, "total_vertex_theta_checks": 0, "overall_phys_ok_rate": np.nan, "worst_sample_idx": -1, "worst_vertex_idx": -1, "worst_theta_idx": -1, "max_phys_violation_global": np.nan, "q50_phys_violation_global": np.nan, "q95_phys_violation_global": np.nan, "n_samples_all_vertices_ok": 0, "sample_all_vertices_ok_rate": np.nan}
    files = [
        (vertices_df, seed_dir / f"selected_seed_{seed_tag}_posterior_q50_polygon_vertices.csv", "empty posterior q50 polygon vertices due to missing models"),
        (detail_df, seed_dir / f"selected_seed_{seed_tag}_posterior_q50_vertex_phys_residual_detail.csv", "empty vertex residual detail due to missing models"),
        (sample_df, seed_dir / f"selected_seed_{seed_tag}_posterior_q50_vertex_phys_residual_by_sample.csv", "empty sample summary due to missing models"),
        (theta_df, seed_dir / f"selected_seed_{seed_tag}_posterior_q50_vertex_phys_residual_by_theta.csv", "empty theta summary due to missing models"),
        (pd.DataFrame([summary]), seed_dir / f"selected_seed_{seed_tag}_posterior_q50_vertex_phys_residual_summary.csv", "global summary noting missing models"),
    ]
    for df, path, desc in files:
        _safe_to_csv(df, path, description=desc)
        _record_artifact(seed, "csv", path, desc)
    manifest_path = seed_dir / f"selected_seed_{seed_tag}_polygon_vertex_phys_residual_manifest.json"
    manifest = {"new_file_name": NEW_SCRIPT_NAME, "base_file_name": BASE_SCRIPT_NAME, "selected_seeds": SELECTED_EXTERNAL_SEEDS, "N_BNN_DOMAIN_PHYS_SAMPLES": N_BNN_DOMAIN_PHYS_SAMPLES, "SOURCE_LOAD_QUANTILE_FOR_PHYS": SOURCE_LOAD_QUANTILE_FOR_PHYS, "PHYS_OK_TOL": PHYS_OK_TOL, "fallback_check_modes_seen": ["missing_model_skip"], "check_mode_has_fallback": True, "retrained_model": False, "rebuilt_dataset": False, "reran_mc_opf": False, "opf_feasibility_check": False, "nn_recovery_physical_residual_check": False, "skip_reason": "missing required all-theta model artifacts", "missing_theta_indices": missing_theta, "nan_residual_column_reasons": {**_nan_reasons, "all_residual_columns": "No residuals computed because required model artifacts are missing."}, "outputs": _manifest_artifacts}
    print(f"[phys-save-json] manifest -> {manifest_path}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _record_artifact(seed, "json", manifest_path, "Polygon vertex physical residual manifest for missing-model skip")
    return summary


def run_vertex_phys_diagnostics_for_seed(seed: int, case, supports: Dict[str, object], model_map: Dict[int, dict]) -> Dict[str, object]:
    seed_dir = _seed_dir(seed)
    seed_tag = _seed_tag(seed)
    x_mu = np.asarray(supports["x_mu"], dtype=float).reshape(1, -1)
    print(f"[phys] seed={seed_tag} fixed source/load quantile q{int(SOURCE_LOAD_QUANTILE_FOR_PHYS*100):02d}; no q05/q95 bands", flush=True)

    deterministic_h = _predict_deterministic_q50_for_seed(seed, x_mu, model_map)
    deterministic_poly, deterministic_info = strict_halfspace_polygon_vertices(THETA_VALUES, deterministic_h)
    if deterministic_info["status"] != "ok":
        print(f"[phys-warning] seed={seed_tag} deterministic q50 strict polygon failed: {deterministic_info}", flush=True)
        deterministic_poly = v7463.support_values_to_polygon(THETA_VALUES, deterministic_h)
        _fallback_modes_seen.add("deterministic_polygon_legacy_fallback")

    vertex_rows: List[dict] = []
    detail_rows: List[dict] = []
    sample_summary_rows: List[dict] = []
    theta_values_for_plot: List[Tuple[np.ndarray, np.ndarray]] = []

    for sample_idx in range(N_BNN_DOMAIN_PHYS_SAMPLES):
        print(f"[phys] seed={seed_tag} sample={sample_idx+1:03d}/{N_BNN_DOMAIN_PHYS_SAMPLES:03d}", flush=True)
        h_values, h_features = _posterior_sample_q50_supports(seed, sample_idx, x_mu, model_map)
        print("[phys] extracting polygon vertices...", flush=True)
        poly, poly_info = strict_halfspace_polygon_vertices(THETA_VALUES, h_values)
        if poly_info["status"] != "ok":
            sample_summary_rows.append(_write_empty_outputs_for_failed_sample(seed, sample_idx, str(poly_info.get("fail_reason", "unknown"))))
            continue
        area = polygon_area(poly)
        max_hs_vio = float(poly_info["max_halfspace_violation"])
        theta_values_for_plot.append((poly.copy(), h_values.copy()))
        for vertex_idx, (p0, q0) in enumerate(poly):
            vertex_rows.append(
                {
                    "seed": seed,
                    "sample_idx": sample_idx,
                    "vertex_idx": vertex_idx,
                    "P0": float(p0),
                    "Q0": float(q0),
                    "polygon_area": area,
                    "max_halfspace_violation": max_hs_vio,
                    "n_vertices": int(poly.shape[0]),
                }
            )
            active, support_res = _active_theta_indices(float(p0), float(q0), h_values)
            if not active:
                print(f"[phys-warning] seed={seed_tag} sample={sample_idx:03d} vertex={vertex_idx:02d} no active/nearest theta; skip", flush=True)
                continue
            if len(active) == 1 and abs(float(support_res[active[0]])) > ACTIVE_THETA_TOL:
                check_mode_override = "nearest_theta_fallback"
                print(f"[phys-warning] seed={seed_tag} sample={sample_idx:03d} vertex={vertex_idx:02d} no active theta; nearest_theta={active[0]}", flush=True)
            else:
                check_mode_override = None
            print(f"[phys] checking vertex {vertex_idx+1:02d}/{poly.shape[0]:02d} active_theta={active}", flush=True)
            for theta_idx in active:
                theta = float(THETA_VALUES[int(theta_idx)])
                alpha = float(np.cos(theta))
                beta = float(np.sin(theta))
                h_vertex = float(alpha * p0 + beta * q0)
                t_vertex = float(-beta * p0 + alpha * q0)
                entry = model_map.get(int(theta_idx), {})
                base_row = {
                    "seed": seed,
                    "sample_idx": sample_idx,
                    "vertex_idx": vertex_idx,
                    "P0": float(p0),
                    "Q0": float(q0),
                    "active_theta_idx": int(theta_idx),
                    "theta": theta,
                    "alpha": alpha,
                    "beta": beta,
                    "h_vertex": h_vertex,
                    "t_vertex": t_vertex,
                }
                try:
                    comp = compute_physics_residual_components_for_vertex(
                        case=case,
                        net=entry["net"],
                        norm=entry["norm"],
                        x_mu=x_mu,
                        theta_idx=int(theta_idx),
                        h_value=h_vertex,
                        t_value=t_vertex,
                        forced_P0=float(p0),
                        forced_Q0=float(q0),
                        sample=True,
                        h_feature=h_features.get(int(theta_idx)),
                        device=base.DEVICE,
                    )
                    if check_mode_override is not None:
                        comp["check_mode"] = check_mode_override
                        _fallback_modes_seen.add(check_mode_override)
                    detail_rows.append({**base_row, **comp})
                except Exception as exc:  # noqa: BLE001 - diagnostics should continue per sample/vertex failures.
                    print(f"[phys-warning] seed={seed_tag} sample={sample_idx:03d} vertex={vertex_idx:02d} theta={theta_idx:02d} residual failed: {exc}", flush=True)
                    fail_comp = {col: np.nan for col in RESIDUAL_COLUMNS}
                    fail_comp.update({"check_mode": "residual_exception", "phys_ok": False, "failure_reason": str(exc)})
                    _fallback_modes_seen.add("residual_exception")
                    detail_rows.append({**base_row, **fail_comp})
        sample_details = [r for r in detail_rows if int(r.get("sample_idx", -1)) == sample_idx]
        vals = [float(r.get("max_phys_violation", np.nan)) for r in sample_details]
        ok_flags = [bool(r.get("phys_ok", False)) for r in sample_details if np.isfinite(float(r.get("max_phys_violation", np.nan)))]
        n_checks = len(vals)
        n_ok = int(sum(ok_flags))
        sample_summary = {
            "seed": seed,
            "sample_idx": sample_idx,
            "n_vertices": int(poly.shape[0]),
            "n_checked_vertex_theta_pairs": int(n_checks),
            "n_phys_ok": n_ok,
            "phys_ok_rate": float(n_ok / n_checks) if n_checks else np.nan,
            "max_phys_violation": _quantile_or_nan(vals, 1.0),
            "q50_phys_violation": _quantile_or_nan(vals, 0.50),
            "q95_phys_violation": _quantile_or_nan(vals, 0.95),
            "max_eq_residual": _quantile_or_nan([r.get("max_eq_residual", np.nan) for r in sample_details], 1.0),
            "max_ineq_violation": _quantile_or_nan([r.get("max_ineq_violation", np.nan) for r in sample_details], 1.0),
            "polygon_area": area,
            "max_halfspace_violation": max_hs_vio,
            "status": "ok",
            "fail_reason": "",
        }
        sample_summary_rows.append(sample_summary)
        print(f"[phys] sample summary ok_rate={sample_summary['phys_ok_rate']:.4f}, max_violation={sample_summary['max_phys_violation']:.6e}", flush=True)

    vertices_df = pd.DataFrame(vertex_rows)
    detail_df = pd.DataFrame(detail_rows)
    sample_df = pd.DataFrame(sample_summary_rows)
    if not detail_df.empty:
        for col in RESIDUAL_COLUMNS:
            if col not in detail_df.columns:
                detail_df[col] = np.nan

    by_theta_rows = []
    if not detail_df.empty:
        for theta_idx, g in detail_df.groupby("active_theta_idx"):
            vals = g["max_phys_violation"].astype(float).values
            finite = vals[np.isfinite(vals)]
            ok = g.loc[np.isfinite(vals), "phys_ok"].astype(bool).values if finite.size else []
            by_theta_rows.append(
                {
                    "seed": seed,
                    "theta_idx": int(theta_idx),
                    "theta": float(THETA_VALUES[int(theta_idx)]),
                    "n_checks": int(finite.size),
                    "phys_ok_rate": float(np.mean(ok)) if finite.size else np.nan,
                    "max_phys_violation": float(np.nanmax(finite)) if finite.size else np.nan,
                    "q50_phys_violation": float(np.nanquantile(finite, 0.50)) if finite.size else np.nan,
                    "q95_phys_violation": float(np.nanquantile(finite, 0.95)) if finite.size else np.nan,
                    "mean_phys_violation": float(np.nanmean(finite)) if finite.size else np.nan,
                }
            )
    by_theta_df = pd.DataFrame(by_theta_rows)

    finite_detail = detail_df[np.isfinite(detail_df.get("max_phys_violation", pd.Series(dtype=float)).astype(float))] if not detail_df.empty else pd.DataFrame()
    if not finite_detail.empty:
        worst = finite_detail.sort_values("max_phys_violation", ascending=False).iloc[0]
        vals = finite_detail["max_phys_violation"].astype(float).values
        total_checks = int(len(finite_detail))
        total_vertices = int(len(vertices_df))
        sample_ok = sample_df[sample_df["status"].eq("ok")].copy()
        all_vertices_ok = sample_ok["max_phys_violation"].astype(float) <= PHYS_OK_TOL
        global_summary = {
            "seed": seed,
            "n_bnn_samples": N_BNN_DOMAIN_PHYS_SAMPLES,
            "source_load_quantile": SOURCE_LOAD_QUANTILE_FOR_PHYS,
            "phys_ok_tol": PHYS_OK_TOL,
            "total_vertices": total_vertices,
            "total_vertex_theta_checks": total_checks,
            "overall_phys_ok_rate": float(finite_detail["phys_ok"].astype(bool).mean()),
            "worst_sample_idx": int(worst["sample_idx"]),
            "worst_vertex_idx": int(worst["vertex_idx"]),
            "worst_theta_idx": int(worst["active_theta_idx"]),
            "max_phys_violation_global": float(np.nanmax(vals)),
            "q50_phys_violation_global": float(np.nanquantile(vals, 0.50)),
            "q95_phys_violation_global": float(np.nanquantile(vals, 0.95)),
            "n_samples_all_vertices_ok": int(all_vertices_ok.sum()),
            "sample_all_vertices_ok_rate": float(all_vertices_ok.mean()) if len(all_vertices_ok) else np.nan,
        }
    else:
        global_summary = {
            "seed": seed,
            "n_bnn_samples": N_BNN_DOMAIN_PHYS_SAMPLES,
            "source_load_quantile": SOURCE_LOAD_QUANTILE_FOR_PHYS,
            "phys_ok_tol": PHYS_OK_TOL,
            "total_vertices": int(len(vertices_df)),
            "total_vertex_theta_checks": 0,
            "overall_phys_ok_rate": np.nan,
            "worst_sample_idx": -1,
            "worst_vertex_idx": -1,
            "worst_theta_idx": -1,
            "max_phys_violation_global": np.nan,
            "q50_phys_violation_global": np.nan,
            "q95_phys_violation_global": np.nan,
            "n_samples_all_vertices_ok": 0,
            "sample_all_vertices_ok_rate": np.nan,
        }
    global_summary_df = pd.DataFrame([global_summary])

    out_vertices = seed_dir / f"selected_seed_{seed_tag}_posterior_q50_polygon_vertices.csv"
    out_detail = seed_dir / f"selected_seed_{seed_tag}_posterior_q50_vertex_phys_residual_detail.csv"
    out_sample = seed_dir / f"selected_seed_{seed_tag}_posterior_q50_vertex_phys_residual_by_sample.csv"
    out_theta = seed_dir / f"selected_seed_{seed_tag}_posterior_q50_vertex_phys_residual_by_theta.csv"
    out_summary = seed_dir / f"selected_seed_{seed_tag}_posterior_q50_vertex_phys_residual_summary.csv"
    _safe_to_csv(vertices_df, out_vertices, description="Posterior q50 strict polygon vertices")
    _record_artifact(seed, "csv", out_vertices, "Posterior q50 strict half-space polygon vertices")
    if SAVE_VERTEX_PHYS_DETAIL:
        _safe_to_csv(detail_df, out_detail, description="Posterior q50 polygon vertex physical residual detail")
        _record_artifact(seed, "csv", out_detail, "Per vertex-active-theta NN recovery physical residual detail")
    if SAVE_VERTEX_PHYS_SUMMARY:
        _safe_to_csv(sample_df, out_sample, description="Posterior q50 vertex physical residual summary by sample")
        _record_artifact(seed, "csv", out_sample, "Posterior sample-level vertex physical residual summary")
        _safe_to_csv(by_theta_df, out_theta, description="Posterior q50 vertex physical residual summary by theta")
        _record_artifact(seed, "csv", out_theta, "Theta-level vertex physical residual summary")
        _safe_to_csv(global_summary_df, out_summary, description="Posterior q50 vertex physical residual global summary")
        _record_artifact(seed, "csv", out_summary, "Global vertex physical residual summary")

    if PLOT_VERTEX_PHYS_RESIDUAL:
        plot_polygon_phys_residual_overlay(seed, deterministic_poly, theta_values_for_plot, vertices_df, detail_df)
        plot_phys_violation_distribution(seed, detail_df)

    manifest_path = seed_dir / f"selected_seed_{seed_tag}_polygon_vertex_phys_residual_manifest.json"
    manifest = {
        "new_file_name": NEW_SCRIPT_NAME,
        "base_file_name": BASE_SCRIPT_NAME,
        "selected_seeds": SELECTED_EXTERNAL_SEEDS,
        "N_BNN_DOMAIN_PHYS_SAMPLES": N_BNN_DOMAIN_PHYS_SAMPLES,
        "SOURCE_LOAD_QUANTILE_FOR_PHYS": SOURCE_LOAD_QUANTILE_FOR_PHYS,
        "PHYS_OK_TOL": PHYS_OK_TOL,
        "fallback_check_modes_seen": sorted(_fallback_modes_seen),
        "check_mode_has_fallback": bool(_fallback_modes_seen),
        "retrained_model": False,
        "rebuilt_dataset": False,
        "reran_mc_opf": False,
        "opf_feasibility_check": False,
        "nn_recovery_physical_residual_check": True,
        "residual_scale_note": "Residual columns are raw pre-normalization physical quantities in the per-unit scale used by the IEEE-33 base case; pcc_*_per_unit_raw duplicate the raw PCC residuals for clarity.",
        "nan_residual_column_reasons": _nan_reasons,
        "outputs": _manifest_artifacts,
        "deterministic_polygon_info": deterministic_info,
        "strict_halfspace_vertex_extraction": True,
    }
    print(f"[phys-save-json] manifest -> {manifest_path}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _record_artifact(seed, "json", manifest_path, "Polygon vertex physical residual manifest")
    return global_summary


def plot_polygon_phys_residual_overlay(seed: int, deterministic_poly: np.ndarray, posterior_polys: List[Tuple[np.ndarray, np.ndarray]], vertices_df: pd.DataFrame, detail_df: pd.DataFrame) -> None:
    seed_dir = _seed_dir(seed)
    seed_tag = _seed_tag(seed)
    plt.figure(figsize=(7.4, 6.5))
    for idx, (poly, _) in enumerate(posterior_polys):
        pp = _closed(poly)
        plt.plot(pp[:, 0], pp[:, 1], color="#94a3b8", linewidth=0.7, alpha=0.25, label="posterior q50 sampled polygons" if idx == 0 else None)
    if deterministic_poly is not None and len(deterministic_poly) >= 3:
        dp = _closed(deterministic_poly)
        plt.plot(dp[:, 0], dp[:, 1], color="#f97316", linewidth=2.2, linestyle="--", label="deterministic q50 polygon")
    if not vertices_df.empty and not detail_df.empty:
        vertex_max = detail_df.groupby(["sample_idx", "vertex_idx"])["max_phys_violation"].max().reset_index()
        vplot = vertices_df.merge(vertex_max, on=["sample_idx", "vertex_idx"], how="left")
        vals = vplot["max_phys_violation"].astype(float).values
        sizes = 20.0 + 80.0 * np.nan_to_num(vals / max(np.nanmax(vals), 1e-12), nan=0.0)
        sc = plt.scatter(vplot["P0"], vplot["Q0"], c=vals, s=sizes, cmap="viridis", alpha=0.75, edgecolors="none", label="vertex physical residual")
        plt.colorbar(sc, label="max physical violation (raw p.u.)")
        if np.isfinite(vals).any():
            worst_i = int(np.nanargmax(vals))
            worst = vplot.iloc[worst_i]
            plt.scatter([worst["P0"]], [worst["Q0"]], marker="*", s=220, color="#dc2626", edgecolors="black", linewidths=0.6, label="worst residual vertex")
    plt.xlabel("P_0")
    plt.ylabel("Q_0")
    plt.title(f"Seed {seed}: posterior q50 polygon vertex physical residuals")
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    _save_fig(seed_dir / f"selected_seed_{seed_tag}_posterior_q50_polygon_phys_residual_overlay", seed, "Posterior q50 polygon overlay colored by vertex physical residual")
    plt.close()


def plot_phys_violation_distribution(seed: int, detail_df: pd.DataFrame) -> None:
    seed_dir = _seed_dir(seed)
    seed_tag = _seed_tag(seed)
    vals = np.array([], dtype=float)
    if not detail_df.empty and "max_phys_violation" in detail_df.columns:
        vals = detail_df["max_phys_violation"].astype(float).values
        vals = vals[np.isfinite(vals)]
    plt.figure(figsize=(7.0, 4.5))
    if vals.size:
        plt.boxplot(vals, vert=False, showmeans=True)
        plt.scatter(vals, np.ones_like(vals), color="#2563eb", alpha=0.25, s=12)
        plt.axvline(PHYS_OK_TOL, color="#dc2626", linestyle="--", label=f"PHYS_OK_TOL={PHYS_OK_TOL:g}")
        plt.xlabel("max physical violation (raw p.u.)")
    else:
        plt.text(0.5, 0.5, "No finite vertex residuals", ha="center", va="center", transform=plt.gca().transAxes)
        plt.xlabel("max physical violation")
    plt.yticks([])
    plt.title(f"Seed {seed}: vertex physical violation distribution")
    plt.grid(axis="x", alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    _save_fig(seed_dir / f"selected_seed_{seed_tag}_posterior_q50_phys_violation_distribution", seed, "Distribution of posterior q50 polygon vertex physical violations")
    plt.close()


def main() -> None:
    t0 = time.perf_counter()
    print("[v7.4.6.4-polygon-vertex-phys-residual] start", flush=True)
    print(f"[phys-config] PLOT_POLYGON_VERTEX_PHYS_RESIDUAL={PLOT_POLYGON_VERTEX_PHYS_RESIDUAL}", flush=True)
    print(f"[phys-config] N_BNN_DOMAIN_PHYS_SAMPLES={N_BNN_DOMAIN_PHYS_SAMPLES}", flush=True)
    print(f"[phys-config] SOURCE_LOAD_QUANTILE_FOR_PHYS={SOURCE_LOAD_QUANTILE_FOR_PHYS}", flush=True)
    print(f"[phys-config] PHYS_OK_TOL={PHYS_OK_TOL}", flush=True)
    print(f"[phys-config] selected_seeds={SELECTED_EXTERNAL_SEEDS}", flush=True)
    print("[phys-note] This version does not train models, rebuild data1000/mc60, rerun all-theta MC2500, or run OPF feasibility checks.", flush=True)

    if not PLOT_POLYGON_VERTEX_PHYS_RESIDUAL:
        print("[phys] disabled; exiting", flush=True)
        return

    v7463._ensure_dir(v7463.SELECTED_FLEX_DOMAIN_OUT_DIR)
    print("[phys-setup] loading seed ranking and selected seeds via v7.4.6.3 logic", flush=True)
    seed_ranking = safe_load_seed_ranking()
    selected_seeds = v7463.select_external_seeds(seed_ranking)
    print("[phys-setup] building IEEE-33 no-PCC case", flush=True)
    case = base.build_ieee33_case()
    base.ALL_THETA_LIST = list(range(N_THETA))
    base.ALL_THETA_RESULT_DIR = v7463.ALL_THETA_RESULT_DIR
    print("[phys-setup] loading qcal all-theta models", flush=True)
    model_map = base.load_existing_all_theta_models(case)

    summaries = []
    for seed in selected_seeds:
        print(f"[phys] start selected seed={seed:04d}", flush=True)
        missing_theta = [j for j in range(N_THETA) if model_map.get(j, {}).get("net") is None or model_map.get(j, {}).get("norm") is None]
        if missing_theta:
            summaries.append(write_missing_model_outputs(seed, missing_theta))
            print(f"[phys] done selected seed={seed:04d} with missing-model skip", flush=True)
            continue
        supports = safe_get_support_context_for_seed(seed, case)
        summaries.append(run_vertex_phys_diagnostics_for_seed(seed, case, supports, model_map))
        print(f"[phys] done selected seed={seed:04d}", flush=True)

    all_summary = pd.DataFrame(summaries)
    all_summary_path = _abs_path(v7463.SELECTED_FLEX_DOMAIN_OUT_DIR) / "posterior_q50_vertex_phys_residual_all_selected_seed_summary.csv"
    _safe_to_csv(all_summary, all_summary_path, description="All selected seed vertex physical residual summary")
    _record_artifact("all", "csv", all_summary_path, "All selected seed posterior q50 vertex residual summary")
    print("[phys-explain] 本版本只固定源荷 q50。", flush=True)
    print("[phys-explain] posterior samples 只表示 Bayesian neural network 参数不确定性。", flush=True)
    print("[phys-explain] 物理残差是 NN recovery 意义下的软可行性诊断。", flush=True)
    print("[phys-explain] 该诊断不等价于严格 OPF feasibility certificate。", flush=True)
    print("[phys-explain] 如果后续要证明整个 polygon 可行，需要新增 OPF vertex feasibility check。", flush=True)
    print(f"[phys-done] elapsed_sec={time.perf_counter()-t0:.2f}", flush=True)
    if not all_summary.empty:
        print("[phys-done] summary:", flush=True)
        print(all_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
