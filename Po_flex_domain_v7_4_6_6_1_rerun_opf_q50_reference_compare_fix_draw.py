"""v7.4.6.6.1 rerun OPF q50 reference compare draw-realization fix.

This version reruns the original OPF support problem for one selected seed and
12 theta directions to build a fresh OPF q50 half-space polygon. It then
compares that reference with BPINN deterministic q50, BPINN posterior q50, and
cache-based MC q50 references, and validates the rerun OPF q50 polygon vertices
with fixed-P0/Q0 OPF feasibility checks.

Windows PowerShell:
    py -m py_compile Po_flex_domain_v7_4_6_6_1_rerun_opf_q50_reference_compare_fix_draw.py
    py Po_flex_domain_v7_4_6_6_1_rerun_opf_q50_reference_compare_fix_draw.py
"""

from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import Po_flex_domain_v7_4_6_5_vertex_opf_feasibility as opf65
    phys64 = opf65.phys64
    v7463 = opf65.v7463
    base = opf65.base
    gp = opf65.gp
    GRB = opf65.GRB
    GUROBI_AVAILABLE = bool(opf65.GUROBI_IMPORT_AVAILABLE)
    GUROBI_IMPORT_ERROR = opf65.GUROBI_IMPORT_ERROR
except Exception as exc:  # noqa: BLE001
    opf65 = None
    phys64 = None
    v7463 = None
    base = None
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False
    GUROBI_IMPORT_ERROR = repr(exc)

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_SCRIPT_NAME = Path(__file__).name
BASE_RERUN_Q50_FILE = "Po_flex_domain_v7_4_6_6_rerun_opf_q50_reference_compare.py"
BASE_FILE_NAME = getattr(opf65, "BASE_FILE_NAME", "Po_flex_domain_v7_4_6_5_vertex_opf_feasibility.py")
PREVIOUS_OPF_FEASIBILITY_FILE = "Po_flex_domain_v7_4_6_5_vertex_opf_feasibility.py"

# =========================
# Rerun OPF q50 reference config
# =========================
RUN_RERUN_OPF_Q50_REFERENCE = True
RERUN_OPF_Q50_N_MC = 2500
RERUN_OPF_Q50_SEED = 8
RERUN_OPF_Q50_FORCE_RECOMPUTE = True
RERUN_OPF_Q50_USE_EXISTING_MC_REALIZATIONS_IF_AVAILABLE = True
RERUN_OPF_Q50_SAVE_CACHE = True
RERUN_OPF_Q50_QUANTILE = 0.50
RERUN_OPF_Q50_SMOKE_N_MC = None

COMPARE_WITH_BPINN_DETERMINISTIC_Q50 = True
COMPARE_WITH_BPINN_POSTERIOR_Q50 = True
COMPARE_WITH_CACHE_MC_Q50_REFERENCE = True

VALIDATE_RERUN_OPF_Q50_VERTICES = True
RUN_PROJECTION_IF_INFEASIBLE = True

OPF_TIME_LIMIT_SEC = 30
OPF_OUTPUT_FLAG = 0
OPF_FEASIBILITY_TOL = 1e-6
OPF_OPTIMALITY_TOL = 1e-6

SAVE_RERUN_OPF_Q50_DETAIL = True
SAVE_RERUN_OPF_Q50_SUMMARY = True
PLOT_RERUN_OPF_Q50_COMPARE = True

N_THETA = int(getattr(phys64, "N_THETA", 12)) if phys64 is not None else 12
THETA_VALUES = np.asarray(getattr(phys64, "THETA_VALUES", np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)), dtype=float)
SAVE_PNG = bool(getattr(phys64, "SAVE_PNG", True)) if phys64 is not None else True
SAVE_PDF = bool(getattr(phys64, "SAVE_PDF", True)) if phys64 is not None else True
DPI = int(getattr(phys64, "DPI", 300)) if phys64 is not None else 300

_manifest_artifacts: List[Dict[str, object]] = []
_any_sampling_mismatch_warning = ""
_draw_realization_signature = ""
_draw_realization_selected_candidate = ""
_draw_realization_precheck_passed = False


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


def _record_artifact(seed: int | str, artifact_type: str, path: str | Path, description: str) -> None:
    path = _abs_path(path)
    _manifest_artifacts.append({"seed": seed, "artifact_type": artifact_type, "filename": path.name, "path": str(path), "description": description})
    print(f"[rerun-opf-q50-artifact] {artifact_type} seed={seed} path={path}", flush=True)


def _safe_to_csv(df: pd.DataFrame, path: str | Path, *, description: str) -> Path:
    path = _abs_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[rerun-opf-q50-save-csv] {description} -> {path}", flush=True)
    df.to_csv(path, index=False)
    if not path.exists():
        raise RuntimeError(f"CSV save failed: {path}")
    return path


def _safe_save_fig(base_path_without_ext: Path, seed: int, description: str) -> None:
    base_path_without_ext = _abs_path(base_path_without_ext)
    base_path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        png = base_path_without_ext.with_suffix(".png")
        print(f"[rerun-opf-q50-save-fig] -> {png}", flush=True)
        plt.savefig(png, dpi=DPI)
        _record_artifact(seed, "figure_png", png, description)
    if SAVE_PDF:
        pdf = base_path_without_ext.with_suffix(".pdf")
        print(f"[rerun-opf-q50-save-fig] -> {pdf}", flush=True)
        plt.savefig(pdf)
        _record_artifact(seed, "figure_pdf", pdf, description)


def _closed(poly: np.ndarray) -> np.ndarray:
    if poly is None or len(poly) == 0:
        return np.empty((0, 2), dtype=float)
    return poly if np.allclose(poly[0], poly[-1]) else np.vstack([poly, poly[0]])


def _polygon_area(poly: np.ndarray) -> float:
    if phys64 is not None:
        return float(phys64.polygon_area(poly))
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or len(poly) < 3:
        return np.nan
    return float(0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))))


def _strict_vertices(theta_values: Iterable[float], h_values: Iterable[float]) -> Tuple[np.ndarray, Dict[str, object]]:
    if phys64 is None:
        return np.empty((0, 2), dtype=float), {"status": "failed", "fail_reason": "phys64 unavailable"}
    return phys64.strict_halfspace_polygon_vertices(theta_values, h_values)


def _hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or len(a) == 0 or len(b) == 0:
        return np.nan
    da = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return float(max(np.min(da, axis=1).max(), np.min(da, axis=0).max()))


def _q(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanquantile(arr, q)) if arr.size else np.nan


def _support_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(ref)
    if not mask.any():
        return {"mae": np.nan, "rmse": np.nan, "max_abs": np.nan, "mape_percent": np.nan}
    err = pred[mask] - ref[mask]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(np.abs(err))),
        "mape_percent": float(100.0 * np.mean(np.abs(err) / np.maximum(np.abs(ref[mask]), 1e-9))),
    }



def effective_rerun_opf_q50_n_mc() -> int:
    if RERUN_OPF_Q50_SMOKE_N_MC is None:
        return int(RERUN_OPF_Q50_N_MC)
    return int(RERUN_OPF_Q50_SMOKE_N_MC)


def _validate_realization_output(case, out):
    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise ValueError(f"_draw_realization output must be tuple/list length 4, got type={type(out)} len={len(out) if hasattr(out, '__len__') else 'NA'}")
    arrs = [np.asarray(x, dtype=float) for x in out]
    names = ["pdv", "qdv", "prv", "qrv"]
    for name, arr in zip(names, arrs):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got shape={arr.shape}")
        if arr.shape[0] != int(case.n_bus):
            raise ValueError(f"{name} length must match case.n_bus={case.n_bus}, got shape={arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    return tuple(arr.copy() for arr in arrs)


def draw_realization_compat(case, pd_mu, qd_mu, pr_mu, qr_mu):
    """Call v7.4.6.3 _draw_realization across known signature variants."""
    global _draw_realization_signature, _draw_realization_selected_candidate
    if v7463 is None or not hasattr(v7463, "_draw_realization"):
        raise RuntimeError("v7463._draw_realization is unavailable")
    sig = inspect.signature(v7463._draw_realization)
    _draw_realization_signature = str(sig)
    if not _draw_realization_selected_candidate:
        print(f"[rerun-opf-q50] _draw_realization signature = {sig}", flush=True)
    n_params = len(sig.parameters)
    candidates = []
    if n_params == 5:
        candidates.append(("case_pd_qd_pr_qr", (case, pd_mu, qd_mu, pr_mu, qr_mu)))
    elif n_params == 4:
        candidates.extend([
            ("case_pd_qd_pr", (case, pd_mu, qd_mu, pr_mu)),
            ("case_pd_pr_qr", (case, pd_mu, pr_mu, qr_mu)),
            ("case_pd_qd_qr", (case, pd_mu, qd_mu, qr_mu)),
            ("pd_qd_pr_qr", (pd_mu, qd_mu, pr_mu, qr_mu)),
        ])
    else:
        raise RuntimeError(f"Unsupported _draw_realization signature: {sig}")
    tried = []
    last_exc = None
    for name, args in candidates:
        try:
            out = v7463._draw_realization(*args)
            pdv, qdv, prv, qrv = _validate_realization_output(case, out)
            if _draw_realization_selected_candidate != name:
                print(f"[rerun-opf-q50] _draw_realization compatible call selected: {name}", flush=True)
            _draw_realization_selected_candidate = name
            return pdv, qdv, prv, qrv
        except Exception as exc:  # noqa: BLE001 - try next supported signature candidate.
            last_exc = exc
            tried.append(name)
            print(f"[rerun-opf-q50-warning] _draw_realization candidate failed: {name}, error={repr(exc)}", flush=True)
    raise RuntimeError(
        "All _draw_realization candidates failed; "
        f"signature={sig}; tried={tried}; last_exception={repr(last_exc)}"
    )


def load_selected_seed_support_cache(seed: int) -> Tuple[Dict[str, object], Path | None]:
    path = _seed_dir(seed) / f"selected_seed_{_seed_tag(seed)}_mc_supports.npz"
    if not path.exists():
        print(f"[rerun-opf-q50-warning] support cache missing: {path}", flush=True)
        return {}, None
    z = np.load(path, allow_pickle=True)
    print(f"[rerun-opf-q50] loaded selected-seed cache {path}", flush=True)
    return {k: z[k] for k in z.files}, path


def load_or_generate_mc_realizations(case, seed: int, cache: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    global _any_sampling_mismatch_warning, _draw_realization_precheck_passed
    aliases = [
        ("pd_mc", "qd_mc", "pr_mc", "qr_mc"),
        ("PD_MC", "QD_MC", "PR_MC", "QR_MC"),
        ("pd_realizations", "qd_realizations", "pr_realizations", "qr_realizations"),
    ]
    if RERUN_OPF_Q50_USE_EXISTING_MC_REALIZATIONS_IF_AVAILABLE:
        for keys in aliases:
            if all(k in cache for k in keys):
                pd_mc, qd_mc, pr_mc, qr_mc = [np.asarray(cache[k], dtype=float)[:effective_rerun_opf_q50_n_mc()] for k in keys]
                print(f"[rerun-opf-q50] using MC realizations from cache keys={keys}", flush=True)
                return pd_mc, qd_mc, pr_mc, qr_mc, "selected_seed_support_cache_realizations", ""
    if {"pd_mu", "qd_mu", "pr_mu", "qr_mu"}.issubset(cache.keys()):
        pd_mu, qd_mu, pr_mu, qr_mu = [np.asarray(cache[k], dtype=float) for k in ["pd_mu", "qd_mu", "pr_mu", "qr_mu"]]
        source = "resampled_from_cached_seed_means"
    else:
        pd_mu, qd_mu, pr_mu, qr_mu = base.draw_external_scenario_by_seed(case, seed)
        source = "resampled_from_base_draw_external_scenario_by_seed"
    _any_sampling_mismatch_warning = (
        "Cache did not contain explicit MC source/load realizations; regenerated realizations with "
        "the v7.4.6.3 _draw_realization logic and np.random.seed(seed+100000)."
    )
    print(f"[rerun-opf-q50-warning] {_any_sampling_mismatch_warning}", flush=True)
    np.random.seed(seed + 100000)
    effective_n_mc = effective_rerun_opf_q50_n_mc()
    print(f"[rerun-opf-q50] effective_n_mc={effective_n_mc} smoke_test_mode={RERUN_OPF_Q50_SMOKE_N_MC is not None}", flush=True)
    test_pdv, test_qdv, test_prv, test_qrv = draw_realization_compat(case, pd_mu, qd_mu, pr_mu, qr_mu)
    _draw_realization_precheck_passed = True
    print("[rerun-opf-q50] draw_realization_compat precheck passed; using first realization as mc_idx=0", flush=True)
    pd_rows, qd_rows, pr_rows, qr_rows = [test_pdv], [test_qdv], [test_prv], [test_qrv]
    for mc_idx in range(1, effective_n_mc):
        pdv, qdv, prv, qrv = draw_realization_compat(case, pd_mu, qd_mu, pr_mu, qr_mu)
        pd_rows.append(pdv)
        qd_rows.append(qdv)
        pr_rows.append(prv)
        qr_rows.append(qrv)
    return np.asarray(pd_rows), np.asarray(qd_rows), np.asarray(pr_rows), np.asarray(qr_rows), source, _any_sampling_mismatch_warning


def solve_support_opf_detail(case, pdv, qdv, prv, qrv, alpha: float, beta: float) -> Dict[str, object]:
    if not GUROBI_AVAILABLE:
        return {"ok": False, "status": "GUROBI_UNAVAILABLE", "status_code": -1, "solve_time_sec": 0.0}
    t0 = time.perf_counter()
    try:
        m, var = opf65._build_pcc_opf_model(
            case,
            pdv,
            qdv,
            prv,
            qrv,
            output_flag=OPF_OUTPUT_FLAG,
            time_limit_sec=OPF_TIME_LIMIT_SEC,
            feasibility_tol=OPF_FEASIBILITY_TOL,
            optimality_tol=OPF_OPTIMALITY_TOL,
        )
        m.setObjective(float(alpha) * var["P0"] + float(beta) * var["Q0"], GRB.MAXIMIZE)
        status = opf65._optimize_with_inf_or_unbd_retry(m)
        elapsed = time.perf_counter() - t0
        if status != GRB.OPTIMAL:
            return {"ok": False, "status": opf65._status_name(status), "status_code": int(status), "solve_time_sec": elapsed}
        p0 = float(var["P0"].X)
        q0 = float(var["Q0"].X)
        return {
            "ok": True,
            "status": "OPTIMAL",
            "status_code": int(status),
            "solve_time_sec": elapsed,
            "h": float(alpha * p0 + beta * q0),
            "P0": p0,
            "Q0": q0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": "EXCEPTION", "status_code": -2, "solve_time_sec": time.perf_counter() - t0, "failure_reason": repr(exc)}


def rerun_opf_supports_for_seed(case, seed: int, pd_mc: np.ndarray, qd_mc: np.ndarray, pr_mc: np.ndarray, qr_mc: np.ndarray, theta_values: np.ndarray) -> Dict[str, np.ndarray]:
    n_mc = int(pd_mc.shape[0])
    n_theta = int(len(theta_values))
    H = np.full((n_mc, n_theta), np.nan, dtype=float)
    P0 = np.full_like(H, np.nan)
    Q0 = np.full_like(H, np.nan)
    success = np.zeros((n_mc, n_theta), dtype=bool)
    solve_time = np.full_like(H, np.nan)
    status = np.empty((n_mc, n_theta), dtype=object)
    for mc_idx in range(n_mc):
        for theta_idx, theta in enumerate(theta_values):
            alpha = float(np.cos(theta))
            beta = float(np.sin(theta))
            sol = solve_support_opf_detail(case, pd_mc[mc_idx], qd_mc[mc_idx], pr_mc[mc_idx], qr_mc[mc_idx], alpha, beta)
            status[mc_idx, theta_idx] = sol.get("status", "unknown")
            solve_time[mc_idx, theta_idx] = float(sol.get("solve_time_sec", np.nan))
            if bool(sol.get("ok", False)):
                success[mc_idx, theta_idx] = True
                H[mc_idx, theta_idx] = float(sol["h"])
                P0[mc_idx, theta_idx] = float(sol["P0"])
                Q0[mc_idx, theta_idx] = float(sol["Q0"])
            print(
                f"[rerun-opf-q50] seed={seed:04d} mc={mc_idx+1:04d}/{n_mc:04d} "
                f"theta={theta_idx:02d}/{n_theta-1:02d} status={status[mc_idx, theta_idx]}",
                flush=True,
            )
    return {"H_OPF_RERUN": H, "P0_OPF_RERUN": P0, "Q0_OPF_RERUN": Q0, "success_mask": success, "solve_time": solve_time, "status": status}


def load_or_rerun_opf_supports(case, seed: int, pd_mc: np.ndarray, qd_mc: np.ndarray, pr_mc: np.ndarray, qr_mc: np.ndarray) -> Tuple[Dict[str, np.ndarray], str, Path]:
    cache_path = _seed_dir(seed) / f"selected_seed_{_seed_tag(seed)}_rerun_opf_q50_supports.npz"
    if cache_path.exists() and not RERUN_OPF_Q50_FORCE_RECOMPUTE:
        print(f"[rerun-opf-q50] loading rerun OPF support cache {cache_path}", flush=True)
        z = np.load(cache_path, allow_pickle=True)
        return {k: z[k] for k in z.files}, "loaded_rerun_cache", cache_path
    supports = rerun_opf_supports_for_seed(case, seed, pd_mc, qd_mc, pr_mc, qr_mc, THETA_VALUES)
    H = supports["H_OPF_RERUN"]
    h_q50 = np.nanquantile(H, RERUN_OPF_Q50_QUANTILE, axis=0)
    supports.update({"theta_values": THETA_VALUES, "pd_mc": pd_mc, "qd_mc": qd_mc, "pr_mc": pr_mc, "qr_mc": qr_mc, "h_opf_rerun_q50": h_q50})
    for theta_idx, val in enumerate(h_q50):
        sr = float(np.nanmean(supports["success_mask"][:, theta_idx]))
        print(f"[rerun-opf-q50] theta={theta_idx:02d} q50={val:.8f} success_rate={sr:.4f}", flush=True)
    if RERUN_OPF_Q50_SAVE_CACHE:
        print(f"[rerun-opf-q50-save-npz] -> {cache_path}", flush=True)
        np.savez_compressed(cache_path, **supports)
        _record_artifact(seed, "npz", cache_path, "Rerun OPF q50 supports and MC realizations")
    return supports, "recomputed", cache_path


def get_bpinN_det_support(seed: int, x_mu: np.ndarray, model_map: Dict[int, dict]) -> Tuple[np.ndarray, str]:
    if not COMPARE_WITH_BPINN_DETERMINISTIC_Q50:
        return np.full(N_THETA, np.nan), "disabled"
    missing = [j for j in range(N_THETA) if model_map.get(j, {}).get("net") is None or model_map.get(j, {}).get("norm") is None]
    if missing:
        return np.full(N_THETA, np.nan), f"missing models {missing}"
    return phys64._predict_deterministic_q50_for_seed(seed, x_mu.reshape(1, -1), model_map), "ok"


def get_bpinN_posterior_supports(seed: int, x_mu: np.ndarray, model_map: Dict[int, dict]) -> Tuple[np.ndarray, str]:
    if not COMPARE_WITH_BPINN_POSTERIOR_Q50:
        return np.empty((0, N_THETA)), "disabled"
    missing = [j for j in range(N_THETA) if model_map.get(j, {}).get("net") is None or model_map.get(j, {}).get("norm") is None]
    if missing:
        return np.empty((0, N_THETA)), f"missing models {missing}"
    n_samples = int(getattr(phys64, "N_BNN_DOMAIN_PHYS_SAMPLES", 50))
    rows = []
    for sample_idx in range(n_samples):
        h, _ = phys64._posterior_sample_q50_supports(seed, sample_idx, x_mu.reshape(1, -1), model_map)
        rows.append(h)
    return np.asarray(rows, dtype=float), "ok"


def get_cache_mc_q50(cache: Dict[str, object]) -> Tuple[np.ndarray, str]:
    if not COMPARE_WITH_CACHE_MC_Q50_REFERENCE:
        return np.full(N_THETA, np.nan), "disabled"
    if "H_MC" not in cache:
        return np.full(N_THETA, np.nan), "cache missing H_MC"
    H = np.asarray(cache["H_MC"], dtype=float)
    if "success_mask" in cache:
        H = np.where(np.asarray(cache["success_mask"], dtype=bool), H, np.nan)
    return np.nanquantile(H[:, :N_THETA], RERUN_OPF_Q50_QUANTILE, axis=0), "ok"


def supports_to_poly(h: np.ndarray) -> Tuple[np.ndarray, Dict[str, object], float]:
    poly, info = _strict_vertices(THETA_VALUES, h)
    return poly, info, _polygon_area(poly)


def build_support_compare_df(seed: int, opf_supports: Dict[str, np.ndarray], h_cache: np.ndarray, h_det: np.ndarray, h_post: np.ndarray) -> pd.DataFrame:
    h_opf = np.asarray(opf_supports["h_opf_rerun_q50"], dtype=float)
    success = np.asarray(opf_supports["success_mask"], dtype=bool)
    solve_time = np.asarray(opf_supports["solve_time"], dtype=float)
    post_mean = np.nanmean(h_post, axis=0) if h_post.size else np.full(N_THETA, np.nan)
    post_q05 = np.nanquantile(h_post, 0.05, axis=0) if h_post.size else np.full(N_THETA, np.nan)
    post_q50 = np.nanquantile(h_post, 0.50, axis=0) if h_post.size else np.full(N_THETA, np.nan)
    post_q95 = np.nanquantile(h_post, 0.95, axis=0) if h_post.size else np.full(N_THETA, np.nan)
    rows = []
    for j, theta in enumerate(THETA_VALUES):
        denom = max(abs(h_opf[j]), 1e-9) if np.isfinite(h_opf[j]) else np.nan
        rows.append({
            "seed": seed,
            "theta_idx": j,
            "theta": float(theta),
            "alpha": float(np.cos(theta)),
            "beta": float(np.sin(theta)),
            "h_opf_rerun_q50": float(h_opf[j]),
            "h_cache_mc_q50": float(h_cache[j]) if np.isfinite(h_cache[j]) else np.nan,
            "h_bpinN_det_q50": float(h_det[j]) if np.isfinite(h_det[j]) else np.nan,
            "h_bpinN_post_q50_mean": float(post_mean[j]) if np.isfinite(post_mean[j]) else np.nan,
            "h_bpinN_post_q50_q05": float(post_q05[j]) if np.isfinite(post_q05[j]) else np.nan,
            "h_bpinN_post_q50_q50": float(post_q50[j]) if np.isfinite(post_q50[j]) else np.nan,
            "h_bpinN_post_q50_q95": float(post_q95[j]) if np.isfinite(post_q95[j]) else np.nan,
            "abs_error_bpinN_det_vs_opf": abs(float(h_det[j] - h_opf[j])) if np.isfinite(h_det[j]) and np.isfinite(h_opf[j]) else np.nan,
            "rel_error_bpinN_det_vs_opf_percent": 100.0 * abs(float(h_det[j] - h_opf[j])) / denom if np.isfinite(h_det[j]) and np.isfinite(denom) else np.nan,
            "abs_error_cache_mc_vs_rerun_opf": abs(float(h_cache[j] - h_opf[j])) if np.isfinite(h_cache[j]) and np.isfinite(h_opf[j]) else np.nan,
            "rel_error_cache_mc_vs_rerun_opf_percent": 100.0 * abs(float(h_cache[j] - h_opf[j])) / denom if np.isfinite(h_cache[j]) and np.isfinite(denom) else np.nan,
            "posterior_contains_opf_q50": bool(np.isfinite(post_q05[j]) and post_q05[j] <= h_opf[j] <= post_q95[j]),
            "success_rate_opf_rerun": float(np.nanmean(success[:, j])),
            "mean_solve_time_sec": float(np.nanmean(solve_time[:, j])),
        })
    return pd.DataFrame(rows)


def validate_rerun_vertices(seed: int, case, pd_mu, qd_mu, pr_mu, qr_mu, poly: np.ndarray, poly_info: Dict[str, object], area: float) -> pd.DataFrame:
    rows = []
    if not VALIDATE_RERUN_OPF_Q50_VERTICES:
        return pd.DataFrame(rows)
    for vidx, (p0, q0) in enumerate(np.asarray(poly, dtype=float)):
        strict = opf65.solve_pcc_fixed_opf_feasibility(case, pd_mu, qd_mu, pr_mu, qr_mu, float(p0), float(q0), time_limit_sec=OPF_TIME_LIMIT_SEC, feasibility_tol=OPF_FEASIBILITY_TOL, output_flag=OPF_OUTPUT_FLAG)
        proj = {"projection_status": "not_run", "projection_feasible": False, "p0_projected": np.nan, "q0_projected": np.nan, "delta_p0_projected": np.nan, "delta_q0_projected": np.nan, "projection_l1_distance": 0.0 if strict.get("opf_feasible") else np.nan, "projection_l2_distance": 0.0 if strict.get("opf_feasible") else np.nan}
        if (not bool(strict.get("opf_feasible", False))) and RUN_PROJECTION_IF_INFEASIBLE:
            proj = opf65.solve_nearest_feasible_pq_projection(case, pd_mu, qd_mu, pr_mu, qr_mu, float(p0), float(q0), objective="l1_pq")
        print(
            f"[vertex-opf] rerun_opf_q50 vertex={vidx+1:02d}/{len(poly):02d} "
            f"feasible={strict.get('opf_feasible')} projection_l1={proj.get('projection_l1_distance', np.nan)}",
            flush=True,
        )
        rows.append({
            "seed": seed,
            "source_polygon": "rerun_opf_q50_reference",
            "vertex_idx": vidx,
            "P0": float(p0),
            "Q0": float(q0),
            "polygon_area": area,
            "max_halfspace_violation": float(poly_info.get("max_halfspace_violation", np.nan)),
            "opf_status": strict.get("opf_status"),
            "opf_feasible": bool(strict.get("opf_feasible", False)),
            "opf_solve_time_sec": strict.get("opf_solve_time_sec", np.nan),
            "opf_max_constraint_residual": strict.get("opf_max_constraint_residual", np.nan),
            "projection_status": proj.get("projection_status"),
            "projection_feasible": bool(proj.get("projection_feasible", False)),
            "p0_projected": proj.get("p0_projected", np.nan),
            "q0_projected": proj.get("q0_projected", np.nan),
            "delta_p0_projected": proj.get("delta_p0_projected", np.nan),
            "delta_q0_projected": proj.get("delta_q0_projected", np.nan),
            "projection_l1_distance": proj.get("projection_l1_distance", np.nan),
            "projection_l2_distance": proj.get("projection_l2_distance", np.nan),
        })
    return pd.DataFrame(rows)


def make_domain_summary(seed: int, n_mc: int, opf_supports: Dict[str, np.ndarray], h_opf: np.ndarray, h_cache: np.ndarray, h_det: np.ndarray, h_post: np.ndarray, poly_opf: np.ndarray, poly_cache: np.ndarray, poly_det: np.ndarray) -> pd.DataFrame:
    det_metrics = _support_metrics(h_det, h_opf)
    area_opf = _polygon_area(poly_opf)
    area_cache = _polygon_area(poly_cache)
    area_det = _polygon_area(poly_det)
    post_contains = []
    if h_post.size:
        q05 = np.nanquantile(h_post, 0.05, axis=0)
        q95 = np.nanquantile(h_post, 0.95, axis=0)
        post_contains = [bool(np.isfinite(q05[j]) and q05[j] <= h_opf[j] <= q95[j]) for j in range(N_THETA)]
    return pd.DataFrame([{
        "seed": seed,
        "n_theta": N_THETA,
        "n_mc": n_mc,
        "opf_rerun_success_rate": float(np.nanmean(opf_supports["success_mask"])),
        "support_mae_bpinN_det_vs_opf": det_metrics["mae"],
        "support_rmse_bpinN_det_vs_opf": det_metrics["rmse"],
        "support_max_abs_error_bpinN_det_vs_opf": det_metrics["max_abs"],
        "support_mape_bpinN_det_vs_opf_percent": det_metrics["mape_percent"],
        "area_opf_rerun_q50": area_opf,
        "area_cache_mc_q50": area_cache,
        "area_bpinN_det_q50": area_det,
        "area_error_bpinN_det_vs_opf_percent": 100.0 * abs(area_det - area_opf) / max(abs(area_opf), 1e-9) if np.isfinite(area_det) and np.isfinite(area_opf) else np.nan,
        "hausdorff_bpinN_det_vs_opf": _hausdorff(poly_det, poly_opf),
        "hausdorff_cache_mc_vs_opf": _hausdorff(poly_cache, poly_opf),
        "posterior_support_coverage_rate": float(np.mean(post_contains)) if post_contains else np.nan,
    }])


def plot_support_compare(seed: int, compare_df: pd.DataFrame) -> None:
    tag = _seed_tag(seed)
    seed_dir = _seed_dir(seed)
    x = compare_df["theta_idx"].to_numpy(int)
    plt.figure(figsize=(9, 4.8))
    plt.plot(x, compare_df["h_opf_rerun_q50"], marker="o", linewidth=2.2, label="rerun OPF q50")
    if compare_df["h_cache_mc_q50"].notna().any():
        plt.plot(x, compare_df["h_cache_mc_q50"], marker="s", linestyle="--", label="cache MC q50")
    if compare_df["h_bpinN_det_q50"].notna().any():
        plt.plot(x, compare_df["h_bpinN_det_q50"], marker="x", linestyle="-.", label="BPINN deterministic q50")
    if compare_df["h_bpinN_post_q50_q05"].notna().any():
        plt.fill_between(x, compare_df["h_bpinN_post_q50_q05"], compare_df["h_bpinN_post_q50_q95"], color="#fb923c", alpha=0.18, label="BPINN posterior q05-q95")
    plt.xlabel("theta_idx")
    plt.ylabel("h_theta q50")
    plt.title(f"Seed {seed}: q50 support comparison")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    _safe_save_fig(seed_dir / f"selected_seed_{tag}_rerun_opf_q50_support_compare", seed, "Rerun OPF q50 support comparison")
    plt.close()


def plot_domain_compare(seed: int, poly_opf: np.ndarray, poly_cache: np.ndarray, poly_det: np.ndarray, posterior_polys: List[np.ndarray], vertex_df: pd.DataFrame) -> None:
    tag = _seed_tag(seed)
    seed_dir = _seed_dir(seed)
    plt.figure(figsize=(7.4, 6.5))
    for idx, poly in enumerate(posterior_polys[:50]):
        if len(poly) >= 3:
            pp = _closed(poly)
            plt.plot(pp[:, 0], pp[:, 1], color="#94a3b8", alpha=0.20, linewidth=0.7, label="BPINN posterior q50 polygons" if idx == 0 else None)
    for poly, color, style, label, lw in [
        (poly_opf, "#dc2626", "-", "rerun OPF q50 polygon", 2.4),
        (poly_cache, "#2563eb", "--", "cache MC q50 polygon", 1.8),
        (poly_det, "#f97316", "-.", "BPINN deterministic q50 polygon", 1.8),
    ]:
        if len(poly) >= 3:
            pp = _closed(poly)
            plt.plot(pp[:, 0], pp[:, 1], color=color, linestyle=style, linewidth=lw, label=label)
    if not vertex_df.empty:
        feas = vertex_df[vertex_df["opf_feasible"].astype(bool)]
        infeas = vertex_df[~vertex_df["opf_feasible"].astype(bool)]
        if not feas.empty:
            plt.scatter(feas["P0"], feas["Q0"], color="#16a34a", marker="o", s=35, label="OPF feasible vertices")
        if not infeas.empty:
            l1 = pd.to_numeric(infeas["projection_l1_distance"], errors="coerce").fillna(0.0).to_numpy(float)
            plt.scatter(infeas["P0"], infeas["Q0"], color="#7f1d1d", marker="x", s=45 + 120 * l1 / max(np.nanmax(l1), 1e-12), label="OPF infeasible vertices")
            if {"p0_projected", "q0_projected"}.issubset(infeas.columns):
                for _, row in infeas.iterrows():
                    if np.isfinite(row.get("p0_projected", np.nan)) and np.isfinite(row.get("q0_projected", np.nan)):
                        plt.arrow(row["P0"], row["Q0"], row["p0_projected"] - row["P0"], row["q0_projected"] - row["Q0"], color="#7f1d1d", alpha=0.45, length_includes_head=True, head_width=0.01)
    plt.xlabel("P_0")
    plt.ylabel("Q_0")
    plt.title(f"Seed {seed}: q50 domain comparison")
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    _safe_save_fig(seed_dir / f"selected_seed_{tag}_rerun_opf_q50_domain_compare", seed, "Rerun OPF q50 domain comparison")
    plt.close()


def plot_projection_distribution(seed: int, vertex_df: pd.DataFrame) -> None:
    tag = _seed_tag(seed)
    seed_dir = _seed_dir(seed)
    vals = pd.to_numeric(vertex_df.get("projection_l1_distance", pd.Series(dtype=float)), errors="coerce").dropna()
    vals = vals[vals > 0]
    plt.figure(figsize=(7.0, 4.5))
    if len(vals):
        plt.hist(vals, bins=min(30, max(5, int(np.sqrt(len(vals))))), color="#dc2626", alpha=0.82)
        plt.xlabel("projection_l1_distance")
    else:
        plt.text(0.5, 0.5, "No infeasible rerun OPF q50 vertices", ha="center", va="center", transform=plt.gca().transAxes)
    plt.title(f"Seed {seed}: rerun OPF q50 projection distances")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    _safe_save_fig(seed_dir / f"selected_seed_{tag}_rerun_opf_q50_projection_distance_distribution", seed, "Rerun OPF q50 projection distance distribution")
    plt.close()


def posterior_polygons_from_supports(h_post: np.ndarray) -> List[np.ndarray]:
    out = []
    for row in np.asarray(h_post, dtype=float):
        poly, info = _strict_vertices(THETA_VALUES, row)
        if info.get("status") == "ok":
            out.append(poly)
    return out


def write_manifest(seed: int, manifest: Dict[str, object]) -> None:
    path = _seed_dir(seed) / f"selected_seed_{_seed_tag(seed)}_rerun_opf_q50_reference_compare_manifest.json"
    print(f"[rerun-opf-q50-save-json] manifest -> {path}", flush=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _record_artifact(seed, "json", path, "Rerun OPF q50 reference comparison manifest")


def empty_outputs(seed: int, reason: str) -> Dict[str, object]:
    seed_dir = _seed_dir(seed)
    tag = _seed_tag(seed)
    compare_df = pd.DataFrame()
    summary_df = pd.DataFrame([{"seed": seed, "n_theta": N_THETA, "n_mc": RERUN_OPF_Q50_N_MC, "opf_rerun_success_rate": np.nan, "support_mae_bpinN_det_vs_opf": np.nan, "support_rmse_bpinN_det_vs_opf": np.nan, "support_max_abs_error_bpinN_det_vs_opf": np.nan, "support_mape_bpinN_det_vs_opf_percent": np.nan, "area_opf_rerun_q50": np.nan, "area_cache_mc_q50": np.nan, "area_bpinN_det_q50": np.nan, "area_error_bpinN_det_vs_opf_percent": np.nan, "hausdorff_bpinN_det_vs_opf": np.nan, "hausdorff_cache_mc_vs_opf": np.nan, "posterior_support_coverage_rate": np.nan}])
    vertex_df = pd.DataFrame()
    for df, name, desc in [
        (compare_df, f"selected_seed_{tag}_rerun_opf_q50_support_compare_by_theta.csv", "empty support compare"),
        (summary_df, f"selected_seed_{tag}_rerun_opf_q50_domain_compare_summary.csv", "empty domain compare summary"),
        (vertex_df, f"selected_seed_{tag}_rerun_opf_q50_vertex_opf_feasibility_detail.csv", "empty rerun OPF q50 vertex feasibility"),
    ]:
        p = seed_dir / name
        _safe_to_csv(df, p, description=f"{desc}: {reason}")
        _record_artifact(seed, "csv", p, desc)
    manifest = {
        "new_file_name": NEW_SCRIPT_NAME,
        "base_file_name": BASE_FILE_NAME,
        "previous_opf_feasibility_file": PREVIOUS_OPF_FEASIBILITY_FILE,
        "selected_seed": seed,
        "RERUN_OPF_Q50_N_MC": RERUN_OPF_Q50_N_MC,
        "RERUN_OPF_Q50_SMOKE_N_MC": RERUN_OPF_Q50_SMOKE_N_MC,
        "smoke_test_mode": RERUN_OPF_Q50_SMOKE_N_MC is not None,
        "effective_n_mc": effective_rerun_opf_q50_n_mc(),
        "RERUN_OPF_Q50_QUANTILE": RERUN_OPF_Q50_QUANTILE,
        "RERUN_OPF_Q50_FORCE_RECOMPUTE": RERUN_OPF_Q50_FORCE_RECOMPUTE,
        "source_load_realization_source": "not_available",
        "retrained_model": False,
        "rebuilt_dataset": False,
        "reran_alltheta_mc2500": False,
        "reran_opf_q50_supports": False,
        "used_nn_recovery_as_feasibility_certificate": False,
        "opf_vertex_feasibility_check": False,
        "projection_check": False,
        "gurobi_available": GUROBI_AVAILABLE,
        "skip_reason": reason,
        "outputs": _manifest_artifacts,
        "any_sampling_mismatch_warning": _any_sampling_mismatch_warning,
        "draw_realization_signature": _draw_realization_signature,
        "draw_realization_selected_candidate": _draw_realization_selected_candidate,
        "draw_realization_precheck_passed": _draw_realization_precheck_passed,
    }
    write_manifest(seed, manifest)
    return summary_df.iloc[0].to_dict()


def run_seed(seed: int) -> Dict[str, object]:
    if not GUROBI_AVAILABLE:
        return empty_outputs(seed, f"gurobi unavailable: {GUROBI_IMPORT_ERROR}")
    case = base.build_ieee33_case()
    base.ALL_THETA_LIST = list(range(N_THETA))
    if v7463 is not None:
        base.ALL_THETA_RESULT_DIR = v7463.ALL_THETA_RESULT_DIR
    model_map = base.load_existing_all_theta_models(case)
    cache, _ = load_selected_seed_support_cache(seed)
    pd_mc, qd_mc, pr_mc, qr_mc, realization_source, sampling_warning = load_or_generate_mc_realizations(case, seed, cache)
    supports, support_source, rerun_cache_path = load_or_rerun_opf_supports(case, seed, pd_mc, qd_mc, pr_mc, qr_mc)
    h_opf = np.asarray(supports["h_opf_rerun_q50"], dtype=float)
    poly_opf, poly_opf_info, area_opf = supports_to_poly(h_opf)

    if {"x_mu"}.issubset(cache.keys()):
        x_mu = np.asarray(cache["x_mu"], dtype=float)
    elif {"pd_mu", "pr_mu"}.issubset(cache.keys()):
        x_mu = base.make_feature_vector(case, np.asarray(cache["pd_mu"], dtype=float), np.asarray(cache["pr_mu"], dtype=float))
    else:
        pd_mu_context, _, pr_mu_context, _ = base.draw_external_scenario_by_seed(case, seed)
        x_mu = base.make_feature_vector(case, pd_mu_context, pr_mu_context)
    h_det, det_status = get_bpinN_det_support(seed, x_mu, model_map)
    h_post, post_status = get_bpinN_posterior_supports(seed, x_mu, model_map)
    h_cache, cache_status = get_cache_mc_q50(cache)
    poly_det, poly_det_info, _ = supports_to_poly(h_det)
    poly_cache, poly_cache_info, _ = supports_to_poly(h_cache)
    posterior_polys = posterior_polygons_from_supports(h_post)

    compare_df = build_support_compare_df(seed, supports, h_cache, h_det, h_post)
    domain_df = make_domain_summary(seed, int(pd_mc.shape[0]), supports, h_opf, h_cache, h_det, h_post, poly_opf, poly_cache, poly_det)
    print(f"[compare] support_mae_bpinN_det_vs_opf={domain_df.iloc[0]['support_mae_bpinN_det_vs_opf']}", flush=True)

    if {"pd_mu", "qd_mu", "pr_mu", "qr_mu"}.issubset(cache.keys()):
        pd_mu, qd_mu, pr_mu, qr_mu = [np.asarray(cache[k], dtype=float) for k in ["pd_mu", "qd_mu", "pr_mu", "qr_mu"]]
    else:
        pd_mu, qd_mu, pr_mu, qr_mu = base.draw_external_scenario_by_seed(case, seed)
    vertex_df = validate_rerun_vertices(seed, case, pd_mu, qd_mu, pr_mu, qr_mu, poly_opf, poly_opf_info, area_opf)

    seed_dir = _seed_dir(seed)
    tag = _seed_tag(seed)
    feasible_rate = float(vertex_df["opf_feasible"].astype(bool).mean()) if not vertex_df.empty else np.nan
    max_proj = _q(vertex_df.get("projection_l1_distance", pd.Series(dtype=float)), 1.0) if not vertex_df.empty else np.nan
    domain_df.loc[:, "rerun_opf_q50_vertex_feasible_rate"] = feasible_rate
    domain_df.loc[:, "rerun_opf_q50_max_projection_l1_distance"] = max_proj

    paths = [
        (compare_df, seed_dir / f"selected_seed_{tag}_rerun_opf_q50_support_compare_by_theta.csv", "Rerun OPF q50 support comparison by theta"),
        (domain_df, seed_dir / f"selected_seed_{tag}_rerun_opf_q50_domain_compare_summary.csv", "Rerun OPF q50 domain comparison summary"),
        (vertex_df, seed_dir / f"selected_seed_{tag}_rerun_opf_q50_vertex_opf_feasibility_detail.csv", "Rerun OPF q50 vertex OPF feasibility detail"),
    ]
    for df, path, desc in paths:
        _safe_to_csv(df, path, description=desc)
        _record_artifact(seed, "csv", path, desc)
    if PLOT_RERUN_OPF_Q50_COMPARE:
        plot_support_compare(seed, compare_df)
        plot_domain_compare(seed, poly_opf, poly_cache, poly_det, posterior_polys, vertex_df)
        plot_projection_distribution(seed, vertex_df)

    manifest = {
        "new_file_name": NEW_SCRIPT_NAME,
        "base_file_name": BASE_FILE_NAME,
        "previous_opf_feasibility_file": PREVIOUS_OPF_FEASIBILITY_FILE,
        "selected_seed": seed,
        "RERUN_OPF_Q50_N_MC": RERUN_OPF_Q50_N_MC,
        "RERUN_OPF_Q50_SMOKE_N_MC": RERUN_OPF_Q50_SMOKE_N_MC,
        "smoke_test_mode": RERUN_OPF_Q50_SMOKE_N_MC is not None,
        "effective_n_mc": effective_rerun_opf_q50_n_mc(),
        "RERUN_OPF_Q50_QUANTILE": RERUN_OPF_Q50_QUANTILE,
        "RERUN_OPF_Q50_FORCE_RECOMPUTE": RERUN_OPF_Q50_FORCE_RECOMPUTE,
        "source_load_realization_source": realization_source,
        "retrained_model": False,
        "rebuilt_dataset": False,
        "reran_alltheta_mc2500": False,
        "reran_opf_q50_supports": support_source == "recomputed",
        "used_nn_recovery_as_feasibility_certificate": False,
        "opf_vertex_feasibility_check": bool(VALIDATE_RERUN_OPF_Q50_VERTICES),
        "projection_check": bool(RUN_PROJECTION_IF_INFEASIBLE),
        "gurobi_available": GUROBI_AVAILABLE,
        "outputs": _manifest_artifacts,
        "any_sampling_mismatch_warning": sampling_warning,
        "draw_realization_signature": _draw_realization_signature,
        "draw_realization_selected_candidate": _draw_realization_selected_candidate,
        "draw_realization_precheck_passed": _draw_realization_precheck_passed,
        "deterministic_bpinN_status": det_status,
        "posterior_bpinN_status": post_status,
        "cache_mc_q50_status": cache_status,
        "opf_q50_polygon_status": poly_opf_info,
        "cache_mc_q50_polygon_status": poly_cache_info,
        "deterministic_bpinN_polygon_status": poly_det_info,
        "rerun_cache_path": str(rerun_cache_path),
    }
    write_manifest(seed, manifest)
    print(
        f"[rerun-opf-q50-done] seed={seed:04d} feasible_rate={feasible_rate} max_projection_l1={max_proj} "
        f"area_opf={domain_df.iloc[0]['area_opf_rerun_q50']} area_bpinN_det={domain_df.iloc[0]['area_bpinN_det_q50']}",
        flush=True,
    )
    return domain_df.iloc[0].to_dict()


def main() -> None:
    print("[v7.4.6.6.1-rerun-opf-q50-reference-compare-fix-draw] start", flush=True)
    print(f"[rerun-opf-q50-config] RUN_RERUN_OPF_Q50_REFERENCE={RUN_RERUN_OPF_Q50_REFERENCE}", flush=True)
    print(f"[rerun-opf-q50-config] seed={RERUN_OPF_Q50_SEED} n_mc={RERUN_OPF_Q50_N_MC} smoke_n_mc={RERUN_OPF_Q50_SMOKE_N_MC} effective_n_mc={effective_rerun_opf_q50_n_mc()} force={RERUN_OPF_Q50_FORCE_RECOMPUTE}", flush=True)
    print(f"[rerun-opf-q50-config] quantile={RERUN_OPF_Q50_QUANTILE} OPF_TIME_LIMIT_SEC={OPF_TIME_LIMIT_SEC}", flush=True)
    if not RUN_RERUN_OPF_Q50_REFERENCE:
        print("[rerun-opf-q50] disabled; exiting", flush=True)
        return
    _selected_output_dir().mkdir(parents=True, exist_ok=True)
    summary = run_seed(int(RERUN_OPF_Q50_SEED))
    all_summary = pd.DataFrame([summary])
    all_path = _selected_output_dir() / "rerun_opf_q50_reference_compare_all_selected_seed_summary.csv"
    _safe_to_csv(all_summary, all_path, description="All selected seed rerun OPF q50 reference comparison summary")
    _record_artifact("all", "csv", all_path, "All selected seed rerun OPF q50 reference comparison summary")
    print("[rerun-opf-q50-explain] 本版本强制用原始 OPF 重新计算 selected seed 的 q50 support reference。", flush=True)
    print("[rerun-opf-q50-explain] MC q50 half-space polygon 顶点再用 fixed-P0/Q0 OPF feasibility 检查；若顶点不可行，说明有限方向 q50 half-space polygon 本身可能存在外近似不可行。", flush=True)
    print("[rerun-opf-q50-explain] 本版本不训练 BPINN、不重建训练数据集、不重跑 all-theta MC2500 external evaluation，也不使用 NN recovery 作为 feasibility certificate。", flush=True)
    print("[rerun-opf-q50-done] summary:", flush=True)
    print(all_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
