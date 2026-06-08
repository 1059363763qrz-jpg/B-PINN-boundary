"""v7.4.6 selected external-seed P0-Q0 probability flex-domain plots.

Visualization/evaluation utility built on the existing v7.4.4 qcal all-theta
MC2500 workflow and v7.4.5 selected-seed visualization idea.

Windows PowerShell:
    py Po_flex_domain_v7_4_6_3_no_pcc_qcal_selected_seed_flex_domain_plots_with_cdf_bands.py

Usage notes:
- Edit SELECTED_EXTERNAL_SEEDS to choose the external seed(s) to plot, e.g.
  SELECTED_EXTERNAL_SEEDS = [8, 2, 19, 20, 13].
- Edit MC_DOMAIN_QUANTILE_N to control the MC support sample count.
- The first run for a seed may need MC-OPF support computation. If a support
  cache exists and RECOMPUTE_MC_SUPPORTS=False, the cache is used directly.
- If no cache exists, the script prints a clear warning and recomputes MC-OPF
  supports for only the selected seed(s); set RECOMPUTE_MC_SUPPORTS=True to
  force recomputation even when a cache exists.

This script does not train models, rebuild the training dataset, or modify model
artifacts. It only performs visualization-oriented support-to-polygon flex-domain
reconstruction for selected external seeds. It writes CSV logs, npz support
caches, flex-domain plots, per-theta CDF plots with Bayesian posterior bands, and a visualization manifest.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Po_flex_domain_v7_4_4_no_pcc_data1000_mc60_qcal_alltheta_mc2500_configurable_seeds as base

SCRIPT_DIR = Path(__file__).resolve().parent


# =========================
# Selected-seed flex-domain plot config
# =========================
RUN_SELECTED_SEED_FLEX_DOMAIN_PLOTS = True

AVAILABLE_EXTERNAL_SEEDS = list(range(1, 21))
SELECTED_EXTERNAL_SEEDS = [8]
SEED_SELECTION_MODE = "manual"  # "manual" | "best_mean_arms" | "best_max_arms" | "top_n_mean_arms"
AUTO_SELECT_TOP_N = 3

DOMAIN_QUANTILE_TAUS = [0.05, 0.50, 0.95]
MC_DOMAIN_QUANTILE_N = 2500
MC_SUPPORT_MODE = "joint_realization_cloud"
# "joint_realization_cloud":
#   Same MC realization is used across all theta directions.
#   This is preferred for drawing P0-Q0 realization cloud and polygon domains.
#   It may not exactly reproduce the per-theta independent MC samples used in v7.4.4 CDF/ARMS evaluation.
# Future optional mode:
# "match_v744_per_theta":
#   Reserved for strict per-theta external CDF reproduction.
MC_CLOUD_MAX_PLOTS = 120

RECOMPUTE_MC_SUPPORTS = False
SAVE_SELECTED_SEED_SUPPORT_CACHE = True
SAVE_LONG_MC_SUPPORT_CSV = False

PLOT_MC_CLOUD = True
PLOT_MC_QUANTILE_BAND = True
PLOT_BPINN_QUANTILE_BAND = True
PLOT_Q50_OVERLAY = True

SAVE_PNG = True
SAVE_PDF = True
DPI = 300

N_THETA = 12

PLOT_THETA_CDFS = True
SAVE_INDIVIDUAL_THETA_CDF = True
SAVE_THETA_CDF_MONTAGE = True
CDF_GRID_N = 500
CDF_PLOT_THETA_LIST = list(range(N_THETA))
CDF_USE_SHARED_X_MARGIN = 0.05
CDF_INCLUDE_QUANTILE_MARKERS = True
CDF_INCLUDE_ARMS_IN_TITLE = True
CDF_MONTAGE_NROWS = 4
CDF_MONTAGE_NCOLS = 3

# deterministic curve: sample=False single GMM CDF.
# posterior band: sample=True Bayesian forward samples, producing CDF bands from model-parameter uncertainty.
PLOT_BPINN_CDF_POSTERIOR_BAND = True
N_BNN_CDF_SAMPLES = 50
POSTERIOR_BAND_TAUS = [0.05, 0.50, 0.95]
PLOT_BPINN_CDF_MEDIAN = True
PLOT_BPINN_CDF_DETERMINISTIC = True
PLOT_BPINN_CDF_SAMPLE_LINES = False
MAX_BPINN_CDF_SAMPLE_LINES = 10
CDF_POSTERIOR_BAND_ALPHA = 0.20
CDF_POSTERIOR_SAMPLE_ALPHA = 0.12
CDF_POSTERIOR_SAMPLE_LINEWIDTH = 0.8
SAVE_CDF_POSTERIOR_SUMMARY_CSV = True

LONG_SELECTED_FLEX_DOMAIN_OUT_DIR = (
    "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_"
    "qcal_only_comparison_mc2500_seeds01_20_combined/selected_seed_flex_domain_plots"
)
SHORT_SELECTED_FLEX_DOMAIN_OUT_DIR = "v746_selected_seed_flex_domain_plots"
USE_SHORT_OUTPUT_DIR = True
SELECTED_FLEX_DOMAIN_OUT_DIR = (
    SHORT_SELECTED_FLEX_DOMAIN_OUT_DIR if USE_SHORT_OUTPUT_DIR else LONG_SELECTED_FLEX_DOMAIN_OUT_DIR
)
COMBINED_EXTERNAL_METRICS_CSV = (
    "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_"
    "qcal_only_comparison_mc2500_seeds01_20_combined/all_theta_multiseed_external_by_theta_seed.csv"
)

ALL_THETA_RESULT_DIR = (
    "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_"
    "qcal_only_alltheta"
)
THETA_VALUES = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
SUPPORT_EPS = 1e-9
POLYGON_COORD_MAX = 1e4


_manifest_rows: List[Dict[str, object]] = []


def _abs_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else SCRIPT_DIR / p


def _ensure_dir(d: str | Path) -> Path:
    d = _abs_path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_parent(path: str | Path) -> Path:
    path = _abs_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _safe_to_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False, description: str = "") -> Path:
    path = _ensure_parent(path)
    print(f"[SaveCSV] {description} -> {path}", flush=True)
    df.to_csv(path, index=index)
    if not path.exists():
        raise RuntimeError(f"CSV save failed, file not found after write: {path}")
    print(f"[SaveCSV-done] {path}", flush=True)
    return path


def _safe_savez_compressed(path: str | Path, **kwargs) -> Path:
    path = _ensure_parent(path)
    print(f"[SaveNPZ] -> {path}", flush=True)
    np.savez_compressed(path, **kwargs)
    if not path.exists():
        raise RuntimeError(f"NPZ save failed, file not found after write: {path}")
    print(f"[SaveNPZ-done] {path}", flush=True)
    return path


def _seed_tag(seed: int) -> str:
    return f"{int(seed):04d}"


def _seed_dir(seed: int) -> Path:
    d = _abs_path(SELECTED_FLEX_DOMAIN_OUT_DIR) / f"seed_{_seed_tag(seed)}"
    d.mkdir(parents=True, exist_ok=True)
    print(f"[Path] seed_dir seed={seed}: {d}", flush=True)
    return d


def _record_artifact(artifact_type: str, seed: int | str, path: str | Path, description: str) -> None:
    path = _abs_path(path)
    _manifest_rows.append(
        {
            "artifact_type": artifact_type,
            "seed": seed,
            "filename": path.name,
            "path": str(path),
            "description": description,
        }
    )
    print(f"[Artifact] {artifact_type} seed={seed} path={path}", flush=True)


def _save_fig(base_path_without_ext: Path, seed: int, description: str, png_artifact_type: str = "figure_png", pdf_artifact_type: str = "figure_pdf") -> None:
    base_path_without_ext = _abs_path(base_path_without_ext)
    base_path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        png = base_path_without_ext.with_suffix(".png")
        print(f"[SaveFig] -> {png}", flush=True)
        plt.savefig(png, dpi=DPI)
        if not png.exists():
            raise RuntimeError(f"PNG save failed, file not found after write: {png}")
        _record_artifact(png_artifact_type, seed, png, description)
    if SAVE_PDF:
        pdf = base_path_without_ext.with_suffix(".pdf")
        print(f"[SaveFig] -> {pdf}", flush=True)
        plt.savefig(pdf)
        if not pdf.exists():
            raise RuntimeError(f"PDF save failed, file not found after write: {pdf}")
        _record_artifact(pdf_artifact_type, seed, pdf, description)


def _closed(poly: np.ndarray) -> np.ndarray:
    if poly is None or len(poly) == 0:
        return np.empty((0, 2), dtype=float)
    if np.allclose(poly[0], poly[-1]):
        return poly
    return np.vstack([poly, poly[0]])


def load_seed_ranking(metrics_csv: str | Path = COMBINED_EXTERNAL_METRICS_CSV) -> pd.DataFrame:
    """Load combined MC2500 metrics and write per-seed ranking."""
    metrics_csv = _abs_path(metrics_csv)
    if not metrics_csv.exists():
        candidates = [
            _abs_path(COMBINED_EXTERNAL_METRICS_CSV),
            _abs_path("training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_qcal_only_comparison_mc2500_seeds01_10_combined/all_theta_multiseed_external_by_theta_seed.csv"),
            _abs_path("training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_qcal_only_comparison_mc2500/all_theta_multiseed_external_by_theta_seed.csv"),
        ]
        raise FileNotFoundError(
            f"Combined external metrics CSV not found: {metrics_csv}. Candidate paths: {candidates}"
        )
    df = pd.read_csv(metrics_csv)
    required = {"seed", "theta_idx", "external_cdf_arms"}
    if not required.issubset(df.columns):
        raise ValueError(f"Metrics CSV missing columns {sorted(required - set(df.columns))}; columns={list(df.columns)}")
    rows = []
    for seed, g in df.groupby("seed"):
        arms = g["external_cdf_arms"].astype(float)
        rows.append(
            {
                "seed": int(seed),
                "mean_arms_percent": float(arms.mean()),
                "max_arms_percent": float(arms.max()),
                "median_arms_percent": float(arms.median()),
                "q90_arms_percent": float(arms.quantile(0.90)),
                "n_theta": int(g["theta_idx"].nunique()),
                "n_theta_gt_5_percent": int((arms > 5.0).sum()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["mean_arms_percent", "max_arms_percent"]).reset_index(drop=True)
    out = _abs_path(SELECTED_FLEX_DOMAIN_OUT_DIR) / "selected_seed_flex_domain_seed_ranking.csv"
    _safe_to_csv(ranking, out, index=False, description="Per-external-seed ARMS ranking used for seed selection")
    _record_artifact("csv", "all", out, "Per-external-seed ARMS ranking used for seed selection")
    print(f"[SeedSelect] available seeds = {sorted(ranking['seed'].astype(int).tolist())}", flush=True)
    return ranking


def select_external_seeds(seed_ranking: pd.DataFrame) -> List[int]:
    mode = str(SEED_SELECTION_MODE).strip().lower()
    available = set(int(s) for s in seed_ranking["seed"].values)
    if mode == "manual":
        selected = [int(s) for s in SELECTED_EXTERNAL_SEEDS if int(s) in available]
        missing = [int(s) for s in SELECTED_EXTERNAL_SEEDS if int(s) not in available]
        for seed in missing:
            print(f"[SeedSelect-warning] requested seed={seed} not found in metrics CSV", flush=True)
    elif mode == "best_mean_arms":
        selected = [int(seed_ranking.sort_values("mean_arms_percent").iloc[0]["seed"])]
    elif mode == "best_max_arms":
        selected = [int(seed_ranking.sort_values("max_arms_percent").iloc[0]["seed"])]
    elif mode == "top_n_mean_arms":
        selected = [int(s) for s in seed_ranking.sort_values("mean_arms_percent").head(AUTO_SELECT_TOP_N)["seed"]]
    else:
        raise ValueError(f"Unsupported SEED_SELECTION_MODE={SEED_SELECTION_MODE!r}")
    if not selected:
        raise RuntimeError("No valid external seeds selected. Check SELECTED_EXTERNAL_SEEDS and metrics CSV.")
    print(f"[SeedSelect] mode = {mode}", flush=True)
    print(f"[SeedSelect] selected seeds = {selected}", flush=True)
    for seed in selected:
        row = seed_ranking[seed_ranking["seed"] == seed].iloc[0]
        print(
            f"[SeedSelect] seed {seed}: mean_ARMS={row['mean_arms_percent']:.4f}, "
            f"max_ARMS={row['max_arms_percent']:.4f}, n_theta_gt5={int(row['n_theta_gt_5_percent'])}",
            flush=True,
        )
    return selected


def _draw_realization(case, pd_mu, qd_mu, pr_mu):
    std_pd = 0.10 * np.maximum(pd_mu, 1e-3)
    std_pr = 0.12 * np.maximum(pr_mu, 1e-3)
    pdv = pd_mu.copy()
    prv = pr_mu.copy()
    for i in range(case.n_bus):
        if pd_mu[i] > 1e-9:
            pdv[i] = base.sample_trunc_normal(pd_mu[i], std_pd[i], 0.0, None)
    for k, bus in enumerate(case.pv_buses):
        prv[bus] = base.sample_trunc_normal(pr_mu[bus], std_pr[bus], 0.0, float(case.pv_pmax[k]))
    qdv = qd_mu * (pdv / np.maximum(pd_mu, 1e-6))
    qrv = prv * math.tan(math.acos(case.pv_pf))
    return pdv, qdv, prv, qrv


def load_or_compute_mc_supports_for_seed(seed: int, case) -> Dict[str, object]:
    """Load cached H_MC or compute supports for one selected external seed."""
    seed_dir = _seed_dir(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = seed_dir / f"selected_seed_{_seed_tag(seed)}_mc_supports.npz"
    if cache_path.exists() and not RECOMPUTE_MC_SUPPORTS:
        print(f"[MC] seed={seed} loading support cache {cache_path}", flush=True)
        z = np.load(cache_path, allow_pickle=True)
        _record_artifact("npz_cache", seed, cache_path, "Cached selected-seed MC support matrix")
        return {k: z[k] for k in z.files}
    if not RECOMPUTE_MC_SUPPORTS:
        print(
            f"[MC-warning] seed={seed} support cache missing at {cache_path}; "
            "recomputing MC-OPF supports for this selected seed.",
            flush=True,
        )

    print(f"[MC] seed={seed} recomputing MC supports with M={MC_DOMAIN_QUANTILE_N}", flush=True)
    pd_mu, qd_mu, pr_mu, qr_mu = base.draw_external_scenario_by_seed(case, seed)
    x_mu = base.make_feature_vector(case, pd_mu, pr_mu)
    theta_values = np.asarray(THETA_VALUES, dtype=float)
    alpha_values = np.cos(theta_values)
    beta_values = np.sin(theta_values)
    H_MC = np.full((MC_DOMAIN_QUANTILE_N, N_THETA), np.nan, dtype=float)
    P0_MC = np.full_like(H_MC, np.nan)
    Q0_MC = np.full_like(H_MC, np.nan)
    success_mask = np.zeros_like(H_MC, dtype=bool)
    long_rows = []
    np.random.seed(seed + 100000)
    t0 = time.perf_counter()
    for mc_idx in range(MC_DOMAIN_QUANTILE_N):
        pdv, qdv, prv, qrv = _draw_realization(case, pd_mu, qd_mu, pr_mu)
        for theta_idx, (alpha, beta) in enumerate(zip(alpha_values, beta_values)):
            sol = base.solve_flex_support_gurobi_33bus(
                case,
                pdv,
                qdv,
                prv,
                qrv,
                float(alpha),
                float(beta),
                return_detail=True,
                stabilize_dispatch=base.STABILIZE_OPF_DISPATCH,
            )
            ok = bool(sol.get("ok", False))
            success_mask[mc_idx, theta_idx] = ok
            if ok:
                H_MC[mc_idx, theta_idx] = float(sol["h"])
                P0_MC[mc_idx, theta_idx] = float(sol["P0"])
                Q0_MC[mc_idx, theta_idx] = float(sol["Q0"])
            if SAVE_LONG_MC_SUPPORT_CSV:
                long_rows.append(
                    {
                        "seed": seed,
                        "mc_idx": mc_idx,
                        "theta_idx": theta_idx,
                        "theta_rad": float(theta_values[theta_idx]),
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "h_mc": H_MC[mc_idx, theta_idx],
                        "success": ok,
                        "p0_opt": P0_MC[mc_idx, theta_idx],
                        "q0_opt": Q0_MC[mc_idx, theta_idx],
                    }
                )
        if (mc_idx + 1) % 100 == 0 or (mc_idx + 1) == MC_DOMAIN_QUANTILE_N:
            print(
                f"[MC] seed={seed} mc={mc_idx+1}/{MC_DOMAIN_QUANTILE_N} "
                f"success_rate={float(success_mask[:mc_idx+1].mean()):.4f}",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    config = {
        "seed": seed,
        "mc_count": MC_DOMAIN_QUANTILE_N,
        "theta_count": N_THETA,
        "recompute_mc_supports": RECOMPUTE_MC_SUPPORTS,
        "elapsed_sec": elapsed,
    }
    if SAVE_SELECTED_SEED_SUPPORT_CACHE:
        _safe_savez_compressed(
            cache_path,
            seed=np.array(seed),
            theta_values=theta_values,
            alpha_values=alpha_values,
            beta_values=beta_values,
            H_MC=H_MC,
            P0_MC=P0_MC,
            Q0_MC=Q0_MC,
            success_mask=success_mask,
            mc_count=np.array(MC_DOMAIN_QUANTILE_N),
            x_mu=x_mu,
            pd_mu=pd_mu,
            qd_mu=qd_mu,
            pr_mu=pr_mu,
            qr_mu=qr_mu,
            config_json_string=np.array(json.dumps(config)),
        )
        _record_artifact("npz_cache", seed, cache_path, "Cached selected-seed MC support matrix")
    if SAVE_LONG_MC_SUPPORT_CSV:
        long_csv = seed_dir / f"selected_seed_{_seed_tag(seed)}_mc_supports_long.csv"
        _safe_to_csv(pd.DataFrame(long_rows), long_csv, index=False, description="Long-form MC support values by realization and theta")
        _record_artifact("csv", seed, long_csv, "Long-form MC support values by realization and theta")
    return {
        "seed": np.array(seed),
        "theta_values": theta_values,
        "alpha_values": alpha_values,
        "beta_values": beta_values,
        "H_MC": H_MC,
        "P0_MC": P0_MC,
        "Q0_MC": Q0_MC,
        "success_mask": success_mask,
        "mc_count": np.array(MC_DOMAIN_QUANTILE_N),
        "x_mu": x_mu,
        "pd_mu": pd_mu,
        "qd_mu": qd_mu,
        "pr_mu": pr_mu,
        "qr_mu": qr_mu,
        "config_json_string": np.array(json.dumps(config)),
    }


def predict_bpinn_quantiles_for_seed(seed: int, case, x_mu: np.ndarray, model_map: Dict[int, dict]) -> pd.DataFrame:
    rows = []
    x_mu = np.asarray(x_mu, dtype=float).reshape(1, -1)
    for theta_idx in range(N_THETA):
        theta = float(THETA_VALUES[theta_idx])
        alpha = float(np.cos(theta))
        beta = float(np.sin(theta))
        entry = model_map.get(theta_idx, {})
        if entry.get("net") is None or entry.get("norm") is None:
            print(f"[BPINN-warning] seed={seed} theta={theta_idx:02d} missing model", flush=True)
            rows.append({"seed": seed, "theta_idx": theta_idx, "theta_rad": theta, "alpha": alpha, "beta": beta, "status": "missing_model"})
            continue
        net = entry["net"]
        norm = entry["norm"]
        x_norm = (x_mu - norm["x_mu_mean"]) / norm["x_mu_std"]
        xt = base.torch.tensor(x_norm, dtype=base.torch.float32, device=base.DEVICE)
        with base.torch.no_grad():
            _, w, mu, sigma = net.forward_gmm_single(xt, sample=False)
        w_np = w.detach().cpu().numpy().reshape(-1)
        mu_np = mu.detach().cpu().numpy().reshape(-1)
        sig_np = sigma.detach().cpu().numpy().reshape(-1)
        q_norm = np.asarray(base.gmm_quantile(DOMAIN_QUANTILE_TAUS, w_np, mu_np, sig_np)).reshape(-1)
        h_mean = float(norm["h_mean"][0, 0])
        h_std = float(norm["h_std"][0, 0])
        q = h_mean + h_std * q_norm
        print(
            f"[BPINN] seed={seed} theta={theta_idx:02d} loaded model OK "
            f"q05={q[0]:.5f} q50={q[1]:.5f} q95={q[2]:.5f}",
            flush=True,
        )
        rows.append(
            {
                "seed": seed,
                "theta_idx": theta_idx,
                "theta_rad": theta,
                "alpha": alpha,
                "beta": beta,
                "bpinn_q05": float(q[0]),
                "bpinn_q50": float(q[1]),
                "bpinn_q95": float(q[2]),
                "model_path": entry.get("model_path", ""),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def support_values_to_polygon(theta_values: Iterable[float], h_values: Iterable[float]) -> np.ndarray:
    theta = np.asarray(theta_values, dtype=float).reshape(-1)
    h = np.asarray(h_values, dtype=float).reshape(-1)
    ok = np.isfinite(theta) & np.isfinite(h)
    theta = theta[ok]
    h = h[ok]
    if len(theta) < 3:
        return np.empty((0, 2), dtype=float)
    order = np.argsort(np.mod(theta, 2.0 * np.pi))
    theta = theta[order]
    h = h[order]
    alpha = np.cos(theta)
    beta = np.sin(theta)
    vertices = []
    n = len(theta)
    for i in range(n):
        j = (i + 1) % n
        A = np.array([[alpha[i], beta[i]], [alpha[j], beta[j]]], dtype=float)
        b = np.array([h[i], h[j]], dtype=float)
        try:
            v = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(v)) and np.max(np.abs(v)) <= POLYGON_COORD_MAX:
            vertices.append(v)
    if len(vertices) < 3:
        return np.empty((0, 2), dtype=float)
    return np.asarray(vertices, dtype=float)


def polygon_area(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return float("nan")
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def check_polygon_halfspace_violations(poly: np.ndarray, theta_values: np.ndarray, h_values: np.ndarray) -> dict:
    poly = np.asarray(poly, dtype=float)
    theta_values = np.asarray(theta_values, dtype=float).reshape(-1)
    h_values = np.asarray(h_values, dtype=float).reshape(-1)
    if poly.ndim != 2 or poly.shape[0] == 0:
        return {
            "max_halfspace_violation": np.nan,
            "mean_positive_violation": np.nan,
            "n_violating_vertices": 0,
            "n_vertices": 0,
        }
    alpha = np.cos(theta_values)
    beta = np.sin(theta_values)
    positive_by_vertex = []
    max_violation = 0.0
    n_violating = 0
    for p0, q0 in poly:
        residual = alpha * p0 + beta * q0 - h_values
        pos = residual[np.isfinite(residual) & (residual > 1e-6)]
        if pos.size:
            n_violating += 1
            positive_by_vertex.extend(pos.tolist())
            max_violation = max(max_violation, float(np.max(pos)))
    return {
        "max_halfspace_violation": float(max_violation),
        "mean_positive_violation": float(np.mean(positive_by_vertex)) if positive_by_vertex else 0.0,
        "n_violating_vertices": int(n_violating),
        "n_vertices": int(poly.shape[0]),
    }


def _plot_poly(poly: np.ndarray, *args, close=True, **kwargs):
    if poly is None or len(poly) < 3:
        return
    pp = _closed(poly) if close else poly
    plt.plot(pp[:, 0], pp[:, 1], *args, **kwargs)


def _fill_poly(poly: np.ndarray, *args, **kwargs):
    if poly is None or len(poly) < 3:
        return
    pp = _closed(poly)
    plt.fill(pp[:, 0], pp[:, 1], *args, **kwargs)


def plot_selected_seed_flex_domain_overlay(seed: int, polys: Dict[str, Dict[float, np.ndarray]], cloud_polys: List[np.ndarray], metrics: dict) -> List[Path]:
    seed_dir = _seed_dir(seed)
    plt.figure(figsize=(7.2, 6.4))
    if PLOT_MC_CLOUD:
        for poly in cloud_polys[:MC_CLOUD_MAX_PLOTS]:
            _plot_poly(poly, color="0.75", linewidth=0.5, alpha=0.35, label=None)
    if PLOT_MC_QUANTILE_BAND:
        _fill_poly(polys["MC"].get(0.95), color="#60a5fa", alpha=0.14, label="MC q95 filled domain")
        _plot_poly(polys["MC"].get(0.05), color="#3b82f6", linewidth=1.0, alpha=0.55, linestyle=":")
        _plot_poly(polys["MC"].get(0.95), color="#3b82f6", linewidth=1.0, alpha=0.55, linestyle=":")
    if PLOT_BPINN_QUANTILE_BAND:
        _fill_poly(polys["BPINN"].get(0.95), color="#fb923c", alpha=0.16, label="BPINN q95 filled domain")
        _plot_poly(polys["BPINN"].get(0.05), color="#f97316", linewidth=1.0, alpha=0.6, linestyle=":")
        _plot_poly(polys["BPINN"].get(0.95), color="#f97316", linewidth=1.0, alpha=0.6, linestyle=":")
    if PLOT_Q50_OVERLAY:
        _plot_poly(polys["MC"].get(0.50), color="#2563eb", linewidth=2.2, label="MC q50")
        _plot_poly(polys["BPINN"].get(0.50), color="#f97316", linewidth=2.2, linestyle="--", label="BPINN q50")
    plt.xlabel("P_0")
    plt.ylabel("Q_0")
    plt.title(
        f"Seed {seed} flex domain | MC={metrics.get('mc_count')} Ntheta={N_THETA}\n"
        f"mean ARMS={metrics.get('mean_arms_percent', np.nan):.3f}% max ARMS={metrics.get('max_arms_percent', np.nan):.3f}%"
    )
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    base_path = seed_dir / f"selected_seed_{_seed_tag(seed)}_flex_domain_overlay"
    _save_fig(base_path, seed, "Selected seed flex-domain overlay with MC cloud, q50 boundaries, and q05/q95 reference boundaries")
    plt.close()
    return [base_path.with_suffix(ext) for ext in [".png", ".pdf"]]


def plot_mc_vs_bpinn_quantile_domains(seed: int, polys: Dict[str, Dict[float, np.ndarray]]) -> None:
    seed_dir = _seed_dir(seed)
    plt.figure(figsize=(7.2, 6.4))
    colors = {0.05: "#93c5fd", 0.50: "#2563eb", 0.95: "#1d4ed8"}
    for tau in DOMAIN_QUANTILE_TAUS:
        _plot_poly(polys["MC"].get(tau), color=colors[tau], linewidth=2.0, linestyle="-", label=f"MC q{int(tau*100):02d}")
    colors_b = {0.05: "#fdba74", 0.50: "#f97316", 0.95: "#c2410c"}
    for tau in DOMAIN_QUANTILE_TAUS:
        _plot_poly(polys["BPINN"].get(tau), color=colors_b[tau], linewidth=2.0, linestyle="--", label=f"BPINN q{int(tau*100):02d}")
    if PLOT_MC_QUANTILE_BAND:
        _fill_poly(polys["MC"].get(0.95), color="#60a5fa", alpha=0.10)
    if PLOT_BPINN_QUANTILE_BAND:
        _fill_poly(polys["BPINN"].get(0.95), color="#fb923c", alpha=0.10)
    plt.xlabel("P_0")
    plt.ylabel("Q_0")
    plt.title(f"Seed {seed}: MC vs BPINN quantile domains")
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    _save_fig(seed_dir / f"selected_seed_{_seed_tag(seed)}_mc_vs_bpinn_quantile_domains", seed, "MC and BPINN q05/q50/q95 quantile domain comparison")
    plt.close()


def plot_support_quantiles_by_theta(seed: int, qdf: pd.DataFrame) -> None:
    seed_dir = _seed_dir(seed)
    plt.figure(figsize=(8.5, 4.8))
    x = qdf["theta_idx"].astype(int).values
    for suffix, color, style in [("q05", "#60a5fa", ":"), ("q50", "#2563eb", "-"), ("q95", "#1d4ed8", "--")]:
        plt.plot(x, qdf[f"mc_{suffix}"], color=color, linestyle=style, marker="o", label=f"MC {suffix}")
    for suffix, color, style in [("q05", "#fdba74", ":"), ("q50", "#f97316", "-"), ("q95", "#c2410c", "--")]:
        plt.plot(x, qdf[f"bpinn_{suffix}"], color=color, linestyle=style, marker="x", label=f"BPINN {suffix}")
    plt.xlabel("Theta index")
    plt.ylabel("h_theta")
    plt.title(f"Seed {seed}: support quantiles by theta")
    plt.xticks(range(N_THETA))
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    _save_fig(seed_dir / f"selected_seed_{_seed_tag(seed)}_support_quantiles_by_theta", seed, "Support-function quantiles by theta")
    plt.close()


def plot_polygon_area_compare(seed: int, area_rows: List[dict]) -> None:
    seed_dir = _seed_dir(seed)
    df = pd.DataFrame(area_rows)
    taus = list(DOMAIN_QUANTILE_TAUS)
    x = np.arange(len(taus))
    mc = [float(df[(df["method"] == "MC") & (df["tau"] == tau)]["area"].iloc[0]) for tau in taus]
    bp = [float(df[(df["method"] == "BPINN") & (df["tau"] == tau)]["area"].iloc[0]) for tau in taus]
    width = 0.35
    plt.figure(figsize=(7.0, 4.4))
    plt.bar(x - width / 2, mc, width, label="MC", color="#2563eb", alpha=0.85)
    plt.bar(x + width / 2, bp, width, label="BPINN", color="#f97316", alpha=0.85)
    plt.xticks(x, [f"q{int(t*100):02d}" for t in taus])
    plt.ylabel("Polygon area")
    plt.title(f"Seed {seed}: polygon area comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    _save_fig(seed_dir / f"selected_seed_{_seed_tag(seed)}_polygon_area_compare", seed, "MC vs BPINN polygon area comparison")
    plt.close()



def load_combined_theta_metrics(metrics_csv: str | Path = COMBINED_EXTERNAL_METRICS_CSV) -> pd.DataFrame:
    metrics_csv = _abs_path(metrics_csv)
    if not metrics_csv.exists():
        print(f"[CDF-warning] combined theta metrics CSV not found: {metrics_csv}", flush=True)
        return pd.DataFrame(columns=["seed", "theta_idx", "external_cdf_arms"])
    df = pd.read_csv(metrics_csv)
    required = {"seed", "theta_idx", "external_cdf_arms"}
    if not required.issubset(df.columns):
        raise ValueError(f"Theta metrics CSV missing columns {sorted(required - set(df.columns))}; columns={list(df.columns)}")
    df["seed"] = df["seed"].astype(int)
    df["theta_idx"] = df["theta_idx"].astype(int)
    df["external_cdf_arms"] = df["external_cdf_arms"].astype(float)
    return df


def normal_cdf_np(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))


def gmm_cdf_np(x_grid, weights, mu_raw, sigma_raw):
    x = np.asarray(x_grid, dtype=float).reshape(-1, 1)
    w = np.asarray(weights, dtype=float).reshape(1, -1)
    mu = np.asarray(mu_raw, dtype=float).reshape(1, -1)
    sig = np.maximum(np.asarray(sigma_raw, dtype=float).reshape(1, -1), 1e-9)
    return np.sum(w * normal_cdf_np((x - mu) / sig), axis=1)


def empirical_cdf(h_values):
    h = np.asarray(h_values, dtype=float)
    h = h[np.isfinite(h)]
    h_sorted = np.sort(h)
    if len(h_sorted) == 0:
        return h_sorted, np.asarray([], dtype=float)
    F_emp = np.arange(1, len(h_sorted) + 1, dtype=float) / float(len(h_sorted))
    return h_sorted, F_emp


def get_bpinn_gmm_raw_for_theta(seed, theta_idx, x_mu, model_map):
    entry = model_map.get(theta_idx, {})
    theta = float(THETA_VALUES[theta_idx])
    if entry.get("net") is None or entry.get("norm") is None:
        print(f"[CDF-BPINN-warning] seed={seed} theta={theta_idx:02d} missing_model", flush=True)
        return {"weights": None, "mu_raw": None, "sigma_raw": None, "h_mean": np.nan, "h_std": np.nan, "status": "missing_model", "model_path": entry.get("model_path", ""), "theta_rad": theta}
    net = entry["net"]
    norm = entry["norm"]
    x_mu = np.asarray(x_mu, dtype=float).reshape(1, -1)
    x_norm = (x_mu - norm["x_mu_mean"]) / norm["x_mu_std"]
    xt = base.torch.tensor(x_norm, dtype=base.torch.float32, device=base.DEVICE)
    with base.torch.no_grad():
        _, w, mu, sigma = net.forward_gmm_single(xt, sample=False)
    weights = w.detach().cpu().numpy().reshape(-1)
    mu_norm = mu.detach().cpu().numpy().reshape(-1)
    sigma_norm = sigma.detach().cpu().numpy().reshape(-1)
    h_mean = float(norm["h_mean"][0, 0])
    h_std = float(norm["h_std"][0, 0])
    mu_raw = h_mean + h_std * mu_norm
    sigma_raw = abs(h_std) * sigma_norm
    return {"weights": weights, "mu_raw": mu_raw, "sigma_raw": sigma_raw, "h_mean": h_mean, "h_std": h_std, "status": "ok", "model_path": entry.get("model_path", ""), "theta_rad": theta}



def get_bpinn_gmm_raw_for_theta_sampled(seed, theta_idx, x_mu, model_map, sample=True):
    entry = model_map.get(theta_idx, {})
    theta = float(THETA_VALUES[theta_idx])
    if entry.get("net") is None or entry.get("norm") is None:
        print(f"[CDF-BPINN-warning] seed={seed} theta={theta_idx:02d} missing_model", flush=True)
        return {"weights": None, "mu_raw": None, "sigma_raw": None, "h_mean": np.nan, "h_std": np.nan, "status": "missing_model", "model_path": entry.get("model_path", ""), "theta_rad": theta}
    net = entry["net"]
    norm = entry["norm"]
    x_mu = np.asarray(x_mu, dtype=float).reshape(1, -1)
    x_norm = (x_mu - norm["x_mu_mean"]) / norm["x_mu_std"]
    xt = base.torch.tensor(x_norm, dtype=base.torch.float32, device=base.DEVICE)
    with base.torch.no_grad():
        _, w, mu, sigma = net.forward_gmm_single(xt, sample=sample)
    weights = w.detach().cpu().numpy().reshape(-1)
    mu_norm = mu.detach().cpu().numpy().reshape(-1)
    sigma_norm = sigma.detach().cpu().numpy().reshape(-1)
    h_mean = float(norm["h_mean"][0, 0])
    h_std = float(norm["h_std"][0, 0])
    return {
        "weights": weights,
        "mu_raw": h_mean + h_std * mu_norm,
        "sigma_raw": abs(h_std) * sigma_norm,
        "h_mean": h_mean,
        "h_std": h_std,
        "status": "ok",
        "model_path": entry.get("model_path", ""),
        "theta_rad": theta,
    }


def compute_bpinn_cdf_posterior_band(seed, theta_idx, x_mu, model_map, x_grid):
    if not PLOT_BPINN_CDF_POSTERIOR_BAND:
        return {"status": "disabled", "n_samples": 0}
    cdf_samples = []
    failures = 0
    for r in range(int(N_BNN_CDF_SAMPLES)):
        try:
            params = get_bpinn_gmm_raw_for_theta_sampled(seed, theta_idx, x_mu, model_map, sample=True)
            if params.get("status") != "ok":
                failures += 1
                continue
            cdf_samples.append(gmm_cdf_np(x_grid, params["weights"], params["mu_raw"], params["sigma_raw"]))
        except Exception as exc:
            failures += 1
            print(f"[CDF-posterior-warning] seed={seed} theta={theta_idx:02d} sample={r} failed: {exc}", flush=True)
    if failures:
        print(f"[CDF-posterior-warning] seed={seed} theta={theta_idx:02d} failed_samples={failures}", flush=True)
    if len(cdf_samples) < 5:
        return {"status": "insufficient_samples", "n_samples": len(cdf_samples)}
    arr = np.asarray(cdf_samples, dtype=float)
    q = np.quantile(arr, POSTERIOR_BAND_TAUS, axis=0)
    out = {
        "x_grid": np.asarray(x_grid, dtype=float),
        "cdf_low": q[0],
        "cdf_med": q[1],
        "cdf_high": q[2],
        "status": "ok",
        "n_samples": int(arr.shape[0]),
        "posterior_band_mean_width": float(np.mean(q[2] - q[0])),
        "posterior_band_max_width": float(np.max(q[2] - q[0])),
    }
    if PLOT_BPINN_CDF_SAMPLE_LINES:
        out["cdf_samples"] = arr[:MAX_BPINN_CDF_SAMPLE_LINES]
    return out


def _theta_arms(seed, theta_idx, theta_metrics_df):
    if theta_metrics_df is None or theta_metrics_df.empty:
        return np.nan
    row = theta_metrics_df[(theta_metrics_df["seed"] == int(seed)) & (theta_metrics_df["theta_idx"] == int(theta_idx))]
    return float(row["external_cdf_arms"].iloc[0]) if not row.empty else np.nan


def _cdf_grid(h_mc, gmm_params):
    h = np.asarray(h_mc, dtype=float)
    h = h[np.isfinite(h)]
    vals = []
    if h.size:
        vals.extend([float(np.nanmin(h)), float(np.nanmax(h))])
    if gmm_params.get("status") == "ok":
        mu = np.asarray(gmm_params["mu_raw"], dtype=float)
        sig = np.asarray(gmm_params["sigma_raw"], dtype=float)
        vals.extend([float(np.nanmin(mu - 4.0 * sig)), float(np.nanmax(mu + 4.0 * sig))])
    if not vals:
        vals = [-1.0, 1.0]
    xmin, xmax = min(vals), max(vals)
    margin = CDF_USE_SHARED_X_MARGIN * max(xmax - xmin, 1e-6)
    return np.linspace(xmin - margin, xmax + margin, CDF_GRID_N)


def plot_theta_cdf_for_seed(seed, theta_idx, h_mc, gmm_params, theta_metrics_df, out_dir, x_mu=None, model_map=None):
    out_dir = _ensure_dir(out_dir)
    h_sorted, F_emp = empirical_cdf(h_mc)
    x_grid = _cdf_grid(h_mc, gmm_params)
    arms = _theta_arms(seed, theta_idx, theta_metrics_df)
    posterior = {"status": "disabled", "n_samples": 0}
    if x_mu is not None and model_map is not None and gmm_params.get("status") == "ok":
        posterior = compute_bpinn_cdf_posterior_band(seed, theta_idx, x_mu, model_map, x_grid)
    plt.figure(figsize=(6.6, 4.8))
    if len(h_sorted):
        plt.step(h_sorted, F_emp, where="post", color="black", linewidth=1.8, label="MC empirical CDF")
    if PLOT_BPINN_CDF_POSTERIOR_BAND and posterior.get("status") == "ok":
        if PLOT_BPINN_CDF_SAMPLE_LINES and "cdf_samples" in posterior:
            for sample_curve in posterior["cdf_samples"]:
                plt.plot(x_grid, sample_curve, color="#fb7185", alpha=CDF_POSTERIOR_SAMPLE_ALPHA, linewidth=CDF_POSTERIOR_SAMPLE_LINEWIDTH)
        plt.fill_between(x_grid, posterior["cdf_low"], posterior["cdf_high"], color="#fb7185", alpha=CDF_POSTERIOR_BAND_ALPHA, label="BPINN posterior 90% band")
        if PLOT_BPINN_CDF_MEDIAN:
            plt.plot(x_grid, posterior["cdf_med"], color="#be123c", linestyle="-", linewidth=1.8, label="BPINN posterior median CDF")
    if gmm_params.get("status") == "ok" and PLOT_BPINN_CDF_DETERMINISTIC:
        F_gmm = gmm_cdf_np(x_grid, gmm_params["weights"], gmm_params["mu_raw"], gmm_params["sigma_raw"])
        plt.plot(x_grid, F_gmm, color="#f97316", linestyle="--", linewidth=2.0, label="BPINN deterministic GMM CDF")
        if CDF_INCLUDE_QUANTILE_MARKERS:
            qs = np.asarray(base.gmm_quantile([0.05, 0.50, 0.95], gmm_params["weights"], (gmm_params["mu_raw"] - gmm_params["h_mean"]) / (gmm_params["h_std"] + 1e-12), gmm_params["sigma_raw"] / (abs(gmm_params["h_std"]) + 1e-12))).reshape(-1)
            q_raw = gmm_params["h_mean"] + gmm_params["h_std"] * qs
            for q, c, lab in zip(q_raw, ["#f97316", "#dc2626", "#f97316"], ["BPINN q05", "BPINN q50", "BPINN q95"]):
                plt.axvline(q, color=c, linestyle=":" if "q50" not in lab else "-.", linewidth=1.0, alpha=0.8, label=lab)
    elif gmm_params.get("status") != "ok":
        plt.plot([], [], label=f"BPINN {gmm_params.get('status')}")
    if CDF_INCLUDE_QUANTILE_MARKERS and len(h_sorted):
        for q, lab in zip(np.quantile(h_sorted, [0.05, 0.5, 0.95]), ["MC q05", "MC q50", "MC q95"]):
            plt.axvline(q, color="#2563eb", linestyle=":" if "q50" not in lab else "-.", linewidth=0.9, alpha=0.7, label=lab)
    title = f"seed={seed:04d} theta={theta_idx:02d} theta_rad={THETA_VALUES[theta_idx]:.3f}"
    if CDF_INCLUDE_ARMS_IN_TITLE:
        title += f" ARMS={arms:.3f}%"
    if PLOT_BPINN_CDF_POSTERIOR_BAND:
        title += f" posterior_samples={posterior.get('n_samples', 0)}"
    plt.title(title)
    plt.xlabel("h_theta")
    plt.ylabel("CDF")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    base_path = Path(out_dir) / f"selected_seed_{_seed_tag(seed)}_theta{theta_idx:02d}_cdf"
    _save_fig(base_path, seed, f"Theta {theta_idx:02d} MC empirical CDF vs BPINN GMM CDF with posterior band", png_artifact_type="theta_cdf_png", pdf_artifact_type="theta_cdf_pdf")
    plt.close()


def plot_theta_cdf_montage_for_seed(seed, supports, model_map, theta_metrics_df):
    seed_dir = _seed_dir(seed)
    H_MC = np.asarray(supports["H_MC"], dtype=float)
    success_mask = np.asarray(supports["success_mask"], dtype=bool)
    x_mu = np.asarray(supports["x_mu"], dtype=float)
    fig, axes = plt.subplots(CDF_MONTAGE_NROWS, CDF_MONTAGE_NCOLS, figsize=(12.0, 12.0), squeeze=False)
    for ax, theta_idx in zip(axes.ravel(), range(N_THETA)):
        valid = success_mask[:, theta_idx]
        h_mc = H_MC[valid, theta_idx]
        h_sorted, F_emp = empirical_cdf(h_mc)
        gmm = get_bpinn_gmm_raw_for_theta(seed, theta_idx, x_mu, model_map)
        x_grid = _cdf_grid(h_mc, gmm)
        posterior = compute_bpinn_cdf_posterior_band(seed, theta_idx, x_mu, model_map, x_grid) if gmm.get("status") == "ok" else {"status": "missing_model", "n_samples": 0}
        if len(h_sorted):
            ax.step(h_sorted, F_emp, where="post", color="black", linewidth=1.1, label="MC")
        if posterior.get("status") == "ok":
            ax.fill_between(x_grid, posterior["cdf_low"], posterior["cdf_high"], color="#fb7185", alpha=CDF_POSTERIOR_BAND_ALPHA, label="posterior band")
            if PLOT_BPINN_CDF_MEDIAN:
                ax.plot(x_grid, posterior["cdf_med"], color="#be123c", linewidth=1.1, label="posterior median")
        if gmm.get("status") == "ok" and PLOT_BPINN_CDF_DETERMINISTIC:
            ax.plot(x_grid, gmm_cdf_np(x_grid, gmm["weights"], gmm["mu_raw"], gmm["sigma_raw"]), color="#f97316", linestyle="--", linewidth=1.1, label="deterministic")
        arms = _theta_arms(seed, theta_idx, theta_metrics_df)
        ax.set_title(f"theta={theta_idx:02d}, ARMS={arms:.2f}%", fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[N_THETA:]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(f"Seed {seed:04d}: theta CDF montage with BPINN posterior bands", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.98, 0.97])
    _save_fig(seed_dir / f"selected_seed_{_seed_tag(seed)}_theta_cdf_montage", seed, "Montage of theta MC empirical CDFs vs BPINN GMM CDFs with posterior bands", png_artifact_type="theta_cdf_montage_png", pdf_artifact_type="theta_cdf_montage_pdf")
    plt.close(fig)


def _posterior_values_at_points(points, posterior):
    if posterior.get("status") != "ok":
        return {name: np.nan for name in [
            "posterior_q05_at_mc_q05", "posterior_q50_at_mc_q05", "posterior_q95_at_mc_q05",
            "posterior_q05_at_mc_q50", "posterior_q50_at_mc_q50", "posterior_q95_at_mc_q50",
            "posterior_q05_at_mc_q95", "posterior_q50_at_mc_q95", "posterior_q95_at_mc_q95",
        ]}
    x = posterior["x_grid"]
    labels = ["mc_q05", "mc_q50", "mc_q95"]
    out = {}
    for label, pt in zip(labels, points):
        suffix = label.replace("mc_", "")
        out[f"posterior_q05_at_mc_{suffix}"] = float(np.interp(pt, x, posterior["cdf_low"]))
        out[f"posterior_q50_at_mc_{suffix}"] = float(np.interp(pt, x, posterior["cdf_med"]))
        out[f"posterior_q95_at_mc_{suffix}"] = float(np.interp(pt, x, posterior["cdf_high"]))
    return out


def save_theta_cdf_summary(seed, supports, model_map, theta_metrics_df):
    seed_dir = _seed_dir(seed)
    H_MC = np.asarray(supports["H_MC"], dtype=float)
    success_mask = np.asarray(supports["success_mask"], dtype=bool)
    x_mu = np.asarray(supports["x_mu"], dtype=float)
    rows = []
    for theta_idx in range(N_THETA):
        valid = success_mask[:, theta_idx]
        h_mc = H_MC[valid, theta_idx]
        if len(h_mc) < 20:
            print(f"[CDF-warning] seed={seed} theta={theta_idx:02d} has only {len(h_mc)} valid MC samples", flush=True)
        gmm = get_bpinn_gmm_raw_for_theta(seed, theta_idx, x_mu, model_map)
        if gmm.get("status") == "ok":
            q_norm = np.asarray(base.gmm_quantile([0.05, 0.50, 0.95], gmm["weights"], (gmm["mu_raw"] - gmm["h_mean"]) / (gmm["h_std"] + 1e-12), gmm["sigma_raw"] / (abs(gmm["h_std"]) + 1e-12))).reshape(-1)
            bp_q = gmm["h_mean"] + gmm["h_std"] * q_norm
            x_grid = _cdf_grid(h_mc, gmm)
            posterior = compute_bpinn_cdf_posterior_band(seed, theta_idx, x_mu, model_map, x_grid)
        else:
            bp_q = [np.nan, np.nan, np.nan]
            posterior = {"status": gmm.get("status", "missing_model"), "n_samples": 0}
        mc_points = [
            float(np.nanquantile(h_mc, 0.05)) if len(h_mc) else np.nan,
            float(np.nanquantile(h_mc, 0.50)) if len(h_mc) else np.nan,
            float(np.nanquantile(h_mc, 0.95)) if len(h_mc) else np.nan,
        ]
        row = {
            "seed": seed,
            "theta_idx": theta_idx,
            "theta_rad": float(THETA_VALUES[theta_idx]),
            "n_mc_valid": int(len(h_mc)),
            "mc_h_min": float(np.nanmin(h_mc)) if len(h_mc) else np.nan,
            "mc_h_q05": mc_points[0],
            "mc_h_q50": mc_points[1],
            "mc_h_q95": mc_points[2],
            "mc_h_max": float(np.nanmax(h_mc)) if len(h_mc) else np.nan,
            "bpinn_q05": float(bp_q[0]),
            "bpinn_q50": float(bp_q[1]),
            "bpinn_q95": float(bp_q[2]),
            "bpinn_det_q05": float(bp_q[0]),
            "bpinn_det_q50": float(bp_q[1]),
            "bpinn_det_q95": float(bp_q[2]),
            "external_cdf_arms": _theta_arms(seed, theta_idx, theta_metrics_df),
            "model_status": gmm.get("status", "unknown"),
            "posterior_valid_samples": int(posterior.get("n_samples", 0)),
            "posterior_cdf_status": posterior.get("status", "unknown"),
            "posterior_band_mean_width": float(posterior.get("posterior_band_mean_width", np.nan)),
            "posterior_band_max_width": float(posterior.get("posterior_band_max_width", np.nan)),
        }
        row.update(_posterior_values_at_points(mc_points, posterior))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = seed_dir / f"selected_seed_{_seed_tag(seed)}_theta_cdf_summary.csv"
    if SAVE_CDF_POSTERIOR_SUMMARY_CSV:
        _safe_to_csv(df, out, index=False, description="Per-theta CDF summary with posterior bands")
        _record_artifact("csv", seed, out, "Per-theta CDF summary with posterior band diagnostics")
    return df


def plot_theta_cdfs_for_seed(seed, supports, model_map, theta_metrics_df):
    if not PLOT_THETA_CDFS:
        return
    seed_dir = _seed_dir(seed)
    save_theta_cdf_summary(seed, supports, model_map, theta_metrics_df)
    H_MC = np.asarray(supports["H_MC"], dtype=float)
    success_mask = np.asarray(supports["success_mask"], dtype=bool)
    x_mu = np.asarray(supports["x_mu"], dtype=float)
    if SAVE_INDIVIDUAL_THETA_CDF:
        for theta_idx in CDF_PLOT_THETA_LIST:
            theta_idx = int(theta_idx)
            if theta_idx < 0 or theta_idx >= N_THETA:
                print(f"[CDF-warning] skip invalid theta_idx={theta_idx}", flush=True)
                continue
            h_mc = H_MC[success_mask[:, theta_idx], theta_idx]
            gmm = get_bpinn_gmm_raw_for_theta(seed, theta_idx, x_mu, model_map)
            plot_theta_cdf_for_seed(seed, theta_idx, h_mc, gmm, theta_metrics_df, seed_dir, x_mu=x_mu, model_map=model_map)
    if SAVE_THETA_CDF_MONTAGE:
        plot_theta_cdf_montage_for_seed(seed, supports, model_map, theta_metrics_df)

def save_selected_seed_outputs(seed: int, supports: Dict[str, object], bpinn_df: pd.DataFrame, seed_ranking: pd.DataFrame) -> dict:
    seed_dir = _seed_dir(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    theta_values = np.asarray(supports["theta_values"], dtype=float)
    H_MC = np.asarray(supports["H_MC"], dtype=float)
    success_mask = np.asarray(supports["success_mask"], dtype=bool)
    x_mu = np.asarray(supports["x_mu"], dtype=float)
    mc_quantiles = np.nanquantile(np.where(success_mask, H_MC, np.nan), DOMAIN_QUANTILE_TAUS, axis=0).T
    q_cols = ["q05", "q50", "q95"]
    rows = []
    for theta_idx in range(N_THETA):
        brow = bpinn_df[bpinn_df["theta_idx"] == theta_idx]
        bp = {f"bpinn_{c}": np.nan for c in q_cols}
        if not brow.empty:
            for c in q_cols:
                bp[f"bpinn_{c}"] = float(brow.iloc[0].get(f"bpinn_{c}", np.nan))
        row = {
            "seed": seed,
            "theta_idx": theta_idx,
            "theta_rad": float(theta_values[theta_idx]),
            "alpha": float(np.cos(theta_values[theta_idx])),
            "beta": float(np.sin(theta_values[theta_idx])),
            "mc_q05": float(mc_quantiles[theta_idx, 0]),
            "mc_q50": float(mc_quantiles[theta_idx, 1]),
            "mc_q95": float(mc_quantiles[theta_idx, 2]),
            **bp,
        }
        for c in q_cols:
            mc_val = row[f"mc_{c}"]
            bp_val = row[f"bpinn_{c}"]
            row[f"abs_err_{c}"] = float(abs(bp_val - mc_val)) if np.isfinite(bp_val) and np.isfinite(mc_val) else np.nan
            denom = max(abs(mc_val), SUPPORT_EPS) if np.isfinite(mc_val) else np.nan
            row[f"rel_err_{c}_percent"] = float(100.0 * row[f"abs_err_{c}"] / denom) if np.isfinite(denom) else np.nan
        rows.append(row)
    qdf = pd.DataFrame(rows)
    q_csv = seed_dir / f"selected_seed_{_seed_tag(seed)}_theta_quantiles.csv"
    print(f"[DebugSave] q_csv parent exists before save: {q_csv.parent.exists()} | {q_csv.parent}", flush=True)
    _safe_to_csv(qdf, q_csv, index=False, description="Per-theta MC and BPINN support quantiles")
    _record_artifact("csv", seed, q_csv, "Per-theta MC and BPINN support quantiles")

    polys = {"MC": {}, "BPINN": {}}
    area_rows = []
    vertex_rows = []
    for method in ["MC", "BPINN"]:
        for tau, col in zip(DOMAIN_QUANTILE_TAUS, q_cols):
            h_col = f"mc_{col}" if method == "MC" else f"bpinn_{col}"
            poly = support_values_to_polygon(theta_values, qdf[h_col].values)
            polys[method][float(tau)] = poly
            area = polygon_area(poly)
            checks = check_polygon_halfspace_violations(poly, theta_values, qdf[h_col].values)
            if checks["max_halfspace_violation"] > 1e-4:
                print(
                    f"[Polygon-warning] seed={seed} method={method} tau={float(tau):.2f} "
                    f"max_halfspace_violation={checks['max_halfspace_violation']:.6e}",
                    flush=True,
                )
            area_rows.append({"seed": seed, "method": method, "tau": float(tau), "area": area, **checks})
            for vidx, (p0, q0) in enumerate(poly):
                vertex_rows.append({"seed": seed, "method": method, "tau": float(tau), "vertex_idx": vidx, "P0": float(p0), "Q0": float(q0)})
    vertices_csv = seed_dir / f"selected_seed_{_seed_tag(seed)}_polygon_vertices.csv"
    _safe_to_csv(pd.DataFrame(vertex_rows), vertices_csv, index=False, description="Polygon vertices for MC and BPINN quantile domains")
    _record_artifact("csv", seed, vertices_csv, "Polygon vertices for MC and BPINN q05/q50/q95 domains")
    area_csv = seed_dir / f"selected_seed_{_seed_tag(seed)}_polygon_area_summary.csv"
    _safe_to_csv(pd.DataFrame(area_rows), area_csv, index=False, description="Polygon area summary for MC and BPINN quantile domains")
    _record_artifact("csv", seed, area_csv, "Polygon area summary for MC and BPINN quantile domains")

    cloud_polys = []
    valid_rows = np.where(np.all(success_mask, axis=1))[0]
    for idx in valid_rows[:MC_CLOUD_MAX_PLOTS]:
        cloud_polys.append(support_values_to_polygon(theta_values, H_MC[idx, :]))

    metric_row = seed_ranking[seed_ranking["seed"] == seed]
    mean_arms = float(metric_row["mean_arms_percent"].iloc[0]) if not metric_row.empty else np.nan
    max_arms = float(metric_row["max_arms_percent"].iloc[0]) if not metric_row.empty else np.nan
    metrics = {"mc_count": int(H_MC.shape[0]), "mean_arms_percent": mean_arms, "max_arms_percent": max_arms}

    plot_selected_seed_flex_domain_overlay(seed, polys, cloud_polys, metrics)
    plot_mc_vs_bpinn_quantile_domains(seed, polys)
    plot_support_quantiles_by_theta(seed, qdf)
    plot_polygon_area_compare(seed, area_rows)

    area_df = pd.DataFrame(area_rows)
    area_lookup = {(r.method, float(r.tau)): float(r.area) for r in area_df.itertuples(index=False)}
    summary = {
        "seed": seed,
        "mc_count": int(H_MC.shape[0]),
        "valid_mc_count": int(len(valid_rows)),
        "n_theta": N_THETA,
        "mean_arms_percent": mean_arms,
        "max_arms_percent": max_arms,
        "area_mc_q05": area_lookup.get(("MC", 0.05), np.nan),
        "area_mc_q50": area_lookup.get(("MC", 0.50), np.nan),
        "area_mc_q95": area_lookup.get(("MC", 0.95), np.nan),
        "area_bpinn_q05": area_lookup.get(("BPINN", 0.05), np.nan),
        "area_bpinn_q50": area_lookup.get(("BPINN", 0.50), np.nan),
        "area_bpinn_q95": area_lookup.get(("BPINN", 0.95), np.nan),
        "mean_abs_h_err_q50": float(qdf["abs_err_q50"].mean()),
        "mean_rel_h_err_q50_percent": float(qdf["rel_err_q50_percent"].mean()),
        "max_rel_h_err_q50_percent": float(qdf["rel_err_q50_percent"].max()),
        "output_dir": str(seed_dir),
        "status": "ok" if bpinn_df["status"].eq("ok").sum() == N_THETA else "incomplete_model",
    }
    for tau in DOMAIN_QUANTILE_TAUS:
        mc_area = area_lookup.get(("MC", float(tau)), np.nan)
        bp_area = area_lookup.get(("BPINN", float(tau)), np.nan)
        key = f"area_rel_err_q{int(tau*100):02d}_percent"
        summary[key] = float(100.0 * abs(bp_area - mc_area) / max(abs(mc_area), SUPPORT_EPS)) if np.isfinite(mc_area) and np.isfinite(bp_area) else np.nan
    return summary


def main():
    print("[v7.4.6-selected-seed-flex-domain] start", flush=True)
    assert RUN_SELECTED_SEED_FLEX_DOMAIN_PLOTS is True
    print(f"[Path] SCRIPT_DIR = {SCRIPT_DIR}", flush=True)
    print(f"[Path] SELECTED_FLEX_DOMAIN_OUT_DIR = {_abs_path(SELECTED_FLEX_DOMAIN_OUT_DIR)}", flush=True)
    print(f"[Path] COMBINED_EXTERNAL_METRICS_CSV = {_abs_path(COMBINED_EXTERNAL_METRICS_CSV)}", flush=True)
    print(f"[Config] MC_SUPPORT_MODE = {MC_SUPPORT_MODE}", flush=True)
    _ensure_dir(SELECTED_FLEX_DOMAIN_OUT_DIR)

    seed_ranking = load_seed_ranking(COMBINED_EXTERNAL_METRICS_CSV)
    theta_metrics_df = load_combined_theta_metrics(COMBINED_EXTERNAL_METRICS_CSV)
    selected_seeds = select_external_seeds(seed_ranking)

    print("[Setup] building IEEE-33 no-PCC case", flush=True)
    case = base.build_ieee33_case()
    base.ALL_THETA_LIST = list(range(N_THETA))
    base.ALL_THETA_RESULT_DIR = ALL_THETA_RESULT_DIR
    print("[Setup] loading qcal all-theta models", flush=True)
    model_map = base.load_existing_all_theta_models(case)

    summary_rows = []
    for seed in selected_seeds:
        print(f"[Seed] start seed={seed}", flush=True)
        supports = load_or_compute_mc_supports_for_seed(seed, case)
        bpinn_df = predict_bpinn_quantiles_for_seed(seed, case, np.asarray(supports["x_mu"], dtype=float), model_map)
        summary = save_selected_seed_outputs(seed, supports, bpinn_df, seed_ranking)
        plot_theta_cdfs_for_seed(seed, supports, model_map, theta_metrics_df)
        summary_rows.append(summary)
        print(f"[Seed] done seed={seed} status={summary['status']}", flush=True)

    summary_csv = _abs_path(SELECTED_FLEX_DOMAIN_OUT_DIR) / "selected_seed_flex_domain_summary.csv"
    _safe_to_csv(pd.DataFrame(summary_rows), summary_csv, index=False, description="Selected seed flex-domain summary")
    _record_artifact("csv", "all", summary_csv, "Selected seed flex-domain summary")

    manifest_csv = _abs_path(SELECTED_FLEX_DOMAIN_OUT_DIR) / "visualization_manifest_v7_4_6_3_selected_seed_flex_domain.csv"
    _safe_to_csv(pd.DataFrame(_manifest_rows), manifest_csv, index=False, description="Visualization manifest")
    print("[Done] selected seed flex-domain visualization finished.", flush=True)
    print(f"[Done] output_dir = {SELECTED_FLEX_DOMAIN_OUT_DIR}", flush=True)
    print(f"[Done] summary_csv = {summary_csv}", flush=True)
    print(f"[Done] manifest_csv = {manifest_csv}", flush=True)


if __name__ == "__main__":
    main()
