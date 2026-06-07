"""v7.4.6 selected external-seed P0-Q0 probability flex-domain plots.

Visualization/evaluation utility built on the existing v7.4.4 qcal all-theta
MC2500 workflow and v7.4.5 selected-seed visualization idea.

Windows PowerShell:
    py Po_flex_domain_v7_4_6_no_pcc_qcal_selected_seed_flex_domain_plots.py

Usage notes:
- Edit SELECTED_EXTERNAL_SEEDS to choose the external seed(s) to plot, e.g.
  SELECTED_EXTERNAL_SEEDS = [8, 2, 19, 20, 13].
- Edit MC_DOMAIN_QUANTILE_N to control the MC support sample count.
- The first run for a seed may need MC-OPF support computation. If a support
  cache exists and RECOMPUTE_MC_SUPPORTS=False, the cache is used directly.
- If no cache exists, the script prints a clear warning and recomputes MC-OPF
  supports for only the selected seed(s); set RECOMPUTE_MC_SUPPORTS=True to
  force recomputation even when a cache exists.

This script does not train models, rebuild the training dataset, modify model
artifacts, or run flex synthesis. It writes CSV logs, npz support caches, plots,
and a visualization manifest for selected external seeds.
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

SELECTED_FLEX_DOMAIN_OUT_DIR = (
    "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_"
    "qcal_only_comparison_mc2500_seeds01_20_combined/selected_seed_flex_domain_plots"
)
COMBINED_EXTERNAL_METRICS_CSV = (
    "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_"
    "qcal_only_comparison_mc2500_seeds01_20_combined/all_theta_multiseed_external_by_theta_seed.csv"
)

ALL_THETA_RESULT_DIR = (
    "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_"
    "qcal_only_alltheta"
)
N_THETA = 12
THETA_VALUES = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
SUPPORT_EPS = 1e-9
POLYGON_COORD_MAX = 1e4


_manifest_rows: List[Dict[str, object]] = []


def _seed_tag(seed: int) -> str:
    return f"{int(seed):04d}"


def _seed_dir(seed: int) -> Path:
    return Path(SELECTED_FLEX_DOMAIN_OUT_DIR) / f"seed_{_seed_tag(seed)}"


def _record_artifact(artifact_type: str, seed: int | str, path: str | Path, description: str) -> None:
    path = Path(path)
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


def _save_fig(base_path_without_ext: Path, seed: int, description: str) -> None:
    if SAVE_PNG:
        png = base_path_without_ext.with_suffix(".png")
        plt.savefig(png, dpi=DPI)
        _record_artifact("figure_png", seed, png, description)
    if SAVE_PDF:
        pdf = base_path_without_ext.with_suffix(".pdf")
        plt.savefig(pdf)
        _record_artifact("figure_pdf", seed, pdf, description)


def _closed(poly: np.ndarray) -> np.ndarray:
    if poly is None or len(poly) == 0:
        return np.empty((0, 2), dtype=float)
    if np.allclose(poly[0], poly[-1]):
        return poly
    return np.vstack([poly, poly[0]])


def load_seed_ranking(metrics_csv: str | Path = COMBINED_EXTERNAL_METRICS_CSV) -> pd.DataFrame:
    """Load combined MC2500 metrics and write per-seed ranking."""
    metrics_csv = Path(metrics_csv)
    if not metrics_csv.exists():
        candidates = [
            COMBINED_EXTERNAL_METRICS_CSV,
            "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_qcal_only_comparison_mc2500_seeds01_10_combined/all_theta_multiseed_external_by_theta_seed.csv",
            "training_results_theta_independent_v7style_no_pcc_branch_limit_data1000_mc60_qcal_only_comparison_mc2500/all_theta_multiseed_external_by_theta_seed.csv",
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
    out = Path(SELECTED_FLEX_DOMAIN_OUT_DIR) / "selected_seed_flex_domain_seed_ranking.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out, index=False)
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
        np.savez_compressed(
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
        pd.DataFrame(long_rows).to_csv(long_csv, index=False)
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
        _fill_poly(polys["MC"].get(0.95), color="#60a5fa", alpha=0.14, label="MC q05-q95 band")
        _plot_poly(polys["MC"].get(0.05), color="#3b82f6", linewidth=1.0, alpha=0.55, linestyle=":")
        _plot_poly(polys["MC"].get(0.95), color="#3b82f6", linewidth=1.0, alpha=0.55, linestyle=":")
    if PLOT_BPINN_QUANTILE_BAND:
        _fill_poly(polys["BPINN"].get(0.95), color="#fb923c", alpha=0.16, label="BPINN q05-q95 band")
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
    _save_fig(base_path, seed, "Selected seed flex-domain overlay with MC cloud and BPINN/MC quantile domains")
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
    qdf.to_csv(q_csv, index=False)
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
            area_rows.append({"seed": seed, "method": method, "tau": float(tau), "area": area})
            for vidx, (p0, q0) in enumerate(poly):
                vertex_rows.append({"seed": seed, "method": method, "tau": float(tau), "vertex_idx": vidx, "P0": float(p0), "Q0": float(q0)})
    vertices_csv = seed_dir / f"selected_seed_{_seed_tag(seed)}_polygon_vertices.csv"
    pd.DataFrame(vertex_rows).to_csv(vertices_csv, index=False)
    _record_artifact("csv", seed, vertices_csv, "Polygon vertices for MC and BPINN q05/q50/q95 domains")
    area_csv = seed_dir / f"selected_seed_{_seed_tag(seed)}_polygon_area_summary.csv"
    pd.DataFrame(area_rows).to_csv(area_csv, index=False)
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
    Path(SELECTED_FLEX_DOMAIN_OUT_DIR).mkdir(parents=True, exist_ok=True)

    seed_ranking = load_seed_ranking(COMBINED_EXTERNAL_METRICS_CSV)
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
        summary_rows.append(summary)
        print(f"[Seed] done seed={seed} status={summary['status']}", flush=True)

    summary_csv = Path(SELECTED_FLEX_DOMAIN_OUT_DIR) / "selected_seed_flex_domain_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    _record_artifact("csv", "all", summary_csv, "Selected seed flex-domain summary")

    manifest_csv = Path(SELECTED_FLEX_DOMAIN_OUT_DIR) / "visualization_manifest_v7_4_6_selected_seed_flex_domain.csv"
    pd.DataFrame(_manifest_rows).to_csv(manifest_csv, index=False)
    print("[Done] selected seed flex-domain visualization finished.", flush=True)
    print(f"[Done] output_dir = {SELECTED_FLEX_DOMAIN_OUT_DIR}", flush=True)
    print(f"[Done] summary_csv = {summary_csv}", flush=True)
    print(f"[Done] manifest_csv = {manifest_csv}", flush=True)


if __name__ == "__main__":
    main()
