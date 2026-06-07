"""v7.4.5 selected-seed ARMS heatmap visualization.

Visualization-only utility for already-completed qcal all-theta MC2500 external
validation summaries. This script only reads existing CSV results and writes CSV
summaries/figures; it does not train models, rebuild datasets, run OPF, run
MC2500 validation, load model.pt/norm.pkl, or synthesize flexibility domains.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===== visualization-only configuration =====
VIS_ONLY = True

# Default: read the seed=1..20 combined result. If only seed=1..10 exists,
# manually change this to the seed01_10_combined directory.
SELECTED_HEATMAP_INPUT_DIR = (
    "training_results_theta_independent_v7style_no_pcc_branch_limit_"
    "data1000_mc60_qcal_only_comparison_mc2500_seeds01_20_combined"
)
SELECTED_HEATMAP_INPUT_CSV = (
    SELECTED_HEATMAP_INPUT_DIR + "/all_theta_multiseed_external_by_theta_seed.csv"
)
SELECTED_HEATMAP_OUT_DIR = SELECTED_HEATMAP_INPUT_DIR + "/selected_seed_heatmaps"

# Manual external-scenario seeds for display; edit this list directly, e.g.
# MANUAL_SELECTED_SEEDS = [2, 5, 8, 12, 17]
MANUAL_SELECTED_SEEDS = [8, 2, 5]
KEEP_MANUAL_SEED_ORDER = True
SORT_SELECTED_SEEDS_BY = "manual"  # "manual" | "max_arms" | "mean_arms" | "median_arms"
ANNOTATE_HEATMAP = True
HEATMAP_VMAX = 5.0
SAVE_PNG = True
SAVE_PDF = True
DPI = 300
N_THETA_EXPECTED = 12

ARMS_COLUMN_CANDIDATES = ["external_cdf_arms", "cdf_arms", "ARMS", "arms"]


def _out_path(name):
    return str(Path(SELECTED_HEATMAP_OUT_DIR) / name)


def _save_current_figure(base_name):
    paths = []
    if SAVE_PNG:
        path = _out_path(base_name + ".png")
        plt.savefig(path, dpi=DPI)
        paths.append(path)
    if SAVE_PDF:
        path = _out_path(base_name + ".pdf")
        plt.savefig(path)
        paths.append(path)
    for path in paths:
        print(f"[v7.4.5-output] {path}", flush=True)
    return paths


def _resolve_arms_column(df):
    for col in ARMS_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "Missing ARMS column. Expected one of "
        f"{ARMS_COLUMN_CANDIDATES}; CSV columns are {list(df.columns)}"
    )


def load_and_filter_selected_seeds():
    """Load the combined by-theta/seed CSV and keep MANUAL_SELECTED_SEEDS only."""
    input_csv = Path(SELECTED_HEATMAP_INPUT_CSV)
    if not input_csv.exists():
        raise FileNotFoundError(f"Selected heatmap input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = ["theta_idx", "seed"]
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns {missing_required}; CSV columns are {list(df.columns)}"
        )

    arms_col = _resolve_arms_column(df)
    if arms_col != "external_cdf_arms":
        df = df.rename(columns={arms_col: "external_cdf_arms"})

    df["theta_idx"] = df["theta_idx"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["external_cdf_arms"] = df["external_cdf_arms"].astype(float)

    selected = list(dict.fromkeys(int(s) for s in MANUAL_SELECTED_SEEDS))
    available = set(int(s) for s in df["seed"].unique())
    for seed in selected:
        if seed not in available:
            print(f"[v7.4.5-warning] selected seed={seed:03d} not found in CSV", flush=True)

    df_selected = df[df["seed"].isin(selected)].copy()
    if df_selected.empty:
        raise RuntimeError(
            f"No valid selected seeds from MANUAL_SELECTED_SEEDS={MANUAL_SELECTED_SEEDS} "
            f"were found in {SELECTED_HEATMAP_INPUT_CSV}"
        )

    completeness_rows = []
    for seed in selected:
        dd = df_selected[df_selected["seed"] == seed]
        if dd.empty:
            completeness_rows.append({"seed": seed, "n_theta": 0, "missing_theta": "all"})
            continue
        present = set(int(t) for t in dd["theta_idx"].unique())
        missing_theta = [t for t in range(N_THETA_EXPECTED) if t not in present]
        if missing_theta:
            print(
                f"[v7.4.5-warning] seed={seed:03d} has {len(present)} theta; "
                f"missing theta={missing_theta}",
                flush=True,
            )
        completeness_rows.append(
            {"seed": seed, "n_theta": len(present), "missing_theta": ";".join(map(str, missing_theta))}
        )

    out = _out_path("selected_seed_theta_arms.csv")
    df_selected.to_csv(out, index=False)
    pd.DataFrame(completeness_rows).to_csv(_out_path("selected_seed_theta_completeness.csv"), index=False)
    print(f"[v7.4.5-output] {out}", flush=True)
    print(f"[v7.4.5-output] {_out_path('selected_seed_theta_completeness.csv')}", flush=True)
    return df_selected


def _seed_order(df_selected):
    selected_available = [
        int(s) for s in MANUAL_SELECTED_SEEDS if int(s) in set(df_selected["seed"].astype(int).unique())
    ]
    mode = str(SORT_SELECTED_SEEDS_BY).strip().lower()
    if KEEP_MANUAL_SEED_ORDER or mode == "manual":
        return selected_available

    summary = df_selected.groupby("seed")["external_cdf_arms"].agg(
        max_arms="max", mean_arms="mean", median_arms="median"
    )
    if mode not in {"max_arms", "mean_arms", "median_arms"}:
        raise ValueError(
            "SORT_SELECTED_SEEDS_BY must be one of 'manual', 'max_arms', 'mean_arms', 'median_arms'"
        )
    return [int(s) for s in summary.sort_values(mode).index.tolist()]


def plot_selected_seed_arms_heatmap(df_selected):
    seed_order = _seed_order(df_selected)
    pivot = df_selected.pivot_table(
        index="theta_idx", columns="seed", values="external_cdf_arms", aggfunc="mean"
    )
    pivot = pivot.reindex(index=list(range(N_THETA_EXPECTED)), columns=seed_order)

    n_seed = max(1, len(seed_order))
    figsize = (max(5.5, 1.25 * n_seed + 2.5), 6.0)
    plt.figure(figsize=figsize)
    im = plt.imshow(
        pivot.values,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=HEATMAP_VMAX,
        interpolation="nearest",
    )
    cbar = plt.colorbar(im)
    cbar.set_label("ARMS (%)")
    plt.title("Selected external scenarios ARMS heatmap")
    plt.xlabel("External scenario seed")
    plt.ylabel("Theta index")
    plt.xticks(np.arange(len(seed_order)), [str(s) for s in seed_order])
    plt.yticks(np.arange(N_THETA_EXPECTED), [str(t) for t in range(N_THETA_EXPECTED)])

    if ANNOTATE_HEATMAP:
        arr = pivot.values
        threshold = (HEATMAP_VMAX if HEATMAP_VMAX is not None else np.nanmax(arr)) * 0.55
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if np.isfinite(val):
                    color = "white" if val > threshold else "black"
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    paths = _save_current_figure("selected_seed_arms_heatmap")
    plt.close()
    return paths


def plot_single_seed_theta_bars(df_selected):
    paths = []
    for seed in _seed_order(df_selected):
        dd = df_selected[df_selected["seed"] == seed].copy()
        if dd.empty:
            continue
        dd = dd.set_index("theta_idx").reindex(range(N_THETA_EXPECTED)).reset_index()
        plt.figure(figsize=(7.0, 4.2))
        plt.bar(dd["theta_idx"], dd["external_cdf_arms"], color="#2563eb", alpha=0.85)
        plt.axhline(3.0, color="#f59e0b", linestyle="--", linewidth=1.3, label="3%")
        plt.axhline(5.0, color="#dc2626", linestyle="--", linewidth=1.3, label="5%")
        plt.title(f"External scenario seed {seed:02d}: ARMS by theta")
        plt.xlabel("Theta index")
        plt.ylabel("ARMS (%)")
        plt.xticks(range(N_THETA_EXPECTED))
        plt.legend(loc="best")
        plt.tight_layout()
        paths.extend(_save_current_figure(f"seed{seed:02d}_theta_arms_bar"))
        plt.close()
    return paths


def summarize_selected_seeds(df_selected):
    rows = []
    for seed, g in df_selected.groupby("seed"):
        arms = g["external_cdf_arms"].astype(float)
        imax = arms.idxmax()
        rows.append(
            {
                "seed": int(seed),
                "mean_arms": float(arms.mean()),
                "median_arms": float(arms.median()),
                "max_arms": float(arms.max()),
                "min_arms": float(arms.min()),
                "q90_arms": float(arms.quantile(0.90)),
                "q95_arms": float(arms.quantile(0.95)),
                "n_gt_3pct": int((arms > 3.0).sum()),
                "n_gt_4pct": int((arms > 4.0).sum()),
                "n_gt_5pct": int((arms > 5.0).sum()),
                "worst_theta_idx": int(df_selected.loc[imax, "theta_idx"]),
                "worst_theta_arms": float(df_selected.loc[imax, "external_cdf_arms"]),
            }
        )
    out_df = pd.DataFrame(rows).sort_values(["max_arms", "mean_arms"], ascending=[True, True])
    out = _out_path("selected_seed_summary.csv")
    out_df.to_csv(out, index=False)
    print(f"[v7.4.5-output] {out}", flush=True)
    return out_df


def summarize_selected_theta(df_selected):
    rows = []
    for theta_idx, g in df_selected.groupby("theta_idx"):
        arms = g["external_cdf_arms"].astype(float)
        rows.append(
            {
                "theta_idx": int(theta_idx),
                "mean_arms": float(arms.mean()),
                "median_arms": float(arms.median()),
                "max_arms": float(arms.max()),
                "min_arms": float(arms.min()),
                "q90_arms": float(arms.quantile(0.90)),
                "q95_arms": float(arms.quantile(0.95)),
                "n_gt_3pct": int((arms > 3.0).sum()),
                "n_gt_4pct": int((arms > 4.0).sum()),
                "n_gt_5pct": int((arms > 5.0).sum()),
            }
        )
    out_df = pd.DataFrame(rows).sort_values("theta_idx")
    out = _out_path("selected_theta_summary.csv")
    out_df.to_csv(out, index=False)
    print(f"[v7.4.5-output] {out}", flush=True)
    return out_df


def _recommendation_reason(row):
    if int(row["n_gt_3pct"]) == 0:
        return "all theta below 3%"
    if int(row["n_gt_4pct"]) == 0:
        return "all theta below 4%"
    if int(row["n_gt_5pct"]) == 0:
        return "all theta below 5%"
    return "has theta above 5%, not recommended for clean display"


def write_display_recommendation(df_selected, seed_summary):
    rec = seed_summary.copy().sort_values(
        ["n_gt_5pct", "n_gt_4pct", "max_arms", "mean_arms"], ascending=[True, True, True, True]
    )
    rec.insert(0, "recommended_rank", np.arange(1, len(rec) + 1, dtype=int))
    rec["reason"] = rec.apply(_recommendation_reason, axis=1)
    cols = [
        "recommended_rank",
        "seed",
        "reason",
        "mean_arms",
        "max_arms",
        "n_gt_3pct",
        "n_gt_4pct",
        "n_gt_5pct",
    ]
    out = _out_path("selected_seed_display_recommendation.csv")
    rec[cols].to_csv(out, index=False)
    print(f"[v7.4.5-output] {out}", flush=True)
    return rec[cols]


def main_v7_4_5_selected_seed_heatmaps():
    print("[v7.4.5-selected-seed-heatmaps] start", flush=True)
    assert VIS_ONLY is True
    Path(SELECTED_HEATMAP_OUT_DIR).mkdir(parents=True, exist_ok=True)

    df_selected = load_and_filter_selected_seeds()
    seed_summary = summarize_selected_seeds(df_selected)
    summarize_selected_theta(df_selected)
    plot_selected_seed_arms_heatmap(df_selected)
    plot_single_seed_theta_bars(df_selected)
    write_display_recommendation(df_selected, seed_summary)

    print(f"[v7.4.5-output] {_out_path('selected_seed_theta_arms.csv')}", flush=True)
    print(f"[v7.4.5-output] {_out_path('selected_seed_summary.csv')}", flush=True)
    print(f"[v7.4.5-output] {_out_path('selected_theta_summary.csv')}", flush=True)
    print(f"[v7.4.5-output] {_out_path('selected_seed_display_recommendation.csv')}", flush=True)
    print(f"[v7.4.5-output] {_out_path('selected_seed_arms_heatmap.png')}", flush=True)
    print(f"[v7.4.5-output] {_out_path('selected_seed_arms_heatmap.pdf')}", flush=True)
    for seed in _seed_order(df_selected):
        print(f"[v7.4.5-output] {_out_path(f'seed{seed:02d}_theta_arms_bar.png')}", flush=True)
        print(f"[v7.4.5-output] {_out_path(f'seed{seed:02d}_theta_arms_bar.pdf')}", flush=True)
    print("[v7.4.5-selected-seed-heatmaps] done", flush=True)


if __name__ == "__main__":
    main_v7_4_5_selected_seed_heatmaps()
