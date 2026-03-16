#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_results.py

Analysis pipeline for Nighttime Lights SR results, matched to test.py output.

Expected inputs from test.py:
    ./runs_sr/test_metrics_learned.csv
    ./runs_sr/test_metrics_bicubic.csv
    ./runs_sr/test_visuals_learned/
    ./runs_sr/test_visuals_bicubic/

Implements:
1. organize data
2. compute global performance
3. normalize metrics
4. improvement distribution
5. metric correlation analysis
6. radiometric accuracy analysis
7. visualizations
8. spatial case studies
9. metric agreement analysis

Outputs are written under:
    ./runs_sr/analysis/
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
RUNS_DIR = Path("./runs_sr")
LEARNED_CSV = RUNS_DIR / "test_metrics_learned.csv"
BICUBIC_CSV = RUNS_DIR / "test_metrics_bicubic.csv"

LEARNED_VIS_DIR = RUNS_DIR / "test_visuals_learned"
BICUBIC_VIS_DIR = RUNS_DIR / "test_visuals_bicubic"

OUT_DIR = RUNS_DIR / "analysis"
TOP_K = 10

METRICS = [
    "psnr",
    "ssim",
    "rmse",
    "mae",
    "bias",
    "r2",
    "pearson",
]

# Higher is better
HIGHER_BETTER = {"psnr", "ssim", "r2", "pearson"}

# Lower is better
LOWER_BETTER = {"rmse", "mae"}

# Closer to zero is better
ZERO_BETTER = {"bias"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def metric_direction(metric: str) -> str:
    if metric in HIGHER_BETTER:
        return "higher"
    if metric in LOWER_BETTER:
        return "lower"
    if metric in ZERO_BETTER:
        return "zero"
    raise ValueError(f"Unknown direction for metric: {metric}")


def normalize_metric(series: pd.Series, metric: str) -> pd.Series:
    """
    Normalize to [0,1] where larger always means better.
    """
    x = series.astype(float).copy()

    if metric in ZERO_BETTER:
        x = x.abs()
        xmin, xmax = x.min(), x.max()
        if np.isclose(xmax, xmin):
            return pd.Series(np.ones(len(x)), index=x.index)
        return 1.0 - (x - xmin) / (xmax - xmin)

    xmin, xmax = x.min(), x.max()
    if np.isclose(xmax, xmin):
        return pd.Series(np.ones(len(x)), index=x.index)

    x_norm = (x - xmin) / (xmax - xmin)

    if metric in LOWER_BETTER:
        x_norm = 1.0 - x_norm

    return x_norm


def compute_delta(learned: pd.Series, bicubic: pd.Series, metric: str) -> pd.Series:
    """
    Positive delta always means learned SR is better than bicubic.
    """
    if metric in HIGHER_BETTER:
        return learned - bicubic
    if metric in LOWER_BETTER:
        return bicubic - learned
    if metric in ZERO_BETTER:
        return bicubic.abs() - learned.abs()
    raise ValueError(f"Unknown delta rule for metric: {metric}")


def is_improved(learned: pd.Series, bicubic: pd.Series, metric: str) -> pd.Series:
    if metric in HIGHER_BETTER:
        return learned > bicubic
    if metric in LOWER_BETTER:
        return learned < bicubic
    if metric in ZERO_BETTER:
        return learned.abs() < bicubic.abs()
    raise ValueError(f"Unknown improvement rule for metric: {metric}")


def load_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)

    required = {"image_id", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if df["image_id"].duplicated().any():
        dups = df[df["image_id"].duplicated()]["image_id"].tolist()
        raise ValueError(f"{path} contains duplicate image_id values, e.g. {dups[:5]}")

    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def copy_visual(image_id: str, src_dir: Path, dst_dir: Path) -> None:
    src = src_dir / f"{image_id}.png"
    if src.exists():
        ensure_dir(dst_dir)
        shutil.copy2(src, dst_dir / src.name)


# -----------------------------------------------------------------------------
# 1. Organize data
# -----------------------------------------------------------------------------
def organize_data():
    learned_df = load_csv_checked(LEARNED_CSV).sort_values("image_id").reset_index(drop=True)
    bicubic_df = load_csv_checked(BICUBIC_CSV).sort_values("image_id").reset_index(drop=True)

    common_ids = sorted(set(learned_df["image_id"]).intersection(set(bicubic_df["image_id"])))
    if len(common_ids) == 0:
        raise ValueError("No overlapping image_id values between learned and bicubic CSVs.")

    learned_df = learned_df[learned_df["image_id"].isin(common_ids)].sort_values("image_id").reset_index(drop=True)
    bicubic_df = bicubic_df[bicubic_df["image_id"].isin(common_ids)].sort_values("image_id").reset_index(drop=True)

    merged = pd.merge(
        learned_df,
        bicubic_df,
        on="image_id",
        suffixes=("_learned", "_bicubic"),
    )

    learned_long = learned_df.copy()
    learned_long["method"] = "learned"

    bicubic_long = bicubic_df.copy()
    bicubic_long["method"] = "bicubic"

    long_df = pd.concat([learned_long, bicubic_long], ignore_index=True)
    long_df = long_df[["image_id", "method", *METRICS]]

    out = OUT_DIR / "1_organized_data"
    ensure_dir(out)
    save_csv(learned_df, out / "learned_aligned.csv")
    save_csv(bicubic_df, out / "bicubic_aligned.csv")
    save_csv(merged, out / "merged_wide.csv")
    save_csv(long_df, out / "merged_long.csv")

    return learned_df, bicubic_df, merged, long_df


# -----------------------------------------------------------------------------
# 2. Compute global performance
# -----------------------------------------------------------------------------
def compute_global_performance(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in long_df.groupby("method"):
        for metric in METRICS:
            vals = sub[metric].astype(float)
            rows.append({
                "method": method,
                "metric": metric,
                "direction": metric_direction(metric),
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(ddof=1),
                "min": vals.min(),
                "max": vals.max(),
                "n": len(vals),
            })

    global_df = pd.DataFrame(rows)
    save_csv(global_df, OUT_DIR / "2_global_performance" / "global_performance.csv")
    return global_df


# -----------------------------------------------------------------------------
# 3. Normalize metrics
# -----------------------------------------------------------------------------
def compute_normalized_metrics(long_df: pd.DataFrame):
    norm_df = long_df.copy()

    for metric in METRICS:
        norm_df[f"{metric}_norm"] = normalize_metric(norm_df[metric], metric)

    summary_rows = []
    for method, sub in norm_df.groupby("method"):
        row = {"method": method}
        for metric in METRICS:
            row[f"{metric}_norm_mean"] = sub[f"{metric}_norm"].mean()
            row[f"{metric}_norm_median"] = sub[f"{metric}_norm"].median()
        summary_rows.append(row)

    norm_summary = pd.DataFrame(summary_rows)

    out = OUT_DIR / "3_normalized_metrics"
    save_csv(norm_df, out / "normalized_per_image.csv")
    save_csv(norm_summary, out / "normalized_summary.csv")

    return norm_df, norm_summary


# -----------------------------------------------------------------------------
# 4. Improvement distribution
# -----------------------------------------------------------------------------
def compute_improvement_distribution(merged: pd.DataFrame):
    imp_df = pd.DataFrame({"image_id": merged["image_id"]})

    summary_rows = []
    for metric in METRICS:
        learned_col = f"{metric}_learned"
        bicubic_col = f"{metric}_bicubic"

        delta = compute_delta(merged[learned_col], merged[bicubic_col], metric)
        improved = is_improved(merged[learned_col], merged[bicubic_col], metric)

        imp_df[f"{metric}_delta"] = delta
        imp_df[f"{metric}_improved"] = improved.astype(int)

        summary_rows.append({
            "metric": metric,
            "mean_delta": delta.mean(),
            "median_delta": delta.median(),
            "std_delta": delta.std(ddof=1),
            "min_delta": delta.min(),
            "max_delta": delta.max(),
            "improved_count": int(improved.sum()),
            "improved_percent": float(improved.mean() * 100.0),
            "n": len(delta),
        })

    imp_summary = pd.DataFrame(summary_rows)

    out = OUT_DIR / "4_improvement_distribution"
    save_csv(imp_df, out / "per_image_improvements.csv")
    save_csv(imp_summary, out / "improvement_summary.csv")

    return imp_df, imp_summary


# -----------------------------------------------------------------------------
# 5. Metric correlation analysis
# -----------------------------------------------------------------------------
def compute_metric_correlations(learned_df: pd.DataFrame, bicubic_df: pd.DataFrame, imp_df: pd.DataFrame):
    out = OUT_DIR / "5_metric_correlation_analysis"
    ensure_dir(out)

    learned_pearson = learned_df[METRICS].corr(method="pearson")
    learned_spearman = learned_df[METRICS].corr(method="spearman")

    bicubic_pearson = bicubic_df[METRICS].corr(method="pearson")
    bicubic_spearman = bicubic_df[METRICS].corr(method="spearman")

    delta_cols = [f"{m}_delta" for m in METRICS]
    delta_pearson = imp_df[delta_cols].corr(method="pearson")
    delta_spearman = imp_df[delta_cols].corr(method="spearman")
    delta_pearson.index = METRICS
    delta_pearson.columns = METRICS
    delta_spearman.index = METRICS
    delta_spearman.columns = METRICS

    save_csv(learned_pearson.reset_index().rename(columns={"index": "metric"}), out / "learned_corr_pearson.csv")
    save_csv(learned_spearman.reset_index().rename(columns={"index": "metric"}), out / "learned_corr_spearman.csv")
    save_csv(bicubic_pearson.reset_index().rename(columns={"index": "metric"}), out / "bicubic_corr_pearson.csv")
    save_csv(bicubic_spearman.reset_index().rename(columns={"index": "metric"}), out / "bicubic_corr_spearman.csv")
    save_csv(delta_pearson.reset_index().rename(columns={"index": "metric"}), out / "delta_corr_pearson.csv")
    save_csv(delta_spearman.reset_index().rename(columns={"index": "metric"}), out / "delta_corr_spearman.csv")

    return learned_pearson, learned_spearman, bicubic_pearson, bicubic_spearman, delta_pearson, delta_spearman


# -----------------------------------------------------------------------------
# 6. Radiometric accuracy analysis
# -----------------------------------------------------------------------------
def compute_radiometric_accuracy(long_df: pd.DataFrame):
    radiometric_metrics = ["rmse", "mae", "bias", "r2", "pearson"]

    rows = []
    for method, sub in long_df.groupby("method"):
        row = {"method": method}
        for metric in radiometric_metrics:
            vals = sub[metric].astype(float)
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_median"] = vals.median()
            row[f"{metric}_std"] = vals.std(ddof=1)
        rows.append(row)

    radiometric_df = pd.DataFrame(rows)
    save_csv(radiometric_df, OUT_DIR / "6_radiometric_accuracy_analysis" / "radiometric_summary.csv")
    return radiometric_df


# -----------------------------------------------------------------------------
# 7. Visualizations
# -----------------------------------------------------------------------------
def save_heatmap(df: pd.DataFrame, title: str, out_path: Path):
    ensure_dir(out_path.parent)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(df.values, aspect="auto")

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title(title)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def make_visualizations(global_df, norm_summary, imp_df,
                        learned_pearson, learned_spearman,
                        bicubic_pearson, delta_pearson):
    out = OUT_DIR / "7_visualizations"
    ensure_dir(out)

    # Bar plots of means
    bar_dir = out / "global_mean_bars"
    ensure_dir(bar_dir)
    for metric in METRICS:
        sub = global_df[global_df["metric"] == metric]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(sub["method"], sub["mean"])
        ax.set_title(f"Mean {metric}")
        ax.set_ylabel(metric)
        plt.tight_layout()
        plt.savefig(bar_dir / f"mean_{metric}.png", dpi=200)
        plt.close(fig)

    # Boxplots
    box_dir = out / "boxplots"
    ensure_dir(box_dir)
    for metric in METRICS:
        learned_vals = pd.read_csv(OUT_DIR / "1_organized_data" / "learned_aligned.csv")[metric].values
        bicubic_vals = pd.read_csv(OUT_DIR / "1_organized_data" / "bicubic_aligned.csv")[metric].values

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot([bicubic_vals, learned_vals], labels=["bicubic", "learned"])
        ax.set_title(f"{metric} distribution")
        ax.set_ylabel(metric)
        plt.tight_layout()
        plt.savefig(box_dir / f"boxplot_{metric}.png", dpi=200)
        plt.close(fig)

    # Improvement histograms
    hist_dir = out / "improvement_histograms"
    ensure_dir(hist_dir)
    for metric in METRICS:
        delta_col = f"{metric}_delta"

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(imp_df[delta_col].values, bins=30)
        ax.axvline(0.0, linestyle="--")
        ax.set_title(f"Improvement distribution: {metric}")
        ax.set_xlabel("positive = learned better")
        ax.set_ylabel("count")
        plt.tight_layout()
        plt.savefig(hist_dir / f"hist_{metric}.png", dpi=200)
        plt.close(fig)

    # Normalized mean comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(METRICS))
    width = 0.35

    bicubic_row = norm_summary[norm_summary["method"] == "bicubic"].iloc[0]
    learned_row = norm_summary[norm_summary["method"] == "learned"].iloc[0]

    bicubic_vals = [bicubic_row[f"{m}_norm_mean"] for m in METRICS]
    learned_vals = [learned_row[f"{m}_norm_mean"] for m in METRICS]

    ax.bar(x - width / 2, bicubic_vals, width, label="bicubic")
    ax.bar(x + width / 2, learned_vals, width, label="learned")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, rotation=45, ha="right")
    ax.set_ylabel("normalized score")
    ax.set_title("Normalized mean metric comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "normalized_mean_metric_comparison.png", dpi=200)
    plt.close(fig)

    # Heatmaps
    save_heatmap(learned_pearson, "Learned metric correlation (Pearson)", out / "heatmap_learned_corr_pearson.png")
    save_heatmap(learned_spearman, "Learned metric correlation (Spearman)", out / "heatmap_learned_corr_spearman.png")
    save_heatmap(bicubic_pearson, "Bicubic metric correlation (Pearson)", out / "heatmap_bicubic_corr_pearson.png")
    save_heatmap(delta_pearson, "Improvement correlation (Pearson)", out / "heatmap_delta_corr_pearson.png")


# -----------------------------------------------------------------------------
# 8. Spatial case studies
# -----------------------------------------------------------------------------
def save_spatial_case_studies(merged: pd.DataFrame):
    out = OUT_DIR / "8_spatial_case_studies"
    ensure_dir(out)

    for metric in METRICS:
        learned_col = f"{metric}_learned"
        bicubic_col = f"{metric}_bicubic"

        case_df = merged[["image_id", learned_col, bicubic_col]].copy()
        case_df = case_df.rename(columns={
            learned_col: "learned_value",
            bicubic_col: "bicubic_value",
        })
        case_df["delta_vs_bicubic"] = compute_delta(
            merged[learned_col], merged[bicubic_col], metric
        )

        direction = metric_direction(metric)

        if direction == "higher":
            top_model = case_df.sort_values("learned_value", ascending=False).head(TOP_K)
            worst_model = case_df.sort_values("learned_value", ascending=True).head(TOP_K)
        elif direction == "lower":
            top_model = case_df.sort_values("learned_value", ascending=True).head(TOP_K)
            worst_model = case_df.sort_values("learned_value", ascending=False).head(TOP_K)
        else:  # zero
            case_df["abs_learned"] = case_df["learned_value"].abs()
            top_model = case_df.sort_values("abs_learned", ascending=True).head(TOP_K)
            worst_model = case_df.sort_values("abs_learned", ascending=False).head(TOP_K)

        best_improvement = case_df.sort_values("delta_vs_bicubic", ascending=False).head(TOP_K)
        worst_improvement = case_df.sort_values("delta_vs_bicubic", ascending=True).head(TOP_K)

        metric_dir = out / metric
        ensure_dir(metric_dir)

        save_csv(top_model, metric_dir / f"top_{TOP_K}_learned_{metric}.csv")
        save_csv(worst_model, metric_dir / f"worst_{TOP_K}_learned_{metric}.csv")
        save_csv(best_improvement, metric_dir / f"top_{TOP_K}_improvement_{metric}.csv")
        save_csv(worst_improvement, metric_dir / f"worst_{TOP_K}_improvement_{metric}.csv")

        # Copy corresponding visuals
        for label, df_cases in {
            "top_learned": top_model,
            "worst_learned": worst_model,
            "best_improvement": best_improvement,
            "worst_improvement": worst_improvement,
        }.items():
            for image_id in df_cases["image_id"]:
                copy_visual(image_id, LEARNED_VIS_DIR, metric_dir / "images" / label / "learned")
                copy_visual(image_id, BICUBIC_VIS_DIR, metric_dir / "images" / label / "bicubic")


# -----------------------------------------------------------------------------
# 9. Metric agreement analysis
# -----------------------------------------------------------------------------
def compute_metric_agreement(merged: pd.DataFrame):
    out = OUT_DIR / "9_metric_agreement_analysis"
    ensure_dir(out)

    rank_df = pd.DataFrame({"image_id": merged["image_id"]})

    for metric in METRICS:
        col = f"{metric}_learned"
        direction = metric_direction(metric)

        if direction == "higher":
            rank_df[f"{metric}_rank"] = merged[col].rank(ascending=False, method="average")
        elif direction == "lower":
            rank_df[f"{metric}_rank"] = merged[col].rank(ascending=True, method="average")
        else:
            rank_df[f"{metric}_rank"] = merged[col].abs().rank(ascending=True, method="average")

    rank_cols = [f"{m}_rank" for m in METRICS]
    rank_corr = rank_df[rank_cols].corr(method="spearman")
    rank_corr.index = METRICS
    rank_corr.columns = METRICS

    def top_k_ids(metric: str):
        col = f"{metric}_learned"
        direction = metric_direction(metric)

        if direction == "higher":
            return set(merged.sort_values(col, ascending=False)["image_id"].head(TOP_K))
        if direction == "lower":
            return set(merged.sort_values(col, ascending=True)["image_id"].head(TOP_K))
        return set(
            merged.assign(abs_metric=merged[col].abs())
                  .sort_values("abs_metric", ascending=True)["image_id"]
                  .head(TOP_K)
        )

    rows = []
    top_sets = {m: top_k_ids(m) for m in METRICS}
    for m1 in METRICS:
        row = {"metric": m1}
        for m2 in METRICS:
            inter = len(top_sets[m1].intersection(top_sets[m2]))
            union = len(top_sets[m1].union(top_sets[m2]))
            row[m2] = inter / union if union > 0 else np.nan
        rows.append(row)

    topk_overlap = pd.DataFrame(rows).set_index("metric")

    save_csv(rank_df, out / "metric_ranks.csv")
    save_csv(rank_corr.reset_index().rename(columns={"index": "metric"}), out / "rank_correlation_spearman.csv")
    save_csv(topk_overlap.reset_index().rename(columns={"index": "metric"}), out / f"top_{TOP_K}_overlap_jaccard.csv")

    # Also save as heatmaps
    save_heatmap(rank_corr, "Metric agreement by rank correlation", OUT_DIR / "7_visualizations" / "heatmap_metric_rank_agreement.png")
    save_heatmap(topk_overlap, f"Top-{TOP_K} overlap (Jaccard)", OUT_DIR / "7_visualizations" / f"heatmap_top_{TOP_K}_overlap.png")


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def write_summary():
    summary = [
        "Analysis completed.",
        "",
        f"Input learned CSV:  {LEARNED_CSV}",
        f"Input bicubic CSV:  {BICUBIC_CSV}",
        f"Learned visuals:    {LEARNED_VIS_DIR}",
        f"Bicubic visuals:    {BICUBIC_VIS_DIR}",
        "",
        f"Outputs saved to:   {OUT_DIR}",
        "",
        "Subfolders:",
        "1_organized_data",
        "2_global_performance",
        "3_normalized_metrics",
        "4_improvement_distribution",
        "5_metric_correlation_analysis",
        "6_radiometric_accuracy_analysis",
        "7_visualizations",
        "8_spatial_case_studies",
        "9_metric_agreement_analysis",
    ]

    ensure_dir(OUT_DIR)
    with open(OUT_DIR / "analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("\n".join(summary))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    learned_df, bicubic_df, merged, long_df = organize_data()
    global_df = compute_global_performance(long_df)
    norm_df, norm_summary = compute_normalized_metrics(long_df)
    imp_df, imp_summary = compute_improvement_distribution(merged)

    learned_pearson, learned_spearman, bicubic_pearson, bicubic_spearman, delta_pearson, delta_spearman = \
        compute_metric_correlations(learned_df, bicubic_df, imp_df)

    radiometric_df = compute_radiometric_accuracy(long_df)

    make_visualizations(
        global_df=global_df,
        norm_summary=norm_summary,
        imp_df=imp_df,
        learned_pearson=learned_pearson,
        learned_spearman=learned_spearman,
        bicubic_pearson=bicubic_pearson,
        delta_pearson=delta_pearson,
    )

    save_spatial_case_studies(merged)
    compute_metric_agreement(merged)
    write_summary()


if __name__ == "__main__":
    main()