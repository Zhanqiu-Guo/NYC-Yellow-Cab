#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = {
    "d": "Demand",
    "r": "Revenue",
}

METRICS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "mape": "MAPE",
    "wape": "WAPE",
}

MODEL_LABELS = {
    "xgb_base": "XGBoost",
    "hybrid": "Hybrid Model",
}

GROUP_COLORS = {
    "flights": "#4C78A8",
    "demand": "#F58518",
    "revenue": "#54A24B",
    "weather": "#B279A2",
}


def metric_col(target, metric, suffix):
    return f"{target}_{metric}_{suffix}"


def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    expected = {"model", "group", "noise_std"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    return df.sort_values(["model", "group", "noise_std"]).reset_index(drop=True)


def savefig(out_dir, name):
    path = os.path.join(out_dir, name)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_delta_curves(df, out_dir):
    groups = [g for g in GROUP_COLORS if g in set(df["group"])]
    model_styles = {
        "xgb_base": ("--", "o"),
        "hybrid": ("-", "s"),
    }

    for target, target_name in TARGETS.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
        axes = axes.ravel()
        fig.suptitle(f"{target_name} Noise Sensitivity: Metric Δ vs Noise Level", fontsize=15)

        for ax, (metric, metric_name) in zip(axes, METRICS.items()):
            y_col = metric_col(target, metric, "delta")
            for model, (linestyle, marker) in model_styles.items():
                for group in groups:
                    sub = df[(df["model"] == model) & (df["group"] == group)]
                    if sub.empty:
                        continue
                    label = f"{MODEL_LABELS.get(model, model)} | {group}"
                    ax.plot(
                        sub["noise_std"],
                        sub[y_col],
                        color=GROUP_COLORS[group],
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=1.8,
                        markersize=4,
                        label=label,
                    )

            ax.axhline(0.0, color="#555555", linewidth=0.8)
            ax.set_title(f"Δ {metric_name}")
            ax.set_xlabel("Noise std multiplier")
            ax.set_ylabel(f"Δ {metric_name}")
            ax.grid(True, alpha=0.25)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        savefig(out_dir, f"noise_delta_curves_{target}.png")


def symmetric_limit(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    lim = np.nanmax(np.abs(values))
    return lim if lim > 0 else 1.0


def annotate_heatmap(ax, data, fmt=".2f"):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)


def plot_max_noise_heatmaps(df, out_dir):
    max_noise = df["noise_std"].max()
    sub = df[df["noise_std"] == max_noise]
    groups = [g for g in GROUP_COLORS if g in set(sub["group"])]
    metrics = list(METRICS.keys())

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.045],
        wspace=0.18,
        hspace=0.30,
    )
    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ])
    cax = fig.add_subplot(gs[:, 2])
    fig.suptitle(f"Metric Δ at Max Noise Level ({max_noise:g})", fontsize=15)

    panels = [
        ("xgb_base", "d"),
        ("hybrid", "d"),
        ("xgb_base", "r"),
        ("hybrid", "r"),
    ]

    all_values = []
    panel_data = []
    for model, target in panels:
        data = np.full((len(groups), len(metrics)), np.nan)
        panel = sub[sub["model"] == model]
        for i, group in enumerate(groups):
            row = panel[panel["group"] == group]
            if row.empty:
                continue
            row = row.iloc[0]
            for j, metric in enumerate(metrics):
                data[i, j] = row[metric_col(target, metric, "delta")]
        panel_data.append(data)
        all_values.extend(data.ravel())

    vmax = symmetric_limit(all_values)
    vmin = -vmax

    for ax, (model, target), data in zip(axes.ravel(), panels, panel_data):
        im = ax.imshow(data, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{MODEL_LABELS.get(model, model)} | {TARGETS[target]}")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([METRICS[m] for m in metrics])
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups)
        annotate_heatmap(ax, data)

    fig.colorbar(im, cax=cax, label="Metric Δ vs clean")
    fig.subplots_adjust(top=0.90)
    savefig(out_dir, "noise_max_level_delta_heatmaps.png")


def plot_hybrid_minus_base(df, out_dir):
    max_noise = df["noise_std"].max()
    sub = df[df["noise_std"] == max_noise]
    groups = [g for g in GROUP_COLORS if g in set(sub["group"])]
    metric_keys = [(t, m) for t in TARGETS for m in METRICS]
    col_labels = [f"{TARGETS[t]}\n{METRICS[m]}" for t, m in metric_keys]

    data = np.full((len(groups), len(metric_keys)), np.nan)
    for i, group in enumerate(groups):
        base = sub[(sub["model"] == "xgb_base") & (sub["group"] == group)]
        hybrid = sub[(sub["model"] == "hybrid") & (sub["group"] == group)]
        if base.empty or hybrid.empty:
            continue
        base = base.iloc[0]
        hybrid = hybrid.iloc[0]
        for j, (target, metric) in enumerate(metric_keys):
            key = metric_col(target, metric, "delta")
            data[i, j] = hybrid[key] - base[key]

    vmax = symmetric_limit(data.ravel())
    fig, ax = plt.subplots(figsize=(14, 4.8))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(
        f"Hybrid Model Δ minus XGBoost Δ at Max Noise ({max_noise:g})\n"
        "Negative means Hybrid Model degraded less than XGBoost"
    )
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    annotate_heatmap(ax, data)
    fig.colorbar(im, ax=ax, shrink=0.9, label="Hybrid Model Δ - XGBoost Δ")
    fig.tight_layout()
    savefig(out_dir, "noise_hybrid_minus_base_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="Plot noise robustness results.")
    parser.add_argument(
        "--csv",
        default=os.path.join("dataset", "noise_robustness_h1_fixed.csv"),
        help="Path to noise_robustness_h1_fixed.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("eval_outputs", "noise_robustness"),
        help="Directory for output figures",
    )
    args = parser.parse_args()

    ensure_out_dir(args.out_dir)
    df = load_results(args.csv)

    plot_delta_curves(df, args.out_dir)
    plot_max_noise_heatmaps(df, args.out_dir)
    plot_hybrid_minus_base(df, args.out_dir)


if __name__ == "__main__":
    main()
