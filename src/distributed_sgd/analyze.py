from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _clean_summary(summary: pd.DataFrame):
    out = summary.copy()
    for col in [
        "mean_iteration_time_s",
        "p95_iteration_time_s",
        "throughput_samples_s",
        "comm_fraction",
        "straggler_impact_rate",
        "final_val_loss",
        "time_to_target_loss_s",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def plot_scalability(summary: pd.DataFrame, out: Path):
    data = summary[summary["workload"].isin(["homogeneous_high_bw", "dynamic_mixed"])]
    if data.empty:
        data = summary
    grouped = data.groupby(["workload", "strategy", "workers"], as_index=False)["throughput_samples_s"].mean()
    for workload, frame in grouped.groupby("workload"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy, sf in frame.groupby("strategy"):
            ax.plot(sf["workers"], sf["throughput_samples_s"], marker="o", label=strategy)
        ax.set_title(f"Scalability: {workload}")
        ax.set_xlabel("Worker processes")
        ax.set_ylabel("Throughput (samples/sec)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, out / f"scalability_{workload}.png")


def plot_baseline_comparison(summary: pd.DataFrame, out: Path):
    data = summary[summary["status"] != "failed"].copy()
    if data.empty:
        return
    max_workers = data["workers"].max()
    data = data[data["workers"] == max_workers]
    grouped = data.groupby(["workload", "strategy"], as_index=False)["mean_iteration_time_s"].mean()
    workloads = list(grouped["workload"].unique())
    strategies = list(grouped["strategy"].unique())
    x = np.arange(len(workloads))
    width = 0.8 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, strategy in enumerate(strategies):
        vals = []
        for workload in workloads:
            s = grouped[(grouped["workload"] == workload) & (grouped["strategy"] == strategy)]["mean_iteration_time_s"]
            vals.append(float(s.iloc[0]) if not s.empty else np.nan)
        ax.bar(x + (i - len(strategies) / 2) * width + width / 2, vals, width, label=strategy)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=25, ha="right")
    ax.set_ylabel("Mean iteration time (s)")
    ax.set_title(f"Baseline comparison at {max_workers} workers")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out / "baseline_comparison.png")


def plot_adaptive_timeline(steps: pd.DataFrame, out: Path):
    data = steps[(steps["strategy"] == "adaptive") & (steps["workload"] == "dynamic_mixed")].copy()
    if data.empty:
        data = steps[steps["strategy"] == "adaptive"].copy()
    if data.empty:
        return
    data["iteration_time_s"] = pd.to_numeric(data["iteration_time_s"], errors="coerce")
    data["worker_cv"] = pd.to_numeric(data["worker_cv"], errors="coerce")
    sample = data[(data["workers"] == data["workers"].max()) & (data["repeat"] == data["repeat"].min())]
    if sample.empty:
        return
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(sample["step"], sample["iteration_time_s"], color="#1f77b4", label="iteration time")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Iteration time (s)", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(sample["step"], sample["worker_cv"], color="#d62728", label="worker CV")
    ax2.set_ylabel("Worker-time coefficient of variation", color="#d62728")
    switches = sample[sample["mode_switched"] == 1]
    for _, row in switches.iterrows():
        ax1.axvline(row["step"], color="black", linestyle="--", alpha=0.5)
    ax1.set_title("Adaptive mode timeline")
    _save(fig, out / "adaptive_mode_timeline.png")


def plot_sensitivity(summary: pd.DataFrame, out: Path):
    data = summary[summary["strategy"] == "adaptive"].copy()
    if data.empty:
        return
    grouped = data.groupby(["workload", "workers"], as_index=False)["mean_iteration_time_s"].mean()
    pivot = grouped.pivot(index="workload", columns="workers", values="mean_iteration_time_s")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("Workers")
    ax.set_title("Adaptive sensitivity surface by workload/scale")
    fig.colorbar(im, ax=ax, label="Mean iteration time (s)")
    _save(fig, out / "sensitivity_heatmap.png")


def plot_failure(steps: pd.DataFrame, out: Path):
    data = steps[steps["workload"] == "worker_crash"].copy()
    if data.empty:
        return
    data["iteration_time_s"] = pd.to_numeric(data["iteration_time_s"], errors="coerce")
    sample = data[(data["workers"] == data["workers"].max()) & (data["repeat"] == data["repeat"].min())]
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, sf in sample.groupby("strategy"):
        ax.plot(sf["step"], sf["iteration_time_s"], marker="o", markersize=2, label=strategy)
    ax.set_xlabel("Step")
    ax.set_ylabel("Iteration time (s)")
    ax.set_title("Failure scenario: worker crash")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out / "failure_recovery_timeline.png")


def write_improvement_table(summary: pd.DataFrame, out: Path):
    rows = []
    grouped = summary[summary["status"] != "failed"].groupby(["workload", "workers"])
    for (workload, workers), frame in grouped:
        adaptive = frame[frame["strategy"] == "adaptive"]["mean_iteration_time_s"].mean()
        static = frame[frame["strategy"] != "adaptive"].groupby("strategy")["mean_iteration_time_s"].mean()
        if np.isnan(adaptive) or static.empty:
            continue
        best_static = static.min()
        best_name = static.idxmin()
        improvement = (best_static - adaptive) / best_static * 100.0
        rows.append({
            "workload": workload,
            "workers": workers,
            "best_static_strategy": best_name,
            "best_static_mean_iteration_time_s": best_static,
            "adaptive_mean_iteration_time_s": adaptive,
            "adaptive_improvement_percent": improvement,
        })
    pd.DataFrame(rows).to_csv(out / "improvement_table.csv", index=False)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Analyze distributed SGD experiment results.")
    parser.add_argument("--results", required=True, help="Directory containing steps.csv and summary.csv.")
    parser.add_argument("--out", required=True, help="Output directory for figures and analysis tables.")
    args = parser.parse_args(argv)

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = _clean_summary(pd.read_csv(results / "summary.csv"))
    steps = pd.read_csv(results / "steps.csv")

    plot_scalability(summary, out)
    plot_baseline_comparison(summary, out)
    plot_adaptive_timeline(steps, out)
    plot_sensitivity(summary, out)
    plot_failure(steps, out)
    write_improvement_table(summary, out)
    print(f"wrote figures and tables to {out}")


if __name__ == "__main__":
    main()
