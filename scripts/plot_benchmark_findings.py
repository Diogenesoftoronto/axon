#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_task_dataset(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        return pd.DataFrame()
    for task in data:
        rows.append(
            {
                "dataset": path.name,
                "context_chars": len(task.get("context", "")),
                "query_chars": len(task.get("query", "")),
                "check_type": task.get("check", {}).get("type", ""),
            }
        )
    return pd.DataFrame(rows)


def load_result_summaries(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    payload = json.loads(path.read_text())
    dataset = Path(payload.get("dataset", path.name)).name
    for s in payload.get("summary", []):
        rows.append(
            {
                "source": path.name,
                "dataset": dataset,
                "mode": s.get("mode", ""),
                "pass_rate": float(s.get("pass_rate", 0.0)),
                "avg_cost_usd": float(s.get("avg_cost_usd", 0.0)),
                "avg_time_s": float(s.get("avg_time_s", 0.0)),
                "pass_ci_low": float(s.get("pass_rate_ci95_low", 0.0)),
                "pass_ci_high": float(s.get("pass_rate_ci95_high", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def plot_task_profile(df_tasks: pd.DataFrame, out_dir: Path) -> None:
    if df_tasks.empty:
        return

    profile = (
        df_tasks.groupby("dataset", as_index=False)
        .agg(tasks=("dataset", "size"), avg_context_chars=("context_chars", "mean"), avg_query_chars=("query_chars", "mean"))
        .sort_values("dataset")
    )
    profile.to_csv(out_dir / "task_dataset_profile.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].bar(profile["dataset"], profile["tasks"], color="#4C78A8")
    axes[0].set_title("Tasks per Dataset")
    axes[0].tick_params(axis="x", rotation=40, labelsize=8)

    axes[1].bar(profile["dataset"], profile["avg_context_chars"], color="#F58518")
    axes[1].set_title("Average Context Length")
    axes[1].tick_params(axis="x", rotation=40, labelsize=8)

    axes[2].bar(profile["dataset"], profile["avg_query_chars"], color="#54A24B")
    axes[2].set_title("Average Query Length")
    axes[2].tick_params(axis="x", rotation=40, labelsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "task_dataset_profile.png", dpi=180)
    plt.close(fig)


def plot_result_frontier(df_results: pd.DataFrame, out_dir: Path) -> None:
    if df_results.empty:
        return

    df_results.to_csv(out_dir / "benchmark_results_flat.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for dataset, group in df_results.groupby("dataset"):
        ax.scatter(group["avg_cost_usd"], group["pass_rate"], label=dataset, s=40, alpha=0.85)

    ax.set_xlabel("Average Cost (USD)")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Cost vs Pass Rate Across Benchmarks")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "results_cost_vs_pass.png", dpi=180)
    plt.close(fig)

    pivot = df_results.pivot_table(index="dataset", columns="mode", values="pass_rate", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.4 * len(pivot.index) + 2)))
    im = ax.imshow(pivot.fillna(0.0).values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Pass Rate Heatmap (Dataset x Mode)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.fillna(0.0).values[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "results_passrate_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate plots and summary CSVs for Altum benchmark findings")
    parser.add_argument(
        "--task-files",
        nargs="*",
        default=[
            "benchmarks/hf_gsm8k_100.json",
            "benchmarks/hf_mbpp_calls_100.json",
            "benchmarks/hf_arc_challenge_100.json",
            "benchmarks/hf_boolq_100.json",
            "benchmarks/hf_hellaswag_100.json",
        ],
        help="Task dataset JSON files (list-style benchmark files)",
    )
    parser.add_argument(
        "--results-glob",
        default="benchmarks/results/results-*.json",
        help="Glob for benchmark result report JSON files",
    )
    parser.add_argument("--out-dir", default="benchmarks/analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_frames = []
    for f in args.task_files:
        p = Path(f)
        if p.exists():
            task_frames.append(load_task_dataset(p))
    if task_frames:
        df_tasks = pd.concat(task_frames, ignore_index=True)
    else:
        df_tasks = pd.DataFrame()

    result_frames = []
    for p in sorted(Path(".").glob(args.results_glob)):
        result_frames.append(load_result_summaries(p))
    if result_frames:
        df_results = pd.concat(result_frames, ignore_index=True)
    else:
        df_results = pd.DataFrame()

    plot_task_profile(df_tasks, out_dir)
    plot_result_frontier(df_results, out_dir)

    print(f"Wrote analysis artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
