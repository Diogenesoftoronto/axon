import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # Axon Benchmark Analysis (Marimo Workflow)

    This notebook loads benchmark result artifacts, keeps the analysis dataframe-first,
    and generates publication-ready tables/plots with explicit filtering for evidence,
    scoring policy, runtime policy, and mode profiles.

    Run with:

    ```bash
    uv run --python .venv/bin/python marimo edit notebooks/benchmark_results_analysis.py
    ```
    """)
    return


@app.cell
def _():
    import json
    import time
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    return Path, datetime, json, mo, pd, plt


@app.cell
def _():
    BENCHMARK_CATALOG = {
        "rlm_challenges.json": {
            "name": "Core RLM Challenges",
            "what_it_tests": "Algorithmic reasoning correctness on deterministic short-answer tasks.",
            "work_profile": "Baseline sanity suite for recursive decomposition quality and regressions.",
        },
        "rlm_hard_coding_planning.json": {
            "name": "Hard Coding and Planning",
            "what_it_tests": "Long-horizon planning and constrained execution reasoning.",
            "work_profile": "Primary stress test for deep planning behavior.",
        },
        "hallucination_guardrails.json": {
            "name": "Hallucination Guardrails",
            "what_it_tests": "Abstention under insufficient evidence.",
            "work_profile": "Safety and anti-fabrication reliability check.",
        },
        "long_context_books_distractor.json": {
            "name": "Long Context Books with Distractors",
            "what_it_tests": "Retrieval and reasoning in long narrative context with distractors.",
            "work_profile": "Long-context selectivity and noise robustness.",
        },
        "information_dense_ledger.json": {
            "name": "Information Dense Ledger",
            "what_it_tests": "Exact aggregation over dense transaction-style context.",
            "work_profile": "Precision/consistency under high information density.",
        },
        "mode_profile_targeted.json": {
            "name": "Mode Profile Targeted",
            "what_it_tests": "Behavioral separation between depth/iteration profiles.",
            "work_profile": "Mode-ablation and policy profiling.",
        },
        "codeforces_hard_like.json": {
            "name": "Codeforces Hard Like (Rust-Strict)",
            "what_it_tests": "Hard algorithmic coding under executable Rust-only constraints.",
            "work_profile": "Upper-bound coding difficulty stress test.",
        },
        "hf_gsm8k_100.json": {
            "name": "HF GSM8K (100)",
            "what_it_tests": "Grade-school math reasoning and answer extraction.",
            "work_profile": "Math reasoning baseline adapter suite.",
        },
        "hf_mbpp_calls_100.json": {
            "name": "HF MBPP Calls (100)",
            "what_it_tests": "Program reasoning against function call expectations.",
            "work_profile": "Code understanding and synthesis baseline adapter suite.",
        },
        "hf_arc_challenge_100.json": {
            "name": "HF ARC Challenge (100)",
            "what_it_tests": "Multiple-choice scientific reasoning.",
            "work_profile": "Knowledge + reasoning baseline adapter suite.",
        },
        "hf_boolq_100.json": {
            "name": "HF BoolQ (100)",
            "what_it_tests": "Binary QA consistency from short passages.",
            "work_profile": "Fast yes/no grounding baseline adapter suite.",
        },
        "hf_hellaswag_100.json": {
            "name": "HF HellaSwag (100)",
            "what_it_tests": "Commonsense completion selection.",
            "work_profile": "Narrative plausibility baseline adapter suite.",
        },
    }
    MODE_LABELS = {
        "previous-default": "Previous Default (d0, i1)",
        "d0-i1": "Default (d0, i1)",
        "d0-i3": "No Recursion Best-of-3 (d0, i3)",
        "d1-i3": "Depth-1 Best-of-3 (d1, i3)",
        "d3-i1": "Depth-3 Single-Pass (d3, i1)",
        "d6-i1": "Deep Recursion Single-Pass (d6, i1)",
    }
    return BENCHMARK_CATALOG, MODE_LABELS


@app.cell
def _(BENCHMARK_CATALOG, MODE_LABELS, Path, json, pd):
    def dataset_label(name: str) -> str:
        key = Path(name or "").name
        meta = BENCHMARK_CATALOG.get(key)
        if meta and meta.get("name"):
            return str(meta["name"])
        stem = Path(key).stem
        return stem.replace("_", " ").replace("-", " ").title()

    def mode_label(name: str) -> str:
        n = str(name or "")
        if n in MODE_LABELS:
            return MODE_LABELS[n]
        return n.replace("_", " ").replace("-", " ").title()

    def dataset_family(name: str) -> str:
        s = (name or "").lower()
        if "plancraft" in s:
            return "plancraft"
        if "codeforces" in s:
            return "codeforces"
        if "books" in s:
            return "books"
        if "ledger" in s:
            return "ledger"
        if s.startswith("hf_"):
            return "hf"
        if "hallucination" in s:
            return "guardrail"
        if "mode_profile" in s:
            return "mode-profile"
        if "hard_coding" in s or "planning" in s:
            return "planning"
        return "other"

    def discover_reports(glob_pattern: str) -> list[Path]:
        out = []
        for p in sorted(Path(".").glob(glob_pattern)):
            if not p.is_file():
                continue
            try:
                payload = json.loads(p.read_text())
            except Exception:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("summary"), list):
                out.append(p)
        return out

    def load_reports(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary_frames = []
        run_frames = []
        for path in paths:
            payload = json.loads(path.read_text())
            dataset_path = str(payload.get("dataset", ""))
            dataset_name = Path(dataset_path).name if dataset_path else path.name
            source_dir = str(path.parent)
            common = {
                "source_file": str(path),
                "source_dir": source_dir,
                "dataset": dataset_path,
                "dataset_name": dataset_name,
                "dataset_label": dataset_label(dataset_name),
                "dataset_family": dataset_family(dataset_name),
                "scoring_policy": payload.get("scoring_policy", "unknown"),
                "runtime_policy": payload.get("runtime_policy", "unknown"),
                "evidence_tier": payload.get("evidence_tier", "unknown"),
                "experiment_tag": payload.get("experiment_tag", ""),
                "created_at": payload.get("created_at", 0),
            }

            rows = []
            for s in payload.get("summary", []):
                row = {
                    **common,
                    "mode": s.get("mode", ""),
                    "mode_label": mode_label(s.get("mode", "")),
                    "model": s.get("model", ""),
                    "sub_model": s.get("sub_model", ""),
                    "total": int(s.get("total", 0) or 0),
                    "ok": int(s.get("ok", 0) or 0),
                    "matched": int(s.get("matched", 0) or 0),
                    "pass_rate": float(s.get("pass_rate", 0.0) or 0.0),
                    "pass_rate_ci95_low": float(s.get("pass_rate_ci95_low", 0.0) or 0.0),
                    "pass_rate_ci95_high": float(s.get("pass_rate_ci95_high", 0.0) or 0.0),
                    "ok_rate": float(s.get("ok_rate", 0.0) or 0.0),
                    "avg_time_s": float(s.get("avg_time_s", 0.0) or 0.0),
                    "avg_tokens_per_task": float(s.get("avg_tokens_per_task", 0.0) or 0.0),
                    "avg_calls_per_task": float(s.get("avg_calls_per_task", 0.0) or 0.0),
                    "avg_cost_usd": float(s.get("avg_cost_usd", 0.0) or 0.0),
                    "hallucination_rate": float(s.get("hallucination_rate", 0.0) or 0.0),
                    "composite_score": float(s.get("composite_score", 0.0) or 0.0),
                    "readiness": s.get("readiness", ""),
                }
                rows.append(row)
            if rows:
                summary_frames.append(pd.DataFrame(rows))

            run_rows = []
            for r in payload.get("results", []):
                ok = bool(r.get("ok", False))
                matched = bool(r.get("matched", False))
                outcome = "error"
                if ok and matched:
                    outcome = "pass"
                elif ok and not matched:
                    outcome = "mismatch"
                run_rows.append(
                    {
                        **common,
                        "mode": r.get("mode", ""),
                        "mode_label": mode_label(r.get("mode", "")),
                        "model": r.get("model", ""),
                        "sub_model": r.get("sub_model", ""),
                        "task_id": r.get("task_id", ""),
                        "run_idx": int(r.get("run_idx", 0) or 0),
                        "ok": ok,
                        "matched": matched,
                        "outcome": outcome,
                        "elapsed_s": float(r.get("elapsed_s", 0.0) or 0.0),
                        "total_tokens": int(r.get("total_tokens", 0) or 0),
                        "llm_calls": int(r.get("llm_calls", 0) or 0),
                        "estimated_cost_usd": float(r.get("estimated_cost_usd", 0.0) or 0.0),
                    }
                )
            if run_rows:
                run_frames.append(pd.DataFrame(run_rows))

        if summary_frames:
            df_summary = pd.concat(summary_frames, ignore_index=True)
        else:
            df_summary = pd.DataFrame(
                columns=[
                    "source_file",
                    "source_dir",
                    "dataset",
                    "dataset_name",
                    "dataset_label",
                    "dataset_family",
                    "mode",
                    "mode_label",
                    "pass_rate",
                    "pass_rate_ci95_low",
                    "pass_rate_ci95_high",
                    "avg_cost_usd",
                    "avg_time_s",
                    "avg_tokens_per_task",
                    "scoring_policy",
                    "runtime_policy",
                    "evidence_tier",
                ]
            )

        if run_frames:
            df_runs = pd.concat(run_frames, ignore_index=True)
        else:
            df_runs = pd.DataFrame(
                columns=[
                    "source_file",
                    "dataset_name",
                    "dataset_label",
                    "mode",
                    "mode_label",
                    "task_id",
                    "run_idx",
                    "ok",
                    "matched",
                    "outcome",
                ]
            )

        return df_summary, df_runs

    def load_task_profile(task_files: list[str]) -> pd.DataFrame:
        frames = []
        for file in task_files:
            p = Path(file)
            if not p.exists():
                continue
            try:
                payload = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(payload, list):
                continue
            rows = []
            for task in payload:
                rows.append(
                    {
                        "dataset": p.name,
                        "dataset_label": dataset_label(p.name),
                        "dataset_family": dataset_family(p.name),
                        "context_chars": len(task.get("context", "")),
                        "query_chars": len(task.get("query", "")),
                        "check_type": str(task.get("check", {}).get("type", "")),
                    }
                )
            frames.append(pd.DataFrame(rows))

        if not frames:
            return pd.DataFrame(
                columns=["dataset", "dataset_label", "dataset_family", "context_chars", "query_chars", "check_type"]
            )
        return pd.concat(frames, ignore_index=True)

    return discover_reports, load_reports, load_task_profile


@app.cell
def _(mo):
    results_glob = mo.ui.text(
        value="benchmarks/results/results-mode-profile-multimodel-smoke-20260226.json",
        label="Result report glob",
    )
    task_files = mo.ui.text_area(
        value="\n".join(
            [
                "benchmarks/hf_gsm8k_100.json",
                "benchmarks/hf_mbpp_calls_100.json",
                "benchmarks/hf_arc_challenge_100.json",
                "benchmarks/hf_boolq_100.json",
                "benchmarks/hf_hellaswag_100.json",
                "benchmarks/mode_profile_targeted.json",
                "benchmarks/codeforces_hard_like.json",
                "benchmarks/long_context_books_distractor.json",
                "benchmarks/information_dense_ledger.json",
            ]
        ),
        rows=8,
        label="Task files for dataset profiling (one per line)",
    )
    return results_glob, task_files


@app.cell
def _(mo, results_glob, task_files):
    mo.output.replace(mo.vstack([results_glob, task_files]))
    return


@app.cell
def _(BENCHMARK_CATALOG, mo):
    catalog_lines = ["## Benchmark Catalog", ""]
    for key, meta in BENCHMARK_CATALOG.items():
        catalog_lines.append(f"### {meta['name']}")
        catalog_lines.append(f"- File: `benchmarks/{key}`")
        catalog_lines.append(f"- What it tests: {meta['what_it_tests']}")
        catalog_lines.append(f"- Work profile: {meta['work_profile']}")
        catalog_lines.append("")
    mo.output.replace(mo.md("\n".join(catalog_lines)))
    return


@app.cell
def _(
    discover_reports,
    load_reports,
    load_task_profile,
    results_glob,
    task_files,
):
    report_paths = discover_reports(results_glob.value)
    df_summary_raw, df_runs_raw = load_reports(report_paths)
    task_file_list = [line.strip() for line in task_files.value.splitlines() if line.strip()]
    df_tasks = load_task_profile(task_file_list)
    return df_runs_raw, df_summary_raw, df_tasks, report_paths


@app.cell
def _(df_summary_raw, mo):
    dataset_options = sorted(df_summary_raw["dataset_label"].dropna().unique().tolist()) if not df_summary_raw.empty else []
    mode_options = sorted(df_summary_raw["mode_label"].dropna().unique().tolist()) if not df_summary_raw.empty else []
    family_options = sorted(df_summary_raw["dataset_family"].dropna().unique().tolist()) if not df_summary_raw.empty else []
    scoring_options = sorted(df_summary_raw["scoring_policy"].dropna().unique().tolist()) if not df_summary_raw.empty else []
    runtime_options = sorted(df_summary_raw["runtime_policy"].dropna().unique().tolist()) if not df_summary_raw.empty else []
    tier_options = sorted(df_summary_raw["evidence_tier"].dropna().unique().tolist()) if not df_summary_raw.empty else []

    dataset_filter = mo.ui.multiselect(options=dataset_options, value=dataset_options, label="Dataset filter")
    mode_filter = mo.ui.multiselect(options=mode_options, value=mode_options, label="Mode filter")
    family_filter = mo.ui.multiselect(options=family_options, value=family_options, label="Dataset family filter")
    scoring_filter = mo.ui.multiselect(options=scoring_options, value=scoring_options, label="Scoring policy filter")
    runtime_filter = mo.ui.multiselect(options=runtime_options, value=runtime_options, label="Runtime policy filter")
    tier_filter = mo.ui.multiselect(options=tier_options, value=tier_options, label="Evidence tier filter")

    filter_controls = mo.vstack(
        [
            mo.hstack([dataset_filter, mode_filter], align="start"),
            mo.hstack([family_filter, scoring_filter, runtime_filter, tier_filter], align="start"),
        ]
    )
    return (
        dataset_filter,
        family_filter,
        filter_controls,
        mode_filter,
        runtime_filter,
        scoring_filter,
        tier_filter,
    )


@app.cell
def _(filter_controls, mo):
    mo.output.replace(mo.vstack([mo.md("## Analysis Filters"), filter_controls]))
    return


@app.cell
def _(
    dataset_filter,
    df_runs_raw,
    df_summary_raw,
    family_filter,
    mode_filter,
    runtime_filter,
    scoring_filter,
    tier_filter,
):
    df_summary = df_summary_raw.copy()
    df_runs = df_runs_raw.copy()

    if not df_summary.empty:
        if dataset_filter.value:
            df_summary = df_summary[df_summary["dataset_label"].isin(dataset_filter.value)]
        if mode_filter.value:
            df_summary = df_summary[df_summary["mode_label"].isin(mode_filter.value)]
        if family_filter.value:
            df_summary = df_summary[df_summary["dataset_family"].isin(family_filter.value)]
        if scoring_filter.value:
            df_summary = df_summary[df_summary["scoring_policy"].isin(scoring_filter.value)]
        if runtime_filter.value:
            df_summary = df_summary[df_summary["runtime_policy"].isin(runtime_filter.value)]
        if tier_filter.value:
            df_summary = df_summary[df_summary["evidence_tier"].isin(tier_filter.value)]

    if not df_runs.empty:
        if dataset_filter.value:
            df_runs = df_runs[df_runs["dataset_label"].isin(dataset_filter.value)]
        if mode_filter.value:
            df_runs = df_runs[df_runs["mode_label"].isin(mode_filter.value)]
        if family_filter.value:
            df_runs = df_runs[df_runs["dataset_family"].isin(family_filter.value)]
        if scoring_filter.value:
            df_runs = df_runs[df_runs["scoring_policy"].isin(scoring_filter.value)]
        if runtime_filter.value:
            df_runs = df_runs[df_runs["runtime_policy"].isin(runtime_filter.value)]
        if tier_filter.value:
            df_runs = df_runs[df_runs["evidence_tier"].isin(tier_filter.value)]
    return df_runs, df_summary


@app.cell
def _(df_summary, mo, report_paths):
    _status_lines = [mo.md(f"Loaded reports: **{len(report_paths)}**")]
    if df_summary.empty:
        _status_lines.append(mo.md("No matching benchmark summaries found for current filters."))
    else:
        _status_lines.append(mo.md(f"Filtered summary rows: **{len(df_summary)}**"))
    mo.output.replace(mo.vstack(_status_lines))
    return


@app.cell
def _(df_summary, mo):
    if not df_summary.empty:
        _top_rows = (
            df_summary.sort_values(["pass_rate", "avg_cost_usd"], ascending=[False, True])
            .loc[:, ["dataset_label", "mode_label", "pass_rate", "avg_cost_usd", "avg_time_s", "source_dir"]]
            .head(15)
        )
        _out = mo.vstack([mo.md("## Top Configurations (Filtered)"), mo.ui.table(_top_rows)])
    else:
        _out = mo.md("No configurations available for current filters.")
    mo.output.replace(_out)
    return


@app.cell
def _(df_summary, mo):
    if not df_summary.empty:
        _mode_rank = (
            df_summary.groupby("mode_label", as_index=False)
            .agg(
                mean_pass_rate=("pass_rate", "mean"),
                mean_cost=("avg_cost_usd", "mean"),
                mean_time_s=("avg_time_s", "mean"),
            )
            .sort_values("mean_pass_rate", ascending=False)
        )
        _best = _mode_rank.iloc[0]
        _out = mo.md(
            f"### Filtered Snapshot\n"
            f"- Best mean pass-rate mode: **{_best['mode_label']}** ({_best['mean_pass_rate']:.2f}%)\n"
            f"- Mean cost for best mode: **${_best['mean_cost']:.6f}** per task\n"
            f"- Mean latency for best mode: **{_best['mean_time_s']:.2f}s** per task"
        )
    else:
        _out = mo.md("Filtered snapshot unavailable with current filters.")
    mo.output.replace(_out)
    return


@app.cell
def _(df_summary, mo):
    if not df_summary.empty:
        _out = mo.vstack(
            [
                mo.md("## Summary DataFrame"),
                mo.ui.table(df_summary.sort_values(["dataset_label", "mode_label", "model", "source_file"])),
            ]
        )
    else:
        _out = mo.md("Summary table is empty for current filters.")
    mo.output.replace(_out)
    return


@app.cell
def _(df_runs, mo):
    if not df_runs.empty:
        _out = mo.vstack(
            [
                mo.md("## Run-Level DataFrame"),
                mo.ui.table(df_runs.sort_values(["dataset_label", "mode_label", "task_id", "run_idx"])),
            ]
        )
    else:
        _out = mo.md("Run-level table is empty for current filters.")
    mo.output.replace(_out)
    return


@app.cell
def _(df_summary, mo, plt):
    if not df_summary.empty:
        _pivot = (
            df_summary.groupby(["dataset_label", "mode_label"], as_index=False)["pass_rate"]
            .mean()
            .pivot(index="dataset_label", columns="mode_label", values="pass_rate")
            .fillna(0.0)
        )

        _fig, _ax = plt.subplots(figsize=(10, max(4.5, 0.4 * len(_pivot.index) + 2)))
        _im = _ax.imshow(_pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
        _ax.set_xticks(range(len(_pivot.columns)))
        _ax.set_xticklabels(_pivot.columns, rotation=35, ha="right")
        _ax.set_yticks(range(len(_pivot.index)))
        _ax.set_yticklabels(_pivot.index)
        _ax.set_title("Pass-rate heatmap")
        for i in range(len(_pivot.index)):
            for j in range(len(_pivot.columns)):
                _ax.text(j, i, f"{_pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
        _fig.colorbar(_im, ax=_ax, fraction=0.03, pad=0.02)
        _fig.tight_layout()
        _out = _fig
    else:
        _out = mo.md("Pass-rate heatmap unavailable for current filters.")
    mo.output.replace(_out)
    return


@app.cell
def _(df_summary, mo, plt):
    if not df_summary.empty:
        _fig, _ax = plt.subplots(figsize=(8.5, 5.2))
        for dataset, grp in df_summary.groupby("dataset_label"):
            _ax.scatter(grp["avg_cost_usd"], grp["pass_rate"], s=45, alpha=0.85, label=dataset)
        _ax.set_xlabel("Average cost (USD)")
        _ax.set_ylabel("Pass rate (%)")
        _ax.set_title("Cost vs pass-rate frontier")
        _ax.grid(alpha=0.3)
        _ax.legend(fontsize=7)
        _fig.tight_layout()
        _out = _fig
    else:
        _out = mo.md("Cost vs pass-rate chart unavailable for current filters.")
    mo.output.replace(_out)
    return


@app.cell
def _(df_runs, mo, plt):
    if not df_runs.empty:
        _agg = (
            df_runs.groupby(["mode_label", "outcome"], as_index=False)
            .size()
            .pivot(index="mode_label", columns="outcome", values="size")
            .fillna(0)
        )
        for col in ["pass", "mismatch", "error"]:
            if col not in _agg.columns:
                _agg[col] = 0
        _agg = _agg[["pass", "mismatch", "error"]]

        _fig, _ax = plt.subplots(figsize=(10, 4.8))
        _bottom = None
        _colors = {"pass": "#54a24b", "mismatch": "#eeca3b", "error": "#e45756"}
        for col in ["pass", "mismatch", "error"]:
            vals = _agg[col].values
            if _bottom is None:
                _ax.bar(_agg.index, vals, label=col, color=_colors[col])
                _bottom = vals
            else:
                _ax.bar(_agg.index, vals, bottom=_bottom, label=col, color=_colors[col])
                _bottom = _bottom + vals
        _ax.set_ylabel("Run count")
        _ax.set_title("Outcome breakdown by mode")
        _ax.tick_params(axis="x", rotation=20)
        _ax.grid(axis="y", alpha=0.3)
        _ax.legend()
        _fig.tight_layout()
        _out = _fig
    else:
        _out = mo.md("Outcome breakdown chart unavailable for current filters.")
    mo.output.replace(_out)
    return


@app.cell
def _(df_summary, mo, plt):
    if not df_summary.empty:
        _mode_ci = (
            df_summary.groupby("mode_label", as_index=False)
            .agg(
                pass_rate=("pass_rate", "mean"),
                ci_low=("pass_rate_ci95_low", "mean"),
                ci_high=("pass_rate_ci95_high", "mean"),
            )
            .sort_values("pass_rate", ascending=False)
        )

        _fig, _ax = plt.subplots(figsize=(9, 4.8))
        _x = range(len(_mode_ci))
        _y = _mode_ci["pass_rate"].values
        # Averaging CI endpoints across mixed slices can invert bounds; clamp around mean.
        _ci_low = _mode_ci["ci_low"].values
        _ci_high = _mode_ci["ci_high"].values
        _yerr_low = (_y - _ci_low).clip(min=0)
        _yerr_high = (_ci_high - _y).clip(min=0)
        _ax.errorbar(_x, _y, yerr=[_yerr_low, _yerr_high], fmt="o", capsize=4)
        _ax.set_xticks(list(_x))
        _ax.set_xticklabels(_mode_ci["mode_label"], rotation=20)
        _ax.set_ylabel("Pass rate (%)")
        _ax.set_title("Mode pass rate with average CI bounds")
        _ax.grid(axis="y", alpha=0.3)
        _fig.tight_layout()
        _out = _fig
    else:
        _out = mo.md("Mode CI chart unavailable for current filters.")
    mo.output.replace(_out)
    return


@app.cell
def _(df_tasks, mo):
    profile = None
    if not df_tasks.empty:
        profile = (
            df_tasks.groupby("dataset_label", as_index=False)
            .agg(
                tasks=("dataset_label", "size"),
                avg_context_chars=("context_chars", "mean"),
                avg_query_chars=("query_chars", "mean"),
            )
            .sort_values("dataset_label")
        )
        _out = mo.vstack([mo.md("## Task Dataset Profile"), mo.ui.table(profile)])
    else:
        _out = mo.md("Task dataset profile unavailable (task files not found).")
    mo.output.replace(_out)
    return (profile,)


@app.cell
def _(mo, plt, profile):
    if profile is not None and not profile.empty:
        _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2))
        _axes[0].bar(profile["dataset_label"], profile["tasks"], color="#4c78a8")
        _axes[0].set_title("Task count")
        _axes[0].tick_params(axis="x", rotation=35, labelsize=8)

        _axes[1].bar(profile["dataset_label"], profile["avg_context_chars"], color="#f58518")
        _axes[1].set_title("Average context chars")
        _axes[1].tick_params(axis="x", rotation=35, labelsize=8)

        _axes[2].bar(profile["dataset_label"], profile["avg_query_chars"], color="#54a24b")
        _axes[2].set_title("Average query chars")
        _axes[2].tick_params(axis="x", rotation=35, labelsize=8)

        _fig.tight_layout()
        _out = _fig
    else:
        _out = mo.md("Task profile charts unavailable.")
    mo.output.replace(_out)
    return


@app.cell
def _(mo):
    export_root = mo.ui.text(value="benchmarks/analysis-marimo", label="Export root")
    export_btn = mo.ui.run_button(label="Export filtered tables")
    mo.output.replace(mo.hstack([export_root, export_btn], justify="start"))
    return export_btn, export_root


@app.cell
def _(
    Path,
    datetime,
    df_runs,
    df_summary,
    export_btn,
    export_root,
    mo,
    profile,
):
    if export_btn.value:
        if df_summary.empty:
            _out = mo.md("No filtered summary rows to export.")
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = Path(export_root.value).resolve() / f"analysis-{stamp}"
            out_dir.mkdir(parents=True, exist_ok=True)

            df_summary.to_csv(out_dir / "summary_filtered.csv", index=False)
            if not df_runs.empty:
                df_runs.to_csv(out_dir / "runs_filtered.csv", index=False)
            if profile is not None and not profile.empty:
                profile.to_csv(out_dir / "task_profile.csv", index=False)

            lines = [
                "| Dataset | Mode | Pass rate | 95% CI | Avg cost (USD) | Avg time (s) | Readiness |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
            for _, row in (
                df_summary.sort_values(["dataset_label", "mode_label"])
                .loc[:, [
                    "dataset_label",
                    "mode_label",
                    "pass_rate",
                    "pass_rate_ci95_low",
                    "pass_rate_ci95_high",
                    "avg_cost_usd",
                    "avg_time_s",
                    "readiness",
                ]]
                .iterrows()
            ):
                ci = f"[{row['pass_rate_ci95_low']:.2f}, {row['pass_rate_ci95_high']:.2f}]"
                lines.append(
                    f"| {row['dataset_label']} | {row['mode_label']} | {row['pass_rate']:.2f}% | {ci} | {row['avg_cost_usd']:.6f} | {row['avg_time_s']:.3f} | {row['readiness']} |"
                )
            (out_dir / "summary_filtered.md").write_text("\n".join(lines) + "\n")

            _out = mo.md(f"Exported filtered analysis tables to `{out_dir}`")
        mo.output.replace(_out)
    return


if __name__ == "__main__":
    app.run()
