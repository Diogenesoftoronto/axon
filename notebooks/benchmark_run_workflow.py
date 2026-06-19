import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Altum Benchmark Runner (Marimo Workflow)

        This notebook runs Altum benchmarks across one or more datasets, captures
        run-level and summary-level dataframes, and emits standardized artifacts
        for paper analysis.

        Run with:

        ```bash
        uv run --python .venv/bin/python marimo edit notebooks/benchmark_run_workflow.py
        ```
        """
    )
    return


@app.cell
def __():
    import json
    import os
    import re
    import time
    from dataclasses import asdict
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from scripts import benchmark_altum as bench

    return Path, asdict, bench, datetime, json, mo, os, pd, plt, re, time


@app.cell
def __():
    BENCHMARK_CATALOG = {
        "benchmarks/rlm_challenges.json": {
            "name": "Core RLM Challenges",
            "what_it_tests": "Algorithmic reasoning correctness on deterministic short-answer tasks.",
            "work_profile": "Baseline sanity suite for recursive decomposition quality and regression checks.",
        },
        "benchmarks/rlm_hard_coding_planning.json": {
            "name": "Hard Coding and Planning",
            "what_it_tests": "Long-horizon planning with multi-step state tracking and constrained execution reasoning.",
            "work_profile": "Primary stress test for deep planning behavior and policy robustness.",
        },
        "benchmarks/hallucination_guardrails.json": {
            "name": "Hallucination Guardrails",
            "what_it_tests": "Whether Altum abstains when evidence is insufficient instead of fabricating.",
            "work_profile": "Safety-oriented reliability check; complements pure pass-rate metrics.",
        },
        "benchmarks/long_context_books_distractor.json": {
            "name": "Long Context Books with Distractors",
            "what_it_tests": "Targeted retrieval and reasoning in long narrative context with confounding distractors.",
            "work_profile": "Evaluates long-context selectivity under realistic noise.",
        },
        "benchmarks/information_dense_ledger.json": {
            "name": "Information Dense Ledger",
            "what_it_tests": "Exact aggregation and bookkeeping across dense transaction records.",
            "work_profile": "Measures precision and consistency in high-entropy structured context.",
        },
        "benchmarks/mode_profile_targeted.json": {
            "name": "Mode Profile Targeted",
            "what_it_tests": "Behavioral differences across depth/iteration profiles on discriminative tasks.",
            "work_profile": "Designed for mode-ablation analysis and policy comparison.",
        },
        "benchmarks/codeforces_hard_like.json": {
            "name": "Codeforces Hard Like (Rust-Strict)",
            "what_it_tests": "Hard algorithmic coding under executable Rust-only answer constraints.",
            "work_profile": "Upper-bound difficulty suite for code-generation rigor.",
        },
    }
    DEFAULT_DATASETS = list(BENCHMARK_CATALOG.keys())
    MODE_PRESETS = {
        "d0-i1": ["--max-depth", "0", "--max-iterations", "1"],
        "d0-i3": ["--max-depth", "0", "--max-iterations", "3"],
        "d6-i1": ["--max-depth", "6", "--max-iterations", "1"],
        "d1-i3": ["--max-depth", "1", "--max-iterations", "3"],
    }
    MODE_LABELS = {
        "d0-i1": "Default (d0, i1)",
        "d0-i3": "No Recursion Best-of-3 (d0, i3)",
        "d6-i1": "Deep Recursion Single-Pass (d6, i1)",
        "d1-i3": "Depth-1 Best-of-3 (d1, i3)",
        "prev-d0-i1": "Previous Default (d0, i1)",
    }
    return BENCHMARK_CATALOG, DEFAULT_DATASETS, MODE_LABELS, MODE_PRESETS


@app.cell
def __(BENCHMARK_CATALOG, mo):
    lines = ["## Benchmark Catalog", ""]
    for dataset_path, meta in BENCHMARK_CATALOG.items():
        lines.append(f"### {meta['name']}")
        lines.append(f"- File: `{dataset_path}`")
        lines.append(f"- What it tests: {meta['what_it_tests']}")
        lines.append(f"- Work profile: {meta['work_profile']}")
        lines.append("")
    mo.md("\n".join(lines))
    return


@app.cell
def __(DEFAULT_DATASETS, MODE_PRESETS, mo):
    datasets = mo.ui.text_area(
        value="\n".join(DEFAULT_DATASETS),
        rows=8,
        label="Benchmark dataset files (one per line)",
    )
    selected_modes = mo.ui.multiselect(
        options=list(MODE_PRESETS.keys()),
        value=list(MODE_PRESETS.keys()),
        label="Mode profiles",
    )
    model_ids = mo.ui.text_area(
        value="hf:MiniMaxAI/MiniMax-M2.5",
        rows=2,
        label="Model ids (comma or newline separated; blank = Altum defaults)",
    )
    sub_model = mo.ui.text(
        value="",
        label="Sub-model override (blank = model id for recursive calls, or Altum default)",
    )

    base_url = mo.ui.text(value="https://api.synthetic.new/openai/v1/", label="Base URL")
    api_key_env = mo.ui.text(value="SYNTHETIC_API_KEY", label="API key env var")
    runs = mo.ui.number(value=1, start=1, stop=20, step=1, label="Runs per task")
    attempts = mo.ui.number(value=1, start=1, stop=10, step=1, label="Attempts per logical run")
    retry_backoff_s = mo.ui.number(value=2.0, start=0.0, stop=30.0, step=0.5, label="Retry backoff (s)")
    timeout_s = mo.ui.number(value=600, start=10, stop=3600, step=10, label="Timeout per Altum call (s)")
    max_tasks = mo.ui.number(value=0, start=0, stop=2000, step=1, label="Max tasks per dataset (0 = all)")
    fetch_pricing = mo.ui.checkbox(value=True, label="Fetch pricing from /models API")
    prompt_cost = mo.ui.number(value=0.0, start=0.0, stop=50.0, step=0.01, label="Prompt cost USD per 1M tokens")
    completion_cost = mo.ui.number(value=0.0, start=0.0, stop=50.0, step=0.01, label="Completion cost USD per 1M tokens")
    altum_verbose = mo.ui.checkbox(value=False, label="Enable `-v` on Altum runs")

    add_prev_ref = mo.ui.checkbox(value=False, label="Include `previous-default` mode from git ref")
    prev_ref = mo.ui.text(value="", label="Previous git ref (when enabled)")

    scoring_policy = mo.ui.text(value="typed-lenient", label="Scoring policy tag")
    runtime_policy = mo.ui.text(value="runtime-hardened", label="Runtime policy tag")
    evidence_tier = mo.ui.dropdown(options=["tier-a", "tier-b", "diagnostic"], value="tier-b", label="Evidence tier tag")
    experiment_tag = mo.ui.text(value="marimo-benchmark", label="Experiment tag (artifact folder prefix)")
    output_root = mo.ui.text(value="benchmarks/marimo-runs", label="Artifact output root")

    run_btn = mo.ui.run_button(label="Run Benchmarks")

    controls = mo.vstack(
        [
            mo.hstack([datasets, selected_modes], align="start", widths=[2, 1]),
            mo.hstack([model_ids, sub_model], align="start"),
            mo.hstack([base_url, api_key_env], justify="start"),
            mo.hstack([runs, attempts, retry_backoff_s, timeout_s, max_tasks], justify="start"),
            mo.hstack([fetch_pricing, prompt_cost, completion_cost, altum_verbose], justify="start"),
            mo.hstack([add_prev_ref, prev_ref], justify="start"),
            mo.hstack([scoring_policy, runtime_policy, evidence_tier], justify="start"),
            mo.hstack([experiment_tag, output_root, run_btn], justify="start"),
        ]
    )

    return (
        add_prev_ref,
        api_key_env,
        attempts,
        altum_verbose,
        base_url,
        completion_cost,
        controls,
        datasets,
        evidence_tier,
        experiment_tag,
        fetch_pricing,
        max_tasks,
        model_ids,
        output_root,
        prev_ref,
        prompt_cost,
        retry_backoff_s,
        run_btn,
        runs,
        scoring_policy,
        selected_modes,
        sub_model,
        runtime_policy,
        timeout_s,
    )


@app.cell
def __(controls, mo):
    mo.md("## Controls")
    controls
    return


@app.cell
def __(BENCHMARK_CATALOG, MODE_LABELS, Path, re):
    def parse_lines(text: str) -> list[str]:
        out: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            out.extend(part.strip() for part in line.split(",") if part.strip())
        return out

    def safe_slug(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip())
        return slug.strip("-") or "run"

    def dataset_label(dataset_path: str) -> str:
        meta = BENCHMARK_CATALOG.get(dataset_path)
        if meta and meta.get("name"):
            return str(meta["name"])
        stem = Path(dataset_path).stem
        return stem.replace("_", " ").replace("-", " ").title()

    def mode_label(mode_name: str) -> str:
        return MODE_LABELS.get(mode_name, mode_name.replace("-", " ").title())

    return dataset_label, mode_label, parse_lines, safe_slug


@app.cell
def __(
    MODE_PRESETS,
    Path,
    add_prev_ref,
    api_key_env,
    asdict,
    attempts,
    altum_verbose,
    base_url,
    bench,
    completion_cost,
    datasets,
    datetime,
    evidence_tier,
    experiment_tag,
    fetch_pricing,
    json,
    dataset_label,
    max_tasks,
    model_ids,
    mode_label,
    os,
    output_root,
    parse_lines,
    pd,
    prev_ref,
    prompt_cost,
    retry_backoff_s,
    run_btn,
    runs,
    safe_slug,
    scoring_policy,
    selected_modes,
    sub_model,
    runtime_policy,
    time,
    timeout_s,
):
    artifact_dir = None
    df_runs = pd.DataFrame()
    df_summary = pd.DataFrame()
    log_df = pd.DataFrame()
    report_rows = {}
    if run_btn.value:
        api_key = os.environ.get(api_key_env.value, "")
        dataset_files = parse_lines(datasets.value)
        selected = list(selected_modes.value)

        if not api_key:
            log_df = pd.DataFrame([{"message": f"Missing API key in env var {api_key_env.value}"}])
        elif not dataset_files:
            log_df = pd.DataFrame([{"message": "No dataset files provided"}])
        elif not selected:
            log_df = pd.DataFrame([{"message": "No mode profiles selected"}])
        else:
            models = parse_lines(model_ids.value)
            if not models:
                models = [""]

            repo_dir = Path(".").resolve()
            current_bin = bench.ensure_release_binary(repo_dir)
            modes = [bench.Mode(name, str(current_bin), MODE_PRESETS[name]) for name in selected]

            prev_ref_issue = False
            if add_prev_ref.value:
                ref = prev_ref.value.strip()
                if ref:
                    prev_bin = bench.build_previous_binary(repo_dir, ref)
                    modes.append(
                        bench.Mode("prev-d0-i1", str(prev_bin), ["--max-depth", "0", "--max-iterations", "1"])
                    )
                else:
                    prev_ref_issue = True

            logs: list[str] = []
            if prev_ref_issue:
                log_df = pd.DataFrame([{"message": "`Include previous-default` enabled but no git ref provided"}])
            else:
                pricing = None
                if fetch_pricing.value:
                    try:
                        pricing = bench.fetch_openai_models_pricing(base_url.value, api_key)
                        logs.append(f"Fetched pricing rows: {len(pricing)}")
                    except Exception as exc:
                        logs.append(f"Warning: failed to fetch /models pricing: {exc}")
                        pricing = None

                sub_model_override = sub_model.value.strip() or None
                run_rows: list[dict] = []

                for dataset_file in dataset_files:
                    task_path = (repo_dir / dataset_file).resolve()
                    if not task_path.exists():
                        logs.append(f"Skipped missing dataset: {dataset_file}")
                        continue

                    task_list = bench.load_tasks(task_path)
                    task_cap = int(max_tasks.value)
                    if task_cap > 0:
                        task_list = task_list[:task_cap]
                    if not task_list:
                        logs.append(f"Skipped empty dataset: {dataset_file}")
                        continue

                    report_rows[dataset_file] = []
                    for model in models:
                        effective_sub_model = sub_model_override
                        if model and not effective_sub_model:
                            effective_sub_model = model

                        model_label = model or "(default)"
                        for mode in modes:
                            for task in task_list:
                                for idx in range(int(runs.value)):
                                    final_result = None
                                    for attempt_idx in range(1, int(attempts.value) + 1):
                                        rr = bench.run_one(
                                            mode=mode,
                                            task=task,
                                            base_url=base_url.value,
                                            api_key=api_key,
                                            model=(model or None),
                                            sub_model=(effective_sub_model or None),
                                            timeout_s=int(timeout_s.value),
                                            prompt_cost_per_1m=float(prompt_cost.value),
                                            completion_cost_per_1m=float(completion_cost.value),
                                            altum_verbose=bool(altum_verbose.value),
                                            is_hallucination_probe=(
                                                bench.normalize_text(task.get("check", {}).get("value", ""))
                                                == "insufficient_information"
                                            ),
                                            pricing=pricing,
                                        )
                                        rr.run_idx = idx + 1
                                        final_result = rr
                                        if (
                                            not rr.ok
                                            and attempt_idx < int(attempts.value)
                                            and bench.is_transient_error(rr.error)
                                        ):
                                            delay = float(retry_backoff_s.value) * float(attempt_idx)
                                            logs.append(
                                                f"[{Path(dataset_file).name}][{model_label}][{mode.name}] {task.get('id', '')} run={idx+1} transient error; retry {attempt_idx+1}/{int(attempts.value)} after {delay:.1f}s"
                                            )
                                            if delay > 0:
                                                time.sleep(delay)
                                            continue
                                        break

                                    assert final_result is not None
                                    report_rows[dataset_file].append(final_result)
                                    row = asdict(final_result)
                                    row["dataset"] = dataset_file
                                    row["dataset_name"] = Path(dataset_file).name
                                    row["dataset_label"] = dataset_label(dataset_file)
                                    row["mode_label"] = mode_label(final_result.mode)
                                    row["scoring_policy"] = scoring_policy.value.strip()
                                    row["runtime_policy"] = runtime_policy.value.strip()
                                    row["evidence_tier"] = evidence_tier.value
                                    row["experiment_tag"] = experiment_tag.value.strip()
                                    run_rows.append(row)

                                    status = "PASS" if final_result.matched else "FAIL"
                                    if not final_result.ok:
                                        status = "ERROR"
                                    logs.append(
                                        f"[{Path(dataset_file).name}][{model_label}][{mode.name}] {task.get('id', '')} run={idx+1} -> {status} ({final_result.elapsed_s:.2f}s)"
                                    )

                df_runs = pd.DataFrame(run_rows)
                summary_rows = []
                for dataset_file, rows in report_rows.items():
                    if not rows:
                        continue
                    for s in bench.summarize(rows):
                        s["dataset"] = dataset_file
                        s["dataset_name"] = Path(dataset_file).name
                        s["dataset_label"] = dataset_label(dataset_file)
                        s["mode_label"] = mode_label(s["mode"])
                        s["scoring_policy"] = scoring_policy.value.strip()
                        s["runtime_policy"] = runtime_policy.value.strip()
                        s["evidence_tier"] = evidence_tier.value
                        s["experiment_tag"] = experiment_tag.value.strip()
                        summary_rows.append(s)

                df_summary = pd.DataFrame(summary_rows)
                if df_summary.empty:
                    log_df = pd.DataFrame({"message": logs or ["No results produced"]})
                else:
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    run_label = safe_slug(experiment_tag.value.strip() or "marimo-benchmark")
                    out_dir = Path(output_root.value).resolve() / f"{run_label}-{stamp}"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    for dataset_file, rows in report_rows.items():
                        if not rows:
                            continue
                        stem = Path(dataset_file).stem
                        ds_summary = [r for r in summary_rows if r["dataset"] == dataset_file]
                        payload = {
                            "created_at": int(time.time()),
                            "dataset": dataset_file,
                            "base_url": base_url.value,
                            "models": models,
                            "sub_model": sub_model_override,
                            "runs": int(runs.value),
                            "attempts_per_run": int(attempts.value),
                            "retry_backoff_s": float(retry_backoff_s.value),
                            "timeout": int(timeout_s.value),
                            "pricing_from_models_api": bool(fetch_pricing.value),
                            "scoring_policy": scoring_policy.value.strip(),
                            "runtime_policy": runtime_policy.value.strip(),
                            "evidence_tier": evidence_tier.value,
                            "experiment_tag": experiment_tag.value.strip(),
                            "summary": ds_summary,
                            "results": [asdict(r) for r in rows],
                        }
                        out_path = out_dir / f"results-{stem}.json"
                        out_path.write_text(json.dumps(payload, indent=2))
                        bench.write_summary_markdown(ds_summary, out_dir / f"summary-{stem}.md")

                    table_lines = [
                        "| Dataset | Mode | Pass rate | 95% CI | Avg cost (USD) | Avg time (s) | Readiness |",
                        "|---|---|---:|---:|---:|---:|---|",
                    ]
                    for row in sorted(summary_rows, key=lambda r: (r["dataset_label"], r["mode_label"])):
                        ci = f"[{row['pass_rate_ci95_low']:.2f}, {row['pass_rate_ci95_high']:.2f}]"
                        table_lines.append(
                            f"| {row['dataset_label']} | {row['mode_label']} | {row['pass_rate']:.2f}% | {ci} | {row['avg_cost_usd']:.6f} | {row['avg_time_s']:.3f} | {row['readiness']} |"
                        )

                    (out_dir / "overall-summary.md").write_text("\n".join(table_lines) + "\n")
                    df_runs.to_csv(out_dir / "runs.csv", index=False)
                    df_summary.to_csv(out_dir / "summary.csv", index=False)
                    (out_dir / "run.log").write_text("\n".join(logs) + "\n")

                    artifact_dir = str(out_dir)
                    log_df = pd.DataFrame({"message": logs})

    return artifact_dir, df_runs, df_summary, log_df, report_rows


@app.cell
def __(artifact_dir, df_runs, df_summary, mo):
    if df_runs is not None:
        if artifact_dir is None:
            _ = mo.md("Benchmark run is ready. No artifacts were written because no successful dataset run completed.")
        else:
            _ = mo.md(f"Artifacts written to `{artifact_dir}`")
            _ = mo.md(f"Run rows: **{len(df_runs)}**, summary rows: **{len(df_summary)}**")
    return


@app.cell
def __(df_summary, mo):
    if df_summary is None or df_summary.empty:
        _ = mo.md("Run the benchmark to generate summary metrics.")
    else:
        _ = mo.md("## Summary DataFrame")
        _ = mo.ui.table(df_summary.sort_values(["dataset_label", "mode_label", "model"]))
    return


@app.cell
def __(df_runs, mo):
    if df_runs is not None and not df_runs.empty:
        _ = mo.md("## Run-Level DataFrame")
        _ = mo.ui.table(df_runs.sort_values(["dataset_label", "mode_label", "task_id", "run_idx"]))
    return


@app.cell
def __(df_summary, pd, plt):
    if df_summary is not None and not df_summary.empty:
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
        _ax.set_title("Pass-rate heatmap by dataset and mode")
        for i in range(len(_pivot.index)):
            for j in range(len(_pivot.columns)):
                _ax.text(j, i, f"{_pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
        _fig.colorbar(_im, ax=_ax, fraction=0.03, pad=0.02)
        _fig.tight_layout()
        _ = _fig
    return


@app.cell
def __(df_summary, plt):
    if df_summary is not None and not df_summary.empty:
        _fig, _ax = plt.subplots(figsize=(8.5, 5))
        for dataset, grp in df_summary.groupby("dataset_label"):
            _ax.scatter(grp["avg_cost_usd"], grp["pass_rate"], s=45, alpha=0.85, label=dataset)
        _ax.set_xlabel("Average cost (USD)")
        _ax.set_ylabel("Pass rate (%)")
        _ax.set_title("Cost vs pass rate")
        _ax.grid(alpha=0.3)
        _ax.legend(fontsize=7)
        _fig.tight_layout()
        _ = _fig
    return


@app.cell
def __(df_runs, pd, plt):
    if df_runs is not None and not df_runs.empty:
        _tmp = df_runs.copy()
        _tmp["outcome"] = _tmp.apply(
            lambda r: "error" if (not bool(r["ok"])) else ("pass" if bool(r["matched"]) else "mismatch"), axis=1
        )
        _summary = (
            _tmp.groupby(["mode_label", "outcome"], as_index=False)
            .size()
            .pivot(index="mode_label", columns="outcome", values="size")
            .fillna(0)
        )
        for col in ["pass", "mismatch", "error"]:
            if col not in _summary.columns:
                _summary[col] = 0
        _summary = _summary[["pass", "mismatch", "error"]]

        _fig, _ax = plt.subplots(figsize=(10, 4.8))
        _bottom = None
        _colors = {"pass": "#54a24b", "mismatch": "#eeca3b", "error": "#e45756"}
        for col in ["pass", "mismatch", "error"]:
            vals = _summary[col].values
            if _bottom is None:
                _ax.bar(_summary.index, vals, label=col, color=_colors[col])
                _bottom = vals
            else:
                _ax.bar(_summary.index, vals, bottom=_bottom, label=col, color=_colors[col])
                _bottom = _bottom + vals
        _ax.set_ylabel("Run count")
        _ax.set_title("Outcome breakdown by mode")
        _ax.grid(axis="y", alpha=0.3)
        _ax.tick_params(axis="x", rotation=20)
        _ax.legend()
        _fig.tight_layout()
        _ = _fig
    return


@app.cell
def __(log_df, mo):
    if log_df is not None and not log_df.empty:
        _ = mo.md("## Run Log")
        _ = mo.ui.table(log_df.tail(400))
    return


if __name__ == "__main__":
    app.run()
