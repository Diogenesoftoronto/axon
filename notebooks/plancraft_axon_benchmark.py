import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def __(mo):
    mo.md(
        r"""
        # PlanCraft + Axon Benchmark (Marimo)

        This notebook evaluates Axon mode profiles on the PlanCraft benchmark by running
        Axon as the policy in the PlanCraft text environment.

        Run with:

        ```bash
        uv run --python .venv/bin/python marimo edit notebooks/plancraft_axon_benchmark.py
        ```
        """
    )
    return


@app.cell
def __():
    import json
    import os
    import random
    import re
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from plancraft.simple import PlancraftGymWrapper, get_plancraft_examples

    return (
        Path,
        PlancraftGymWrapper,
        get_plancraft_examples,
        mo,
        os,
        pd,
        plt,
        random,
        re,
        subprocess,
        tempfile,
        time,
    )


@app.cell
def __(Path):
    AXON_BIN = str(Path("target/release/axon").resolve())
    MODE_PRESETS = {
        "d0-i1": ["--max-depth", "0", "--max-iterations", "1"],
        "d0-i3": ["--max-depth", "0", "--max-iterations", "3"],
        "d6-i1": ["--max-depth", "6", "--max-iterations", "1"],
        "d1-i3": ["--max-depth", "1", "--max-iterations", "3"],
    }
    return AXON_BIN, MODE_PRESETS


@app.cell
def __(re):
    USAGE_RE = re.compile(
        r"usage: prompt_tokens=(?P<prompt>\d+) completion_tokens=(?P<completion>\d+) total_tokens=(?P<total>\d+) prompt_chars=(?P<prompt_chars>\d+)"
    )

    def parse_stderr_metrics(stderr: str) -> dict[str, int]:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        llm_calls = 0
        for line in stderr.splitlines():
            m = USAGE_RE.search(line)
            if m:
                llm_calls += 1
                prompt_tokens += int(m.group("prompt"))
                completion_tokens += int(m.group("completion"))
                total_tokens += int(m.group("total"))
        return {
            "llm_calls": llm_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    return (parse_stderr_metrics,)


@app.cell
def __(mo):
    split = mo.ui.dropdown(
        options=["test.small", "val.small", "test.small.easy", "val.small.easy", "test", "val"],
        value="test.small",
        label="PlanCraft split",
    )
    n_examples = mo.ui.number(value=12, start=1, stop=200, step=1, label="Number of examples")
    seed = mo.ui.number(value=42, start=0, stop=999999, step=1, label="Sample seed")
    modes = mo.ui.multiselect(
        options=[
            "d0-i1",
            "d0-i3",
            "d6-i1",
            "d1-i3",
        ],
        value=[
            "d0-i1",
            "d0-i3",
            "d6-i1",
            "d1-i3",
        ],
        label="Mode profiles",
    )
    base_url = mo.ui.text(value="https://api.synthetic.new/openai/v1/", label="Base URL")
    api_key_env = mo.ui.text(value="SYNTHETIC_API_KEY", label="API key env var")
    model = mo.ui.text(value="hf:MiniMaxAI/MiniMax-M2.5", label="Model")
    sub_model = mo.ui.text(value="hf:MiniMaxAI/MiniMax-M2.5", label="Sub-model")
    timeout_s = mo.ui.number(value=120, start=10, stop=1800, step=10, label="Per-step timeout (s)")
    max_steps = mo.ui.number(value=30, start=5, stop=80, step=1, label="Max environment steps")
    write_artifacts = mo.ui.checkbox(value=True, label="Write artifacts")
    artifact_root = mo.ui.text(value="benchmarks/plancraft-runs", label="Artifact output root")
    run_btn = mo.ui.run_button(label="Run PlanCraft Benchmark")

    controls = mo.vstack(
        [
            mo.hstack([split, n_examples, seed], justify="start"),
            modes,
            mo.hstack([base_url, api_key_env], justify="start"),
            mo.hstack([model, sub_model], justify="start"),
            mo.hstack([write_artifacts, artifact_root], justify="start"),
            mo.hstack([timeout_s, max_steps, run_btn], justify="start"),
        ]
    )
    return (
        artifact_root,
        api_key_env,
        base_url,
        controls,
        max_steps,
        model,
        modes,
        n_examples,
        run_btn,
        seed,
        split,
        sub_model,
        timeout_s,
        write_artifacts,
    )


@app.cell
def __(controls, mo):
    mo.md("## Controls")
    controls
    return


@app.cell
def __(
    AXON_BIN,
    MODE_PRESETS,
    PlancraftGymWrapper,
    api_key_env,
    base_url,
    get_plancraft_examples,
    max_steps,
    model,
    modes,
    n_examples,
    os,
    parse_stderr_metrics,
    random,
    run_btn,
    split,
    sub_model,
    subprocess,
    tempfile,
    time,
):
    def build_action_query() -> str:
        return (
            "You are controlling a Plancraft agent. "
            "Read the observation from context and output exactly one action line. "
            "Allowed formats:\n"
            "move: from [Source] to [Target] with quantity N\n"
            "smelt: from [Source] to [Target] with quantity N\n"
            "impossible: <reason>\n"
            "Do not output code blocks or extra text."
        )

    def run_axon_step(mode_name: str, observation_text: str, timeout_sec: int) -> dict:
        api_key = os.environ.get(api_key_env.value, "")
        if not api_key:
            return {
                "ok": False,
                "answer": "",
                "error": f"Missing API key in env var {api_key_env.value}",
                "elapsed_s": 0.0,
                "metrics": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(observation_text)
            context_path = f.name

        cmd = [
            AXON_BIN,
            "--base-url",
            base_url.value,
            "--api-key",
            api_key,
            "--model",
            model.value.strip(),
            "--sub-model",
            sub_model.value.strip(),
            *MODE_PRESETS[mode_name],
            "query",
            build_action_query(),
            "--context",
            context_path,
        ]

        start = time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
            elapsed = time.monotonic() - start
            stderr = proc.stderr or ""
            metrics = parse_stderr_metrics(stderr)
            if proc.returncode != 0:
                return {
                    "ok": False,
                    "answer": "",
                    "error": (stderr.strip() or (proc.stdout or "").strip() or f"exit {proc.returncode}")[:1200],
                    "elapsed_s": elapsed,
                    "metrics": metrics,
                }
            return {
                "ok": True,
                "answer": (proc.stdout or "").strip(),
                "error": "",
                "elapsed_s": elapsed,
                "metrics": metrics,
            }
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return {
                "ok": False,
                "answer": "",
                "error": f"timeout after {timeout_sec}s",
                "elapsed_s": elapsed,
                "metrics": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        finally:
            try:
                os.unlink(context_path)
            except OSError:
                pass

    examples = None
    rows = None
    if run_btn.value:
        all_examples = get_plancraft_examples(split=split.value)
        rng = random.Random(int(seed.value))
        sample_n = min(int(n_examples.value), len(all_examples))
        examples = rng.sample(all_examples, sample_n)

        rows = []
        for mode_name in modes.value:
            for ex in examples:
                env = PlancraftGymWrapper(example=ex, max_steps=int(max_steps.value), use_text_inventory=True)
                obs, reward, terminated, truncated, info = env.step("")

                total_elapsed = 0.0
                total_calls = 0
                total_tokens = 0
                step_errors = 0
                steps_taken = 0

                while not terminated and not truncated:
                    step_result = run_axon_step(mode_name, obs.get("text", ""), int(timeout_s.value))
                    total_elapsed += step_result["elapsed_s"]
                    total_calls += step_result["metrics"]["llm_calls"]
                    total_tokens += step_result["metrics"]["total_tokens"]
                    if not step_result["ok"]:
                        step_errors += 1

                    action_text = step_result["answer"] if step_result["ok"] else "invalid"
                    obs, reward, terminated, truncated, info = env.step(action_text)
                    steps_taken += 1

                rows.append(
                    {
                        "mode": mode_name,
                        "example_id": ex.id,
                        "target": ex.target,
                        "impossible": bool(ex.impossible),
                        "success": bool(env.success),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "steps": int(steps_taken),
                        "step_errors": int(step_errors),
                        "elapsed_s": round(total_elapsed, 3),
                        "llm_calls": int(total_calls),
                        "total_tokens": int(total_tokens),
                    }
                )

    return examples, rows


@app.cell
def __(mo, rows):
    if rows is None:
        _ = mo.md("Run the benchmark to generate results.")
    return


@app.cell
def __(pd, rows):
    df = None
    summary = None
    if rows is not None:
        df = pd.DataFrame(rows)
        summary = (
            df.groupby("mode", as_index=False)
            .agg(
                tasks=("example_id", "count"),
                pass_rate=("success", "mean"),
                avg_steps=("steps", "mean"),
                avg_errors=("step_errors", "mean"),
                avg_time_s=("elapsed_s", "mean"),
                avg_tokens=("total_tokens", "mean"),
                avg_calls=("llm_calls", "mean"),
            )
        )
        summary["pass_rate"] = (summary["pass_rate"] * 100.0).round(2)
        summary["avg_steps"] = summary["avg_steps"].round(2)
        summary["avg_errors"] = summary["avg_errors"].round(2)
        summary["avg_time_s"] = summary["avg_time_s"].round(2)
        summary["avg_tokens"] = summary["avg_tokens"].round(1)
        summary["avg_calls"] = summary["avg_calls"].round(2)
    return df, summary


@app.cell
def __(mo, summary):
    if summary is not None:
        _ = mo.md("## Mode Summary")
        _ = mo.ui.table(summary)
    return


@app.cell
def __(df, plt):
    if df is not None and not df.empty:
        mode_pass = df.groupby("mode")["success"].mean().sort_values(ascending=False) * 100.0
        _fig, _ax = plt.subplots(figsize=(8, 4.5))
        _ax.bar(mode_pass.index, mode_pass.values)
        _ax.set_ylabel("Pass rate (%)")
        _ax.set_title("PlanCraft success by mode")
        _ax.set_ylim(0, 100)
        _ax.grid(axis="y", alpha=0.3)
        _ax.tick_params(axis="x", rotation=20)
        _fig.tight_layout()
        _ = _fig
    return


@app.cell
def __(df, mo):
    if df is not None:
        _ = mo.md("## Per-example Results")
        _ = mo.ui.table(df)
    return


@app.cell
def __(
    Path,
    artifact_root,
    base_url,
    json,
    max_steps,
    model,
    modes,
    mo,
    n_examples,
    pd,
    rows,
    run_btn,
    seed,
    split,
    sub_model,
    summary,
    timeout_s,
    time,
    write_artifacts,
):
    output_path = None
    if run_btn.value and rows is not None and summary is not None and write_artifacts.value:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = Path(artifact_root.value).resolve() / f"plancraft-{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        _df = pd.DataFrame(rows)
        payload = {
            "created_at": int(time.time()),
            "benchmark": "plancraft",
            "split": split.value,
            "n_examples": int(n_examples.value),
            "seed": int(seed.value),
            "modes": list(modes.value),
            "base_url": base_url.value,
            "model": model.value.strip(),
            "sub_model": sub_model.value.strip(),
            "timeout_s": int(timeout_s.value),
            "max_steps": int(max_steps.value),
            "summary": summary.to_dict(orient="records"),
            "results": rows,
        }

        (out_dir / "results-plancraft.json").write_text(json.dumps(payload, indent=2))
        _df.to_csv(out_dir / "results-plancraft.csv", index=False)
        summary.to_csv(out_dir / "summary-plancraft.csv", index=False)

        lines = [
            "| Mode | Tasks | Pass rate | Avg steps | Avg errors | Avg time (s) | Avg tokens | Avg calls |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['mode']} | {int(row['tasks'])} | {float(row['pass_rate']):.2f}% | {float(row['avg_steps']):.2f} | {float(row['avg_errors']):.2f} | {float(row['avg_time_s']):.2f} | {float(row['avg_tokens']):.1f} | {float(row['avg_calls']):.2f} |"
            )
        (out_dir / "summary-plancraft.md").write_text("\n".join(lines) + "\n")

        _ = mo.md(f"PlanCraft artifacts written to `{out_dir}`")
        output_path = str(out_dir)
    return output_path


@app.cell
def __(mo):
    _ = mo.md(
        """
        Notes:
        - This notebook uses PlanCraft text observations (`use_text_inventory=True`).
        - `success=True` means the environment judged the episode successful.
        - For impossible tasks, success requires a valid `impossible: <reason>` action.
        """
    )
    return


if __name__ == "__main__":
    app.run()
