#!/usr/bin/env python3
import argparse
import glob
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


FORMAT_LEAK_MARKERS = ("```repl", "let me ", "<tool_call", "<function")


@dataclass
class Row:
    source: str
    dataset: str
    model: str
    mode: str
    policy_profile: str
    pass_ok: bool
    matched: bool
    elapsed_s: float
    cost_usd: float
    answer: str


def resolve_paths(inputs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob("results-*.json")))
        elif any(ch in raw for ch in "*?[]"):
            out.extend(sorted(Path(x) for x in glob.glob(raw)))
        else:
            out.append(p)
    uniq = []
    seen = set()
    for p in out:
        rp = p.resolve()
        if rp in seen or not rp.exists():
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


def has_format_leak(answer: str) -> bool:
    low = (answer or "").lower()
    return any(m in low for m in FORMAT_LEAK_MARKERS)


def load_rows(path: Path) -> List[Row]:
    payload = json.loads(path.read_text())
    dataset = Path(payload.get("dataset", path.name)).name
    profile = str(payload.get("policy_profile") or "baseline")
    rows: List[Row] = []
    for r in payload.get("results", []):
        rows.append(
            Row(
                source=path.name,
                dataset=dataset,
                model=str(r.get("model") or ""),
                mode=str(r.get("mode") or ""),
                policy_profile=profile,
                pass_ok=bool(r.get("ok", False)),
                matched=bool(r.get("matched", False)),
                elapsed_s=float(r.get("elapsed_s", 0.0)),
                cost_usd=float(r.get("estimated_cost_usd", 0.0)),
                answer=str(r.get("answer", "")),
            )
        )
    return rows


def summarize(rows: List[Row], alpha: float, beta: float) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str], List[Row]] = defaultdict(list)
    for r in rows:
        by_key[(r.policy_profile, r.model, r.mode)].append(r)

    summary: List[Dict[str, Any]] = []
    for (profile, model, mode), group in by_key.items():
        n = len(group)
        matched = sum(1 for r in group if r.matched)
        format_leaks = sum(
            1 for r in group if r.pass_ok and not r.matched and has_format_leak(r.answer)
        )
        avg_cost = (sum(r.cost_usd for r in group) / n) if n else 0.0
        avg_time = (sum(r.elapsed_s for r in group) / n) if n else 0.0
        pass_rate = (matched / n) if n else 0.0
        leak_rate = (format_leaks / n) if n else 0.0
        objective = pass_rate - (alpha * leak_rate) - (beta * avg_cost)
        summary.append(
            {
                "policy_profile": profile,
                "model": model,
                "mode": mode,
                "n": n,
                "pass_rate": pass_rate * 100.0,
                "format_leak_rate": leak_rate * 100.0,
                "avg_cost_usd": avg_cost,
                "avg_time_s": avg_time,
                "objective": objective,
            }
        )
    summary.sort(key=lambda x: x["objective"], reverse=True)
    return summary


def write_markdown(summary: List[Dict[str, Any]], out: Path, alpha: float, beta: float) -> None:
    lines = [
        "# Prompt Policy Optimization Report",
        "",
        f"Objective: pass_rate - ({alpha} * format_leak_rate) - ({beta} * avg_cost_usd)",
        "",
        "| Rank | Policy | Model | Mode | N | Pass % | Format leak % | Avg cost | Avg time (s) | Objective |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, s in enumerate(summary, 1):
        model = s["model"] or "(default)"
        lines.append(
            f"| {idx} | {s['policy_profile']} | {model} | {s['mode']} | {s['n']} | "
            f"{s['pass_rate']:.2f} | {s['format_leak_rate']:.2f} | {s['avg_cost_usd']:.6f} | "
            f"{s['avg_time_s']:.3f} | {s['objective']:.4f} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def write_catalog_patch(
    summary: List[Dict[str, Any]], out: Path, top_k: int, min_samples: int
) -> None:
    best = [s for s in summary if s["n"] >= min_samples][:top_k]
    payload = {
        "generated_from": "optimize_prompt_policy.py",
        "selection_rule": {
            "top_k": top_k,
            "min_samples": min_samples,
        },
        "recommended_profiles": best,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline prompt-policy optimizer from benchmark results."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Results JSON file(s), directories, or glob patterns.",
    )
    parser.add_argument("--alpha", type=float, default=0.30, help="Format-leak penalty weight")
    parser.add_argument("--beta", type=float, default=5.0, help="Cost penalty weight")
    parser.add_argument(
        "--out-md",
        default="benchmarks/analysis/policy_optimization_report.md",
        help="Markdown report output path",
    )
    parser.add_argument(
        "--out-json",
        default="benchmarks/analysis/policy_optimization_selection.json",
        help="JSON recommendation output path",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=8)
    args = parser.parse_args()

    files = resolve_paths(args.inputs)
    if not files:
        raise SystemExit("No results files found.")

    rows: List[Row] = []
    for p in files:
        rows.extend(load_rows(p))
    if not rows:
        raise SystemExit("No rows found in selected inputs.")

    summary = summarize(rows, alpha=args.alpha, beta=args.beta)
    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    write_markdown(summary, out_md, alpha=args.alpha, beta=args.beta)
    write_catalog_patch(summary, out_json, top_k=args.top_k, min_samples=args.min_samples)

    print(f"Analyzed {len(files)} file(s), {len(rows)} rows.")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
