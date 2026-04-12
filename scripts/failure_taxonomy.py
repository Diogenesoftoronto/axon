#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TRANSIENT_ERROR_MARKERS = [
    "reqwest error: error sending request for url",
    "connection reset",
    "connection refused",
    "temporarily unavailable",
    "service unavailable",
    "gateway timeout",
    "status code: 429",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "http 429",
    "http 502",
    "http 503",
    "http 504",
    "timeout after",
]

PLACEHOLDER_ANSWERS = {
    "final_answer",
    "answer",
    "todo",
    "n/a",
    "na",
    "none",
    "null",
}


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def contains_token(text: str, token: str) -> bool:
    if not token:
        return False
    return re.search(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE) is not None


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?", text)


def lines_in_order(answer: str, expected_lines: List[str]) -> bool:
    if not expected_lines:
        return False
    hay = answer.lower()
    pos = 0
    for line in expected_lines:
        needle = line.strip().lower()
        if not needle:
            continue
        idx = hay.find(needle, pos)
        if idx < 0:
            return False
        pos = idx + len(needle)
    return True


def is_repl_or_tool_trace(answer: str) -> bool:
    lower = answer.lower()
    markers = [
        "```repl",
        "<tool_call",
        "<function",
        "analysis:",
        "let me solve this",
    ]
    return any(m in lower for m in markers)


def classify_runtime_error(error: str) -> Tuple[str, str]:
    e = normalize_text(error)
    if not e:
        return "runtime_nonzero_exit", "non-zero exit without stderr detail"
    if "timeout after" in e:
        return "runtime_timeout", "subprocess timeout"
    if "missing api key" in e or "unauthorized" in e or "status code: 401" in e:
        return "runtime_auth_or_config", "authentication/configuration error"
    if any(marker in e for marker in TRANSIENT_ERROR_MARKERS):
        return "runtime_provider_or_network", "provider/network transient failure"
    return "runtime_nonzero_exit", "non-zero exit"


def classify_semantic_failure(
    answer: str, check: Dict[str, Any], is_hallucination_probe: bool
) -> Tuple[str, str]:
    ctype = str(check.get("type", "unknown"))
    expected = str(check.get("value", ""))

    answer_norm = normalize_text(answer)
    expected_norm = normalize_text(expected)

    if not answer_norm:
        return "empty_output", "empty answer"
    if answer_norm in PLACEHOLDER_ANSWERS:
        return "placeholder_output", "placeholder token instead of answer"
    if is_repl_or_tool_trace(answer):
        return "format_trace_or_scratchpad_leak", "returned scratchpad/tool trace"

    if ctype == "exact":
        if expected_norm and expected_norm in answer_norm and answer_norm != expected_norm:
            if expected_norm == "insufficient_information":
                return (
                    "abstention_format_extra_text",
                    "contains abstention token plus extra text",
                )
            return "format_extra_text_exact", "contains expected value plus extra text"
        if expected_norm == "insufficient_information" and expected_norm not in answer_norm:
            return "hallucinated_when_should_abstain", "did not output required abstention"
        return "wrong_exact_value", "exact answer mismatch"

    if ctype == "number_exact":
        exp_nums = extract_numbers(expected)
        ans_nums = extract_numbers(answer)
        if exp_nums and ans_nums and exp_nums[0] in ans_nums:
            return "numeric_format_issue", "expected number present but check still failed"
        if not ans_nums:
            return "missing_numeric_answer", "no numeric token in answer"
        return "wrong_numeric_value", "numeric answer mismatch"

    if ctype == "choice_exact":
        choice = expected.strip()
        if choice and contains_token(answer, choice):
            return "choice_format_issue", "expected choice token present but malformed"
        return "wrong_choice", "incorrect choice token"

    if ctype == "yesno_exact":
        has_yes = contains_token(answer, "yes")
        has_no = contains_token(answer, "no")
        if has_yes and has_no:
            return "ambiguous_yesno", "contains both yes and no"
        if not (has_yes or has_no):
            return "missing_yesno_token", "no yes/no token found"
        return "wrong_yesno_value", "yes/no answer mismatch"

    if ctype == "lines_in_order":
        values = check.get("values")
        if isinstance(values, list) and values:
            lines = [str(v) for v in values if str(v).strip()]
        else:
            lines = [ln for ln in expected.splitlines() if ln.strip()]
        if lines:
            found = [ln for ln in lines if ln.strip().lower() in answer.lower()]
            if len(found) == len(lines):
                if not lines_in_order(answer, lines):
                    return "lines_wrong_order", "all required lines present in wrong order"
                return "lines_format_mismatch", "line content present but format mismatch"
            if found:
                return "lines_partial_missing", "some required lines missing"
        return "lines_missing", "required lines missing"

    if ctype == "contains":
        return "missing_required_substring", "required substring not present"

    if ctype == "regex":
        return "regex_mismatch", "regex check did not match"

    if ctype == "rust_exec_exact":
        low = answer.lower()
        if "```rust" not in low and "fn main" not in low:
            langs = ("```python", "```cpp", "```java", "```go", "```javascript")
            if any(lang in low for lang in langs):
                return "rust_wrong_language", "returned non-Rust code block"
            return "rust_missing_code", "missing compilable Rust answer"
        return "rust_exec_mismatch", "Rust code did not produce expected output"

    if is_hallucination_probe:
        return "hallucination_guardrail_failure", "failed guardrail probe check"
    return "unclassified_semantic_failure", f"unknown check type: {ctype}"


def resolve_paths(inputs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob("results-*.json")))
            continue
        if any(ch in raw for ch in "*?[]"):
            out.extend(sorted(Path(match) for match in glob.glob(raw)))
            continue
        out.append(p)
    seen = set()
    deduped: List[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(rp)
    return [p for p in deduped if p.exists()]


def resolve_dataset_path(dataset_ref: str, result_path: Path) -> Optional[Path]:
    if not dataset_ref:
        return None
    candidate = Path(dataset_ref)
    tries = [
        candidate,
        (result_path.parent / candidate),
        (Path.cwd() / candidate),
    ]
    for path in tries:
        if path.exists():
            return path.resolve()
    return None


def load_task_map(dataset_ref: str, result_path: Path) -> Dict[str, Dict[str, Any]]:
    dataset_path = resolve_dataset_path(dataset_ref, result_path)
    if dataset_path is None:
        return {}
    try:
        data = json.loads(dataset_path.read_text())
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in data:
        tid = str(row.get("id", "")).strip()
        if tid:
            out[tid] = row
    return out


def preview(s: str, limit: int = 180) -> str:
    s = (s or "").replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    headers = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def top_categories_for_subset(rows: List[Dict[str, Any]], limit: int = 3) -> str:
    counts = Counter(r["category"] for r in rows)
    if not counts:
        return "-"
    return ", ".join(f"{c} ({n})" for c, n in counts.most_common(limit))


def render_markdown_report(
    rows: List[Dict[str, Any]],
    result_files: List[Path],
    output_path: Path,
    examples_per_category: int = 3,
) -> None:
    total_failures = len(rows)
    category_counts = Counter(r["category"] for r in rows)
    mode_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    dataset_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    category_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        mode_groups[row["mode"]].append(row)
        dataset_groups[row["dataset"]].append(row)
        category_examples[row["category"]].append(row)

    lines: List[str] = []
    lines.append("# Failure Taxonomy")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    lines.append(f"Result files: {len(result_files)}")
    lines.append(f"Total failures: {total_failures}")
    lines.append("")

    lines.append("## Category Summary")
    lines.append("")
    lines.append("| Category | Count | Share |")
    lines.append("|---|---:|---:|")
    for category, count in category_counts.most_common():
        share = (count / total_failures * 100.0) if total_failures else 0.0
        lines.append(f"| {category} | {count} | {share:.2f}% |")
    lines.append("")

    lines.append("## Failures by Mode")
    lines.append("")
    lines.append("| Mode | Failures | Top Categories |")
    lines.append("|---|---:|---|")
    for mode, mode_rows in sorted(mode_groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        lines.append(
            f"| {mode} | {len(mode_rows)} | {top_categories_for_subset(mode_rows)} |"
        )
    lines.append("")

    lines.append("## Failures by Dataset")
    lines.append("")
    lines.append("| Dataset | Failures | Top Categories |")
    lines.append("|---|---:|---|")
    for dataset, ds_rows in sorted(
        dataset_groups.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        lines.append(
            f"| {dataset} | {len(ds_rows)} | {top_categories_for_subset(ds_rows)} |"
        )
    lines.append("")

    lines.append("## Example Failures")
    lines.append("")
    for category, _count in category_counts.most_common():
        lines.append(f"### {category}")
        for row in category_examples[category][:examples_per_category]:
            lines.append(
                f"- `{row['dataset']}` | `{row['mode']}` | `{row['task_id']}` | `{row['check_type']}`: {row['detail']}"
            )
            if row["answer_preview"]:
                lines.append(f"  answer: `{row['answer_preview']}`")
            if row["error_preview"]:
                lines.append(f"  error: `{row['error_preview']}`")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n")


def analyze_result_file(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    dataset_ref = str(payload.get("dataset", ""))
    dataset_name = Path(dataset_ref).name if dataset_ref else path.name
    task_map = load_task_map(dataset_ref, path)
    rows: List[Dict[str, Any]] = []

    for row in payload.get("results", []):
        ok = bool(row.get("ok", False))
        matched = bool(row.get("matched", False))
        if ok and matched:
            continue

        task_id = str(row.get("task_id", ""))
        task = task_map.get(task_id, {})
        check = task.get("check", {}) if isinstance(task, dict) else {}
        check_type = str(check.get("type", "unknown"))
        answer = str(row.get("answer", ""))
        error = str(row.get("error", ""))
        is_probe = bool(row.get("is_hallucination_probe", False))

        if not ok:
            category, detail = classify_runtime_error(error)
        else:
            category, detail = classify_semantic_failure(answer, check, is_probe)

        rows.append(
            {
                "source_file": path.name,
                "dataset": dataset_name,
                "mode": str(row.get("mode", "")),
                "model": str(row.get("model", "")),
                "task_id": task_id,
                "check_type": check_type,
                "ok": ok,
                "matched": matched,
                "category": category,
                "detail": detail,
                "answer_preview": preview(answer),
                "error_preview": preview(error),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a failure taxonomy from Axon benchmark results JSON files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Result JSON files, directories, or globs (for example benchmarks/overall-*/results-*.json).",
    )
    parser.add_argument(
        "--out-md",
        default="benchmarks/analysis/failure_taxonomy.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--out-csv",
        default="benchmarks/analysis/failure_taxonomy_rows.csv",
        help="Flat CSV output path.",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=3,
        help="Number of example rows per category in the markdown report.",
    )
    args = parser.parse_args()

    result_files = resolve_paths(args.inputs)
    if not result_files:
        raise SystemExit("No result files found from inputs.")

    rows: List[Dict[str, Any]] = []
    for p in result_files:
        rows.extend(analyze_result_file(p))

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    write_csv(rows, out_csv)
    render_markdown_report(rows, result_files, out_md, args.examples_per_category)

    total = len(rows)
    by_cat = Counter(r["category"] for r in rows)
    print(f"Analyzed {len(result_files)} result file(s), {total} failure row(s).")
    for category, count in by_cat.most_common():
        share = (count / total * 100.0) if total else 0.0
        print(f"- {category}: {count} ({share:.2f}%)")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
