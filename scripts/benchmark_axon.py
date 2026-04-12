#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.request import Request, urlopen


@dataclass
class Mode:
    name: str
    binary: str
    extra_args: List[str]


@dataclass
class RunResult:
    mode: str
    model: str
    sub_model: str
    task_id: str
    run_idx: int
    ok: bool
    matched: bool
    elapsed_s: float
    context_chars: int
    query_chars: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_chars_logged: int
    response_chars_logged: int
    estimated_cost_usd: float
    is_hallucination_probe: bool
    answer: str
    error: str


def mode_max_depth(mode: Mode) -> Optional[int]:
    args = mode.extra_args
    for i, token in enumerate(args):
        if token == "--max-depth" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return None
    return None


USAGE_RE = re.compile(
    r"usage: prompt_tokens=(?P<prompt>\d+) completion_tokens=(?P<completion>\d+) total_tokens=(?P<total>\d+) prompt_chars=(?P<prompt_chars>\d+)"
)
RESP_RE = re.compile(r"response: (?P<chars>\d+) chars")


def parse_stderr_metrics(stderr: str) -> Dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    prompt_chars = 0
    response_chars = 0
    llm_calls = 0

    for line in stderr.splitlines():
        m = USAGE_RE.search(line)
        if m:
            llm_calls += 1
            prompt_tokens += int(m.group("prompt"))
            completion_tokens += int(m.group("completion"))
            total_tokens += int(m.group("total"))
            prompt_chars += int(m.group("prompt_chars"))
        m2 = RESP_RE.search(line)
        if m2:
            response_chars += int(m2.group("chars"))

    return {
        "llm_calls": llm_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_chars_logged": prompt_chars,
        "response_chars_logged": response_chars,
    }


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Task dataset must be a JSON list")
    return data


def load_model_list(path: Path) -> List[str]:
    models: List[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        models.append(s)
    return models


def parse_money_per_token(s: str) -> Optional[float]:
    # Synthetic /models returns strings like "$0.0000006".
    if not s:
        return None
    s = s.strip()
    if s.startswith("$"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return None


def fetch_openai_models_pricing(base_url: str, api_key: str) -> Dict[str, Dict[str, float]]:
    url = base_url.rstrip("/") + "/models"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out: Dict[str, Dict[str, float]] = {}
    for row in data.get("data", []):
        mid = row.get("id")
        pricing = row.get("pricing", {}) or {}
        p = parse_money_per_token(pricing.get("prompt", ""))
        c = parse_money_per_token(pricing.get("completion", ""))
        if mid and p is not None and c is not None:
            out[mid] = {"prompt_per_token": p, "completion_per_token": c}
    return out


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (
        z
        * (((p * (1.0 - p)) / total + (z * z) / (4.0 * total * total)) ** 0.5)
        / denom
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return {"low": low * 100.0, "high": high * 100.0}


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def strip_final_wrapper(text: str) -> str:
    """Remove FINAL(...) or FINAL_VAR(...) wrapper from answer text."""
    text = text.strip()
    for prefix in ("FINAL_VAR(", "FINAL("):
        idx = text.find(prefix)
        if idx >= 0:
            start = idx + len(prefix)
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                i += 1
            if depth == 0:
                return text[start:i-1].strip()
    return text


def is_transient_error(error: str) -> bool:
    e = (error or "").lower()
    markers = [
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
    return any(m in e for m in markers)


def extract_rust_code(answer: str) -> Optional[str]:
    m = re.search(r"```rust\s*(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:rs)?\s*(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if "fn main" in answer:
        return answer.strip()
    return None


def run_rust_program(code: str, timeout_s: int = 10) -> Optional[str]:
    with tempfile.TemporaryDirectory(prefix="axon-rust-check-") as td:
        tdp = Path(td)
        src = tdp / "main.rs"
        binp = tdp / "prog"
        src.write_text(code)
        try:
            comp = subprocess.run(
                ["rustc", str(src), "-O", "-o", str(binp)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if comp.returncode != 0:
                return None
            run = subprocess.run(
                [str(binp)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if run.returncode != 0:
                return None
            return run.stdout.strip()
        except Exception:
            return None


def _contains_token(text: str, token: str) -> bool:
    if not token:
        return False
    return re.search(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE) is not None

def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?", text)

def _lines_in_order(answer: str, expected_lines: List[str]) -> bool:
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

def evaluate(answer: str, check: Dict[str, Any]) -> bool:
    ctype = check.get("type", "exact")
    value = check.get("value", "")
    a_norm = normalize_text(answer)
    v_norm = normalize_text(value)

    if ctype == "exact":
        return a_norm == v_norm
    if ctype == "contains":
        return v_norm in a_norm
    if ctype == "regex":
        return re.search(value, answer, flags=re.IGNORECASE | re.DOTALL) is not None
    if ctype == "number_exact":
        exp_nums = _extract_numbers(str(value))
        ans_nums = _extract_numbers(answer)
        if not exp_nums or not ans_nums:
            return False
        exp = exp_nums[0]
        if exp in ans_nums:
            return True
        try:
            exp_f = float(exp)
            return any(abs(float(n) - exp_f) < 1e-9 for n in ans_nums)
        except ValueError:
            return False
    if ctype == "choice_exact":
        expected = str(value).strip()
        if not expected:
            return False
        return a_norm == normalize_text(expected) or _contains_token(answer, expected)
    if ctype == "yesno_exact":
        expected = normalize_text(str(value))
        if expected not in {"yes", "no"}:
            return False
        return _contains_token(answer, expected)
    if ctype == "lines_in_order":
        values = check.get("values")
        if isinstance(values, list) and values:
            lines = [str(v) for v in values]
        else:
            lines = [ln for ln in str(value).splitlines() if ln.strip()]
        if a_norm == normalize_text("\n".join(lines)):
            return True
        return _lines_in_order(answer, lines)
    if ctype == "rust_exec_exact":
        code = extract_rust_code(answer)
        if not code:
            return False
        out = run_rust_program(code)
        if out is None:
            return False
        return normalize_text(out) == v_norm
    raise ValueError(f"Unknown check type: {ctype}")


def generate_context_from_spec(spec: Dict[str, Any]) -> str:
    gtype = spec.get("type")
    if gtype == "log_haystack_v1":
        size_chars = int(spec["size_chars"])
        filler = spec.get("filler", "INFO lorem ipsum ")
        insertions = spec.get("insertions", [])
        if isinstance(insertions, dict):
            insertions = [insertions]
        marker = spec.get("marker")
        marker_count = int(spec.get("marker_count", 0))
        marker_template = spec.get("marker_template", "{marker}\n")
        if marker and marker_count > 0:
            marker_text = marker_template.replace("{marker}", marker)
            # Reserve space and add evenly distributed marker insertions.
            marker_insertions = []
            for i in range(marker_count):
                frac = (i + 1) / (marker_count + 1)
                marker_insertions.append(
                    {"at": int(size_chars * frac), "text": marker_text}
                )
            insertions = list(insertions) + marker_insertions

        total_ins = sum(len(ins.get("text", "")) for ins in insertions)
        base_len = max(0, size_chars - total_ins)
        if not filler:
            filler = "x"
        base = (filler * ((base_len // len(filler)) + 1))[:base_len]

        def resolve_at(ins: Dict[str, Any]) -> int:
            at = ins.get("at")
            if isinstance(at, int):
                return max(0, min(base_len, at))
            pos = (ins.get("pos") or "").lower()
            if pos == "start":
                return 0
            if pos == "middle":
                return base_len // 2
            if pos == "end":
                return base_len
            raise ValueError(f"Invalid insertion spec: {ins}")

        resolved = []
        for ins in insertions:
            text = ins.get("text", "")
            if not isinstance(text, str):
                raise ValueError(f"Invalid insertion text: {ins}")
            resolved.append((resolve_at(ins), text))
        resolved.sort(key=lambda x: x[0], reverse=True)

        out = base
        for at, text in resolved:
            out = out[:at] + text + out[at:]
        return out

    if gtype == "ledger_v1":
        n_rows = int(spec.get("n_rows", 200))
        seed = int(spec.get("seed", 7))
        regions = ["US", "EU", "APAC", "LATAM"]
        lines = []
        for i in range(1, n_rows + 1):
            day = ((i - 1) % 28) + 1
            txn_id = 100000 + i
            txn_type = "Debit" if ((i * 7 + seed) % 2 == 0) else "Credit"
            amount_cents = ((i * 37 + seed * 13) % 9900) + 100
            amount = f"{amount_cents / 100.0:.2f}"
            region = regions[(i + seed) % len(regions)]
            lines.append(
                f"2025-01-{day:02d},ID={txn_id},TYPE={txn_type},AMT={amount},REGION={region}"
            )
        return "\n".join(lines)

    raise ValueError(f"Unknown context_gen type: {gtype}")


def build_context(task: Dict[str, Any]) -> str:
    if "context_gen" in task:
        return generate_context_from_spec(task["context_gen"])
    return task.get("context", "")


RUST_ONLY_NEGATIVE_EXAMPLES = [
    ("python", "print(answer)"),
    ("cpp", "#include <bits/stdc++.h>\nint main(){std::cout << answer << '\\n';}"),
    ("java", "class Main { public static void main(String[] args){ System.out.println(answer); } }"),
    ("go", "package main\nimport \"fmt\"\nfunc main(){ fmt.Println(answer) }"),
    ("javascript", "console.log(answer);"),
]

RECURSIVE_MODES = {"d1-i3", "d3-i1", "d6-i1"}


def _recursive_strategy_suffix(mode_name: str) -> str:
    if mode_name not in RECURSIVE_MODES:
        return ""
    return (
        "DEPTH-AWARE RECURSIVE STRATEGY (required in this mode):\n"
        "1) Decompose the task into 2-4 concrete subproblems.\n"
        "2) Use recursive delegation for independent or high-complexity subproblems.\n"
        "3) Keep each sub-result short and machine-checkable (numbers/tokens/ordered fields).\n"
        "4) Synthesize sub-results and run one consistency check before finalizing.\n"
        "5) Emit a single canonical final answer only after verification.\n"
        "Use this mini-template internally:\n"
        "- Subproblem 1 -> Subresult 1\n"
        "- Subproblem 2 -> Subresult 2\n"
        "- Verification -> final canonical answer\n"
    )


def _non_rust_output_contract(check: Dict[str, Any]) -> str:
    ctype = str(check.get("type", "exact"))
    hint = "Inside FINAL(...), include only the canonical answer text with no extra prose."
    if ctype == "number_exact":
        hint = "Inside FINAL(...), include only the numeric answer (no words, no units)."
    elif ctype == "choice_exact":
        hint = "Inside FINAL(...), include only the chosen option token/index."
    elif ctype == "yesno_exact":
        hint = "Inside FINAL(...), include only yes or no."
    elif ctype == "lines_in_order":
        hint = "Inside FINAL(...), include only the required lines in the required order."
    elif ctype == "regex":
        hint = "Inside FINAL(...), include only a compact answer matching the required pattern."
    elif ctype == "contains":
        hint = "Inside FINAL(...), include a minimal answer that contains the required target text."

    return (
        "STRICT FINAL OUTPUT CONTRACT (must follow exactly):\n"
        "Any text outside FINAL(...) is an automatic failure.\n"
        "Evaluator gate: output must match regex ^FINAL\\(.*\\)$ on a single line.\n"
        "1) Do reasoning in scratch steps, but do not leave that in the final output.\n"
        "2) Final output must be exactly one line in this form: FINAL(<answer>).\n"
        "3) Do not include markdown fences (especially ```repl), XML/tool-call tags, or explanation outside FINAL(...).\n"
        f"4) {hint}\n"
        "5) Never emit pseudo-tool output or REPL snippets.\n"
        "Formatting examples:\n"
        "- Invalid: Let me solve this... the answer is 17\n"
        "- Invalid: ```repl ... ```\n"
        "- Invalid: {\"answer\": \"17\"}\n"
        "- Valid: FINAL(17)\n"
    )


def build_task_query(task: Dict[str, Any], mode_name: str) -> str:
    base = str(task.get("query", "")).strip()
    check = task.get("check", {}) or {}
    recursive_suffix = _recursive_strategy_suffix(mode_name)
    if check.get("type") != "rust_exec_exact":
        strict_suffix = _non_rust_output_contract(check)
        return f"{base}\n\n{recursive_suffix}\n{strict_suffix}".strip()

    task_id = str(task.get("id", ""))
    start = sum(ord(ch) for ch in task_id) % len(RUST_ONLY_NEGATIVE_EXAMPLES)
    sample_count = min(3, len(RUST_ONLY_NEGATIVE_EXAMPLES))
    picks = [
        RUST_ONLY_NEGATIVE_EXAMPLES[(start + i) % len(RUST_ONLY_NEGATIVE_EXAMPLES)]
        for i in range(sample_count)
    ]
    invalid_samples = "\n\n".join(
        f"Invalid ({lang}) submission example:\n```{lang}\n{snippet}\n```"
        for lang, snippet in picks
    )

    strict_suffix = (
        "STRICT OUTPUT CONTRACT (must follow exactly):\n"
        "1) Output exactly one fenced code block tagged rust: ```rust ... ```.\n"
        "2) Do not output REPL blocks, prose, markdown text, or tool-call XML.\n"
        "3) The program must be complete and compilable by rustc without edits.\n"
        "4) Program must print exactly one line with only the final answer.\n"
        "5) Any non-Rust submission is invalid (e.g., Python/C++/Java/Go/JavaScript).\n\n"
        "Examples of invalid non-Rust submissions:\n"
        f"{invalid_samples}\n\n"
        "Return Rust code only."
    )
    return f"{base}\n\n{recursive_suffix}\n{strict_suffix}".strip()


def run_one(
    mode: Mode,
    task: Dict[str, Any],
    base_url: str,
    api_key: str,
    model: Optional[str],
    sub_model: Optional[str],
    timeout_s: int,
    prompt_cost_per_1m: float,
    completion_cost_per_1m: float,
    axon_verbose: bool,
    is_hallucination_probe: bool,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
    policy_profile: Optional[str] = None,
    inject_policy_into_context: bool = False,
    depth_enforcement: Optional[str] = None,
    require_min_depth: Optional[int] = None,
    require_min_recursive_calls: Optional[int] = None,
) -> RunResult:
    context_text = build_context(task)
    task_query = build_task_query(task, mode.name)
    query_chars = len(task_query)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(context_text)
        context_path = f.name

    cmd = [
        mode.binary,
        "--base-url",
        base_url,
        "--api-key",
        api_key,
    ]
    if model:
        cmd += ["--model", model]
    if sub_model:
        cmd += ["--sub-model", sub_model]
    if axon_verbose:
        cmd += ["-v"]
    cmd += mode.extra_args + [
        "query",
        task_query,
        "--context",
        context_path,
    ]
    if mode.name != "previous-default":
        if policy_profile:
            cmd += ["--policy-profile", policy_profile]
        if inject_policy_into_context:
            cmd += ["--inject-policy-into-context"]
        if depth_enforcement:
            cmd += ["--depth-enforcement", depth_enforcement]
        if require_min_depth is not None:
            cmd += ["--require-min-depth", str(require_min_depth)]
        if require_min_recursive_calls is not None:
            cmd += ["--require-min-recursive-calls", str(require_min_recursive_calls)]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed = time.monotonic() - start
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        metrics = parse_stderr_metrics(stderr)
        if pricing and model and model in pricing:
            p = pricing[model]["prompt_per_token"]
            c = pricing[model]["completion_per_token"]
            est_cost = (metrics["prompt_tokens"] * p) + (metrics["completion_tokens"] * c)
        else:
            est_cost = (
                (metrics["prompt_tokens"] / 1_000_000.0) * prompt_cost_per_1m
                + (metrics["completion_tokens"] / 1_000_000.0) * completion_cost_per_1m
            )

        if proc.returncode != 0:
            return RunResult(
                mode=mode.name,
                model=(model or ""),
                sub_model=(sub_model or ""),
                task_id=task["id"],
                run_idx=0,
                ok=False,
                matched=False,
                elapsed_s=elapsed,
                context_chars=len(context_text),
                query_chars=query_chars,
                llm_calls=metrics["llm_calls"],
                prompt_tokens=metrics["prompt_tokens"],
                completion_tokens=metrics["completion_tokens"],
                total_tokens=metrics["total_tokens"],
                prompt_chars_logged=metrics["prompt_chars_logged"],
                response_chars_logged=metrics["response_chars_logged"],
                estimated_cost_usd=round(est_cost, 8),
                is_hallucination_probe=is_hallucination_probe,
                answer="",
                error=(stderr or stdout or f"exit {proc.returncode}")[:2000],
            )

        stdout = strip_final_wrapper(stdout)
        matched = evaluate(stdout, task["check"])
        return RunResult(
            mode=mode.name,
            model=(model or ""),
            sub_model=(sub_model or ""),
            task_id=task["id"],
            run_idx=0,
            ok=True,
            matched=matched,
            elapsed_s=elapsed,
            context_chars=len(context_text),
            query_chars=query_chars,
            llm_calls=metrics["llm_calls"],
            prompt_tokens=metrics["prompt_tokens"],
            completion_tokens=metrics["completion_tokens"],
            total_tokens=metrics["total_tokens"],
            prompt_chars_logged=metrics["prompt_chars_logged"],
            response_chars_logged=metrics["response_chars_logged"],
            estimated_cost_usd=round(est_cost, 8),
            is_hallucination_probe=is_hallucination_probe,
            answer=stdout[:4000],
            error="",
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return RunResult(
            mode=mode.name,
            model=(model or ""),
            sub_model=(sub_model or ""),
            task_id=task["id"],
            run_idx=0,
            ok=False,
            matched=False,
            elapsed_s=elapsed,
            context_chars=len(context_text),
            query_chars=query_chars,
            llm_calls=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_chars_logged=0,
            response_chars_logged=0,
            estimated_cost_usd=0.0,
            is_hallucination_probe=is_hallucination_probe,
            answer="",
            error=f"timeout after {timeout_s}s",
        )
    finally:
        try:
            os.unlink(context_path)
        except OSError:
            pass


def ensure_release_binary(repo_dir: Path) -> Path:
    subprocess.run(["cargo", "build", "--release", "-q"], cwd=repo_dir, check=True)
    bin_path = repo_dir / "target" / "release" / "axon"
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary not found: {bin_path}")
    return bin_path


def build_previous_binary(repo_dir: Path, ref: str) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="axon-prev-"))
    worktree_dir = temp_root / "worktree"
    subprocess.run(["git", "worktree", "add", str(worktree_dir), ref], cwd=repo_dir, check=True)
    try:
        subprocess.run(["cargo", "build", "--release", "-q"], cwd=worktree_dir, check=True)
        prev_bin = worktree_dir / "target" / "release" / "axon"
        if not prev_bin.exists():
            raise FileNotFoundError(f"Previous binary not found: {prev_bin}")
        copied = temp_root / "axon-prev"
        shutil.copy2(prev_bin, copied)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree_dir)], cwd=repo_dir, check=False)
    return copied


def summarize(results: List[RunResult]) -> List[Dict[str, Any]]:
    by_mode: Dict[str, List[RunResult]] = {}
    for r in results:
        key = f"{r.model}||{r.sub_model}||{r.mode}"
        by_mode.setdefault(key, []).append(r)

    summary = []
    for key, rows in sorted(by_mode.items()):
        model, sub_model, mode = key.split("||", 2)
        total = len(rows)
        ok = sum(1 for r in rows if r.ok)
        matched = sum(1 for r in rows if r.matched)
        avg_time = sum(r.elapsed_s for r in rows) / total if total else 0.0
        time_var = (
            sum((r.elapsed_s - avg_time) ** 2 for r in rows) / total if total else 0.0
        )
        avg_tokens = sum(r.total_tokens for r in rows) / total if total else 0.0
        token_var = (
            sum((r.total_tokens - avg_tokens) ** 2 for r in rows) / total if total else 0.0
        )
        total_cost = sum(r.estimated_cost_usd for r in rows)
        total_tokens = sum(r.total_tokens for r in rows)
        total_calls = sum(r.llm_calls for r in rows)
        hallucination_tasks = 0
        hallucinations = 0
        for r in rows:
            if r.is_hallucination_probe:
                hallucination_tasks += 1
                if r.ok and not r.matched:
                    hallucinations += 1
        pass_ci = wilson_interval(matched, total)
        ok_ci = wilson_interval(ok, total)
        summary.append(
            {
                "mode": mode,
                "model": model,
                "sub_model": sub_model,
                "total": total,
                "ok": ok,
                "matched": matched,
                "ok_rate": round((ok / total) * 100.0, 2) if total else 0.0,
                "ok_rate_ci95_low": round(ok_ci["low"], 2),
                "ok_rate_ci95_high": round(ok_ci["high"], 2),
                "pass_rate": round((matched / total) * 100.0, 2) if total else 0.0,
                "pass_rate_ci95_low": round(pass_ci["low"], 2),
                "pass_rate_ci95_high": round(pass_ci["high"], 2),
                "avg_time_s": round(avg_time, 3),
                "std_time_s": round(time_var ** 0.5, 3),
                "total_tokens": total_tokens,
                "avg_tokens_per_task": round(total_tokens / total, 1) if total else 0.0,
                "std_tokens_per_task": round(token_var ** 0.5, 1),
                "llm_calls": total_calls,
                "avg_calls_per_task": round(total_calls / total, 2) if total else 0.0,
                "total_cost_usd": round(total_cost, 6),
                "avg_cost_usd": round(total_cost / total, 6) if total else 0.0,
                "hallucination_tasks": hallucination_tasks,
                "hallucinations": hallucinations,
                "hallucination_rate": round((hallucinations / hallucination_tasks) * 100.0, 2)
                if hallucination_tasks
                else 0.0,
            }
        )

    # Synthetic composite scoring
    min_cost = min((s["avg_cost_usd"] for s in summary if s["avg_cost_usd"] > 0), default=0.0)
    min_time = min((s["avg_time_s"] for s in summary if s["avg_time_s"] > 0), default=0.0)
    min_tokens = min((s["avg_tokens_per_task"] for s in summary if s["avg_tokens_per_task"] > 0), default=0.0)

    for s in summary:
        pass_rate = s["pass_rate"] / 100.0
        halluc_rate = s["hallucination_rate"] / 100.0
        ok_rate = s["ok_rate"] / 100.0

        quality_score = 100.0 * (0.7 * pass_rate + 0.3 * (1.0 - halluc_rate))

        cost_norm = (min_cost / s["avg_cost_usd"]) if (min_cost > 0 and s["avg_cost_usd"] > 0) else 0.0
        time_norm = (min_time / s["avg_time_s"]) if (min_time > 0 and s["avg_time_s"] > 0) else 0.0
        tokens_norm = (
            (min_tokens / s["avg_tokens_per_task"])
            if (min_tokens > 0 and s["avg_tokens_per_task"] > 0)
            else 0.0
        )
        efficiency_score = 100.0 * (0.5 * cost_norm + 0.3 * time_norm + 0.2 * tokens_norm)
        reliability_score = 100.0 * ok_rate

        composite = 0.55 * quality_score + 0.25 * reliability_score + 0.20 * efficiency_score

        if s["hallucination_rate"] > 20.0:
            composite = min(composite, 60.0)
        if s["ok_rate"] < 70.0:
            composite = min(composite, 50.0)

        if s["pass_rate"] < 50.0:
            readiness = "NotProductionReady"
        elif composite >= 85.0:
            readiness = "Production Candidate"
        elif composite >= 70.0:
            readiness = "Promising"
        elif composite >= 50.0:
            readiness = "Experimental"
        else:
            readiness = "Unsafe/Unreliable"

        s["quality_score"] = round(quality_score, 2)
        s["efficiency_score"] = round(efficiency_score, 2)
        s["reliability_score"] = round(reliability_score, 2)
        s["composite_score"] = round(composite, 2)
        s["readiness"] = readiness
    return summary


def print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\nBenchmark summary")
    print("model\tsub_model\tmode\tpass_rate\tok_rate\thalluc_rate\tavg_cost_usd\tcomposite\treadiness")
    for s in summary:
        print(
            f"{s['model']}\t{s['sub_model']}\t{s['mode']}\t{s['pass_rate']}%\t{s['ok_rate']}%\t{s['hallucination_rate']}%\t{s['avg_cost_usd']}\t{s['composite_score']}\t{s['readiness']}"
        )


def write_summary_markdown(summary: List[Dict[str, Any]], path: Path) -> None:
    lines = [
        "| Model | Sub-model | Mode | Pass rate | 95% CI | Avg time (s) | Avg tokens | Avg cost (USD) | Composite | Readiness |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in summary:
        ci = f"[{s['pass_rate_ci95_low']:.2f}, {s['pass_rate_ci95_high']:.2f}]"
        model = s["model"] or "(default)"
        sub_model = s["sub_model"] or "(default)"
        lines.append(
            f"| {model} | {sub_model} | {s['mode']} | {s['pass_rate']:.2f}% | {ci} | {s['avg_time_s']:.3f} | {s['avg_tokens_per_task']:.1f} | {s['avg_cost_usd']:.6f} | {s['composite_score']:.2f} | {s['readiness']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Axon modes on hard reasoning tasks")
    parser.add_argument("--dataset", default="benchmarks/rlm_challenges.json")
    parser.add_argument("--base-url", default="https://api.synthetic.new/openai/v1/")
    parser.add_argument("--api-key-env", default="SYNTHETIC_API_KEY")
    parser.add_argument("--model", default=None)
    parser.add_argument("--sub-model", default=None)
    parser.add_argument("--models", action="append", default=[], help="Model id (repeatable, or comma-separated)")
    parser.add_argument("--model-list", type=Path, default=None, help="File with one model id per line")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--attempts-per-run",
        type=int,
        default=1,
        help="Max attempts per logical run (retries only on transient transport/provider errors)",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=2.0,
        help="Base backoff seconds between transient retries",
    )
    parser.add_argument("--task-id", action="append", default=[], help="Task id filter (repeatable)")
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit number of tasks after filtering")
    parser.add_argument("--prev-ref", default=None, help="Optional git ref to benchmark as previous code")
    parser.add_argument("--prompt-cost-per-1m", type=float, default=0.0, help="USD per 1M prompt tokens")
    parser.add_argument("--completion-cost-per-1m", type=float, default=0.0, help="USD per 1M completion tokens")
    parser.add_argument(
        "--pricing-from-models-api",
        action="store_true",
        help="Fetch /models and compute cost using per-token pricing when available",
    )
    parser.add_argument("--summary-md", default=None, help="Optional markdown summary output path")
    parser.add_argument("--no-axon-verbose", action="store_true", help="Disable -v on axon runs")
    parser.add_argument("--policy-profile", default=None, help="Optional axon policy profile name")
    parser.add_argument(
        "--inject-policy-into-context",
        action="store_true",
        help="Enable axon policy preamble injection into runtime context",
    )
    parser.add_argument(
        "--depth-enforcement",
        choices=["off", "soft", "strict"],
        default=None,
        help="Optional axon depth enforcement mode",
    )
    parser.add_argument("--require-min-depth", type=int, default=None)
    parser.add_argument("--require-min-recursive-calls", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output json file path")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"Missing API key in env var {args.api_key_env}", file=sys.stderr)
        return 2
    if args.attempts_per_run < 1:
        print("--attempts-per-run must be >= 1", file=sys.stderr)
        return 2
    if args.retry_backoff_s < 0:
        print("--retry-backoff-s must be >= 0", file=sys.stderr)
        return 2

    repo_dir = Path(__file__).resolve().parents[1]
    tasks = load_tasks(repo_dir / args.dataset)
    if args.task_id:
        allow = set(args.task_id)
        tasks = [t for t in tasks if t.get("id") in allow]
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        print("No tasks selected", file=sys.stderr)
        return 2

    current_bin = ensure_release_binary(repo_dir)

    models: List[str] = []
    if args.model_list:
        models.extend(load_model_list(args.model_list))
    for item in args.models:
        for part in item.split(","):
            s = part.strip()
            if s:
                models.append(s)
    if not models and args.model:
        models = [args.model]
    if not models:
        # Empty string means: do not pass --model/--sub-model and let Axon defaults apply.
        models = [""]

    pricing = None
    if args.pricing_from_models_api:
        try:
            pricing = fetch_openai_models_pricing(args.base_url, api_key)
        except Exception as e:
            print(f"Warning: failed to fetch /models pricing: {e}", file=sys.stderr)
            pricing = None

    modes: List[Mode] = [
        Mode("d0-i1", str(current_bin), ["--max-depth", "0", "--max-iterations", "1"]),
        Mode("d0-i3", str(current_bin), ["--max-depth", "0", "--max-iterations", "3"]),
        Mode("d1-i3", str(current_bin), ["--max-depth", "1", "--max-iterations", "3"]),
        Mode("d3-i1", str(current_bin), ["--max-depth", "3", "--max-iterations", "1"]),
        Mode("d6-i1", str(current_bin), ["--max-depth", "6", "--max-iterations", "1"]),
    ]

    if args.prev_ref:
        prev_bin = build_previous_binary(repo_dir, args.prev_ref)
        modes.append(Mode("previous-default", str(prev_bin), ["--max-depth", "0", "--max-iterations", "1"]))

    results: List[RunResult] = []

    for model in models:
        effective_sub_model = args.sub_model
        if model and not effective_sub_model:
            # By default, evaluate recursion with the same model for root and sub-calls.
            effective_sub_model = model
        for mode in modes:
            for task in tasks:
                for i in range(args.runs):
                    if (
                        args.depth_enforcement == "strict"
                        and args.require_min_depth is not None
                    ):
                        mdepth = mode_max_depth(mode)
                        if mdepth is not None and args.require_min_depth > mdepth:
                            label = model if model else "(default)"
                            print(
                                f"[{label}][{mode.name}] {task['id']} run={i+1} -> ERROR (strict depth invalid: require_min_depth={args.require_min_depth} > mode max_depth={mdepth})"
                            )
                            results.append(
                                RunResult(
                                    mode=mode.name,
                                    model=(model or ""),
                                    sub_model=(effective_sub_model or ""),
                                    task_id=task["id"],
                                    run_idx=i + 1,
                                    ok=False,
                                    matched=False,
                                    elapsed_s=0.0,
                                    context_chars=len(build_context(task)),
                                    query_chars=len(build_task_query(task, mode.name)),
                                    llm_calls=0,
                                    prompt_tokens=0,
                                    completion_tokens=0,
                                    total_tokens=0,
                                    prompt_chars_logged=0,
                                    response_chars_logged=0,
                                    estimated_cost_usd=0.0,
                                    is_hallucination_probe=(
                                        normalize_text(task.get("check", {}).get("value", ""))
                                        == "insufficient_information"
                                    ),
                                    answer="",
                                    error=(
                                        f"strict depth invalid: require_min_depth={args.require_min_depth} > mode max_depth={mdepth}"
                                    ),
                                )
                            )
                            continue
                    if (
                        args.depth_enforcement == "strict"
                        and args.require_min_recursive_calls is not None
                        and args.require_min_recursive_calls > 0
                    ):
                        mdepth = mode_max_depth(mode)
                        if mdepth is not None and mdepth <= 0:
                            label = model if model else "(default)"
                            print(
                                f"[{label}][{mode.name}] {task['id']} run={i+1} -> ERROR (strict recursive-call policy invalid: require_min_recursive_calls={args.require_min_recursive_calls} but mode max_depth={mdepth})"
                            )
                            results.append(
                                RunResult(
                                    mode=mode.name,
                                    model=(model or ""),
                                    sub_model=(effective_sub_model or ""),
                                    task_id=task["id"],
                                    run_idx=i + 1,
                                    ok=False,
                                    matched=False,
                                    elapsed_s=0.0,
                                    context_chars=len(build_context(task)),
                                    query_chars=len(build_task_query(task, mode.name)),
                                    llm_calls=0,
                                    prompt_tokens=0,
                                    completion_tokens=0,
                                    total_tokens=0,
                                    prompt_chars_logged=0,
                                    response_chars_logged=0,
                                    estimated_cost_usd=0.0,
                                    is_hallucination_probe=(
                                        normalize_text(task.get("check", {}).get("value", ""))
                                        == "insufficient_information"
                                    ),
                                    answer="",
                                    error=(
                                        "strict recursive-call policy invalid: "
                                        f"require_min_recursive_calls={args.require_min_recursive_calls} "
                                        f"but mode max_depth={mdepth}"
                                    ),
                                )
                            )
                            continue
                    attempts = args.attempts_per_run
                    r: Optional[RunResult] = None
                    for attempt in range(1, attempts + 1):
                        attempt_result = run_one(
                            mode=mode,
                            task=task,
                            base_url=args.base_url,
                            api_key=api_key,
                            model=(model or None),
                            sub_model=(effective_sub_model or None),
                            timeout_s=args.timeout,
                            prompt_cost_per_1m=args.prompt_cost_per_1m,
                            completion_cost_per_1m=args.completion_cost_per_1m,
                            axon_verbose=not args.no_axon_verbose,
                            is_hallucination_probe=(
                                normalize_text(task.get("check", {}).get("value", ""))
                                == "insufficient_information"
                            ),
                            pricing=pricing,
                            policy_profile=args.policy_profile,
                            inject_policy_into_context=args.inject_policy_into_context,
                            depth_enforcement=args.depth_enforcement,
                            require_min_depth=args.require_min_depth,
                            require_min_recursive_calls=args.require_min_recursive_calls,
                        )
                        attempt_result.run_idx = i + 1
                        r = attempt_result
                        if (
                            not attempt_result.ok
                            and attempt < attempts
                            and is_transient_error(attempt_result.error)
                        ):
                            delay = args.retry_backoff_s * float(attempt)
                            label = model if model else "(default)"
                            print(
                                f"[{label}][{mode.name}] {task['id']} run={i+1} transient error; retry {attempt+1}/{attempts} after {delay:.1f}s"
                            )
                            if delay > 0:
                                time.sleep(delay)
                            continue
                        break

                    assert r is not None
                    results.append(r)
                    status = "PASS" if r.matched else "FAIL"
                    if not r.ok:
                        status = "ERROR"
                    label = model if model else "(default)"
                    print(
                        f"[{label}][{mode.name}] {task['id']} run={i+1} -> {status} ({r.elapsed_s:.2f}s)"
                    )

    summary = summarize(results)
    print_summary(summary)

    out_path = (
        Path(args.out)
        if args.out
        else (repo_dir / "benchmarks" / "results" / f"results-{int(time.time())}.json")
    )
    payload = {
        "created_at": int(time.time()),
        "dataset": args.dataset,
        "base_url": args.base_url,
        "models": models,
        "sub_model": args.sub_model,
        "pricing_from_models_api": bool(args.pricing_from_models_api),
        "runs": args.runs,
        "attempts_per_run": args.attempts_per_run,
        "retry_backoff_s": args.retry_backoff_s,
        "timeout": args.timeout,
        "policy_profile": args.policy_profile,
        "inject_policy_into_context": bool(args.inject_policy_into_context),
        "depth_enforcement": args.depth_enforcement,
        "require_min_depth": args.require_min_depth,
        "require_min_recursive_calls": args.require_min_recursive_calls,
        "modes": [asdict(m) for m in modes],
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote report: {out_path}")
    if args.summary_md:
        md_path = Path(args.summary_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        write_summary_markdown(summary, md_path)
        print(f"Wrote markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
