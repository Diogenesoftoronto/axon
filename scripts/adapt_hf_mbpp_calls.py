#!/usr/bin/env python3
import argparse
import ast
import json
import re
import sys
from typing import Any, Dict, List, Tuple


ASSERT_RE = re.compile(r"^\s*assert\s+(.+?)\s*==\s*(.+?)\s*$")


def parse_test_assert(assert_line: str) -> Tuple[str, str]:
    m = ASSERT_RE.match(assert_line)
    if not m:
        raise ValueError(f"Unsupported MBPP assert format: {assert_line!r}")
    call_expr = m.group(1).strip()
    expected_expr = m.group(2).strip()
    expected_obj = ast.literal_eval(expected_expr)
    return call_expr, str(expected_obj)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapt HF MBPP into deterministic call-output Axon tasks"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--calls-per-task", type=int, default=3)
    parser.add_argument("--out", default="-")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"Missing dependency: datasets ({e})", file=sys.stderr)
        print("Install with: uv pip install --python .venv/bin/python datasets", file=sys.stderr)
        return 2

    ds = load_dataset("mbpp", split=args.split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    tasks: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        tests = row.get("test_list", [])
        parsed: List[Tuple[str, str]] = []
        for t in tests:
            try:
                parsed.append(parse_test_assert(t))
            except Exception:
                continue
        if len(parsed) < 1:
            continue
        parsed = parsed[: max(1, args.calls_per_task)]

        call_lines = [p[0] for p in parsed]
        expected_lines = [p[1] for p in parsed]
        expected = "\n".join(expected_lines)

        query = (
            "Problem:\n"
            f"{row.get('text', '').strip()}\n\n"
            "Compute outputs for the following Python function calls and return only the outputs, "
            "one per line, in the same order:\n"
            + "\n".join(call_lines)
        )
        tasks.append(
            {
                "id": f"mbpp_{args.split}_{idx:04d}",
                "context": "You are solving a deterministic programming task from MBPP.",
                "query": query,
                "check": {"type": "lines_in_order", "value": expected, "values": expected_lines},
            }
        )

    payload = json.dumps(tasks, indent=2)
    if args.out == "-":
        print(payload)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"Wrote {len(tasks)} tasks to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
