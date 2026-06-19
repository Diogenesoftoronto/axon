#!/usr/bin/env python3
import argparse
import json
import re
import sys
from typing import Any, Dict, List


def extract_final_number(answer: str) -> str:
    # GSM8K canonical answers usually end with "#### <number>".
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer)
    if not m:
        raise ValueError(f"Could not parse GSM8K final answer: {answer[:120]!r}")
    return m.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapt HF GSM8K into Altum benchmark format")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default="-")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"Missing dependency: datasets ({e})", file=sys.stderr)
        print("Install with: uv pip install --python .venv/bin/python datasets", file=sys.stderr)
        return 2

    ds = load_dataset("gsm8k", "main", split=args.split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    tasks: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        final_number = extract_final_number(row["answer"])
        tasks.append(
            {
                "id": f"gsm8k_{args.split}_{idx:04d}",
                "context": "Solve the arithmetic word problem. Return only the final number.",
                "query": row["question"],
                "check": {"type": "number_exact", "value": final_number},
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
