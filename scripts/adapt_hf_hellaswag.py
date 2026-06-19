#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Any, Dict, List


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapt Hugging Face HellaSwag into Altum deterministic MCQ tasks"
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default="-")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"Missing dependency: datasets ({e})", file=sys.stderr)
        print("Install with: uv pip install --python .venv/bin/python datasets", file=sys.stderr)
        return 2

    ds = load_dataset("hellaswag", split=args.split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    tasks: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        ctx = str(row.get("ctx", "")).strip()
        endings = row.get("endings", [])
        label = str(row.get("label", "")).strip()
        if not ctx or not endings or label == "":
            continue
        if not label.isdigit():
            continue
        label_idx = int(label)
        if label_idx < 0 or label_idx >= len(endings):
            continue

        option_lines = []
        for i, ending in enumerate(endings):
            option_lines.append(f"{i}) {ending}")

        query = (
            "Choose the most plausible continuation. Return only the option index.\n\n"
            f"Context: {ctx}\n"
            "Options:\n"
            + "\n".join(option_lines)
        )
        tasks.append(
            {
                "id": f"hellaswag_{args.split}_{idx:04d}",
                "context": "Commonsense completion benchmark task.",
                "query": query,
                "check": {"type": "choice_exact", "value": str(label_idx)},
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
