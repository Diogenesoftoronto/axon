#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Any, Dict, List


def normalize_question(question: str) -> str:
    q = question.strip()
    if not q.endswith("?"):
        q = q + "?"
    return q


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapt Hugging Face BoolQ into Axon deterministic yes/no tasks"
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

    ds = load_dataset("boolq", split=args.split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    tasks: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        passage = str(row.get("passage", "")).strip()
        question = normalize_question(str(row.get("question", "")))
        answer = row.get("answer", None)
        if not passage or not question or answer is None:
            continue

        expected = "yes" if bool(answer) else "no"
        tasks.append(
            {
                "id": f"boolq_{args.split}_{idx:04d}",
                "context": passage,
                "query": f"Question: {question}\nReturn only yes or no.",
                "check": {"type": "yesno_exact", "value": expected},
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
