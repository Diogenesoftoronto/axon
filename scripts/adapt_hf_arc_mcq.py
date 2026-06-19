#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Any, Dict, List


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapt Hugging Face ARC (AI2 ARC) into Altum deterministic MCQ tasks"
    )
    parser.add_argument("--config", default="ARC-Challenge", help="ARC-Challenge or ARC-Easy")
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

    ds = load_dataset("ai2_arc", args.config, split=args.split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    tasks: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        answer_key = str(row.get("answerKey", "")).strip().upper()
        question = str(row.get("question", "")).strip()
        choices = row.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if not question or not labels or not texts:
            continue
        if answer_key not in labels:
            continue

        lines = []
        for label, text in zip(labels, texts):
            lines.append(f"{label}) {text}")

        query = (
            "Multiple-choice science question. Return only the option label (A, B, C, D, ...).\n\n"
            f"Question: {question}\n"
            "Options:\n"
            + "\n".join(lines)
        )

        tasks.append(
            {
                "id": f"arc_{args.config.lower().replace('-', '_')}_{args.split}_{idx:04d}",
                "context": "Answer exactly with the single correct option label.",
                "query": query,
                "check": {"type": "choice_exact", "value": answer_key},
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
