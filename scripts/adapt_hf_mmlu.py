#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Any, Dict, List


LABELS = ["A", "B", "C", "D"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapt Hugging Face MMLU into Altum deterministic MCQ tasks"
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

    ds = load_dataset("cais/mmlu", "all", split=args.split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    tasks: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        question = str(row.get("question", "")).strip()
        choices = row.get("choices", [])
        answer_idx = row.get("answer", -1)
        if not question or not choices or answer_idx < 0 or answer_idx >= len(choices):
            continue

        answer_label = LABELS[answer_idx] if answer_idx < len(LABELS) else str(answer_idx)
        lines = []
        for i, text in enumerate(choices):
            label = LABELS[i] if i < len(LABELS) else str(i)
            lines.append(f"{label}) {text}")

        subject = str(row.get("subject", "general")).replace("_", " ")
        query = (
            f"Multiple-choice {subject} question. Return only the option label (A, B, C, D).\n\n"
            f"Question: {question}\n"
            "Options:\n"
            + "\n".join(lines)
        )

        tasks.append(
            {
                "id": f"mmlu_{args.split}_{idx:04d}",
                "context": "Answer exactly with the single correct option label.",
                "query": query,
                "check": {"type": "choice_exact", "value": answer_label},
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
