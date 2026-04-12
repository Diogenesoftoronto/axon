#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Any, Dict, List


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapt Hugging Face WinoGrande into Axon deterministic choice tasks"
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

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split=args.split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    tasks: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        sentence = str(row.get("sentence", "")).strip()
        option1 = str(row.get("option1", "")).strip()
        option2 = str(row.get("option2", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not sentence or not option1 or not option2 or answer not in {"1", "2"}:
            continue

        query = (
            "Fill in the blank with the correct option. Return only 1 or 2.\n\n"
            f"Sentence: {sentence}\n"
            f"1) {option1}\n"
            f"2) {option2}"
        )

        tasks.append(
            {
                "id": f"winogrande_{args.split}_{idx:04d}",
                "context": "Commonsense coreference resolution. Return only the option number.",
                "query": query,
                "check": {"type": "choice_exact", "value": answer},
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
