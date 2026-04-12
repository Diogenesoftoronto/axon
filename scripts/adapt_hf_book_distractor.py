#!/usr/bin/env python3
import argparse
import json
import random
import re
import sys
from typing import Any, Dict, List, Optional


def pick_text_field(row: Dict[str, Any], preferred: Optional[str]) -> Optional[str]:
    if preferred and isinstance(row.get(preferred), str) and row.get(preferred).strip():
        return preferred
    for key, value in row.items():
        if isinstance(value, str) and len(value.strip()) >= 80:
            return key
    return None


def clean_excerpt(text: str, max_len: int = 280) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0]
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create long-context book+distractor tasks from a HF text dataset"
    )
    parser.add_argument("--dataset", default="bookcorpusopen")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--size-chars", type=int, default=800000)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default="-")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"Missing dependency: datasets ({e})", file=sys.stderr)
        print("Install with: pip install datasets", file=sys.stderr)
        return 2

    random.seed(args.seed)
    ds = load_dataset(args.dataset, args.config, split=args.split)
    if len(ds) < (args.limit * 3):
        print(
            f"Dataset too small for requested limit. Need at least {args.limit * 3} rows, got {len(ds)}.",
            file=sys.stderr,
        )
        return 2

    indices = list(range(len(ds)))
    random.shuffle(indices)
    picks = indices[: args.limit * 3]

    tasks: List[Dict[str, Any]] = []
    for task_idx in range(args.limit):
        target_row = ds[picks[task_idx * 3]]
        d1_row = ds[picks[task_idx * 3 + 1]]
        d2_row = ds[picks[task_idx * 3 + 2]]

        target_field = pick_text_field(target_row, args.text_field)
        d1_field = pick_text_field(d1_row, args.text_field)
        d2_field = pick_text_field(d2_row, args.text_field)
        if not target_field or not d1_field or not d2_field:
            continue

        target_excerpt = clean_excerpt(target_row[target_field])
        d1_excerpt = clean_excerpt(d1_row[d1_field])
        d2_excerpt = clean_excerpt(d2_row[d2_field])

        answer_name = f"Archivist-{task_idx + 1}"
        answer_item = f"artifact-{100 + task_idx}"
        answer_city = f"City-{task_idx + 1}"

        insertions = [
            {
                "at": 140000 + (task_idx * 5000),
                "text": f"[TARGET] verified courier={answer_name} item={answer_item} city={answer_city}.\\n",
            },
            {
                "at": 360000 + (task_idx * 5000),
                "text": f"[DISTRACTOR] draft courier=Wrong-{task_idx} item=decoy-{task_idx} city=Elsewhere-{task_idx}.\\n",
            },
            {
                "at": 620000 + (task_idx * 5000),
                "text": f"[DISTRACTOR] verified courier=Noise-{task_idx} item=noise-{task_idx} city=NoiseCity-{task_idx}.\\n",
            },
        ]

        filler = (
            f"[BOOK-TARGET] {target_excerpt} "
            f"[BOOK-D1] {d1_excerpt} "
            f"[BOOK-D2] {d2_excerpt}\\n"
        )
        tasks.append(
            {
                "id": f"hf_books_distractor_{task_idx:03d}",
                "context_gen": {
                    "type": "log_haystack_v1",
                    "size_chars": args.size_chars,
                    "filler": filler,
                    "insertions": insertions,
                },
                "query": "Use only [TARGET] verified line. Return exactly courier|item|city.",
                "check": {
                    "type": "exact",
                    "value": f"{answer_name}|{answer_item}|{answer_city}",
                },
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
