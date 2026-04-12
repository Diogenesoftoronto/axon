# Gemini Mode-Profile Review (2026-02-25)

This note captures a targeted Gemini review for the four benchmark modes:

- `current-default` (`--max-depth 0 --max-iterations 1`)
- `current-no-recursion-best-of-3` (`--max-depth 0 --max-iterations 3`)
- `current-depth6-single-pass` (`--max-depth 6 --max-iterations 1`)
- `current-depth1-iter3` (`--max-depth 1 --max-iterations 3`)

## Command

```bash
cat /tmp/gemini_mode_profile_prompt.txt | \
  gemini -m gemini-3-pro-preview -p "Follow tasks A-D exactly." \
  > /tmp/gemini_mode_profile_review.md
```

## Outcome

Gemini identified coverage gaps in explicit mode discrimination and proposed a concrete 8-task deterministic suite.

Adopted artifact:

- `benchmarks/mode_profile_targeted.json`

The new suite includes:

- baseline exact extraction tasks,
- shallow cognitive trap tasks (best-of-3 sensitivity),
- deeper dependency/state-chain tasks (depth sensitivity),
- shallow delegation/merge tasks (depth-1 + iterations sensitivity).
