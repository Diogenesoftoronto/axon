| Model | Sub-model | Mode | Pass rate | 95% CI | Avg time (s) | Avg tokens | Avg cost (USD) | Composite | Readiness |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| hf:MiniMaxAI/MiniMax-M2.5 | hf:MiniMaxAI/MiniMax-M2.5 | current-default | 50.00% | [9.45, 90.55] | 13.894 | 1313.0 | 0.001129 | 76.50 | Promising |
| hf:MiniMaxAI/MiniMax-M2.5 | hf:MiniMaxAI/MiniMax-M2.5 | current-depth1-iter3 | 100.00% | [34.24, 100.00] | 24.164 | 5009.0 | 0.005410 | 84.83 | Promising |
| hf:MiniMaxAI/MiniMax-M2.5 | hf:MiniMaxAI/MiniMax-M2.5 | current-depth6-single-pass | 100.00% | [34.24, 100.00] | 8.499 | 1248.0 | 0.000934 | 100.00 | Production Candidate |
| hf:MiniMaxAI/MiniMax-M2.5 | hf:MiniMaxAI/MiniMax-M2.5 | current-no-recursion-best-of-3 | 100.00% | [34.24, 100.00] | 18.953 | 5504.5 | 0.005468 | 85.31 | Production Candidate |
