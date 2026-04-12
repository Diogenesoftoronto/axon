| Model | Sub-model | Mode | Pass rate | 95% CI | Avg time (s) | Avg tokens | Avg cost (USD) | Composite | Readiness |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| hf:MiniMaxAI/MiniMax-M2.5 | hf:MiniMaxAI/MiniMax-M2.5 | current-default | 50.00% | [9.45, 90.55] | 21.462 | 1653.0 | 0.001745 | 76.23 | Promising |
| hf:MiniMaxAI/MiniMax-M2.5 | hf:MiniMaxAI/MiniMax-M2.5 | current-depth1-iter3 | 100.00% | [34.24, 100.00] | 19.982 | 7098.5 | 0.006987 | 88.41 | Production Candidate |
| hf:MiniMaxAI/MiniMax-M2.5 | hf:MiniMaxAI/MiniMax-M2.5 | current-depth6-single-pass | 0.00% | [0.00, 65.76] | 22.785 | 1443.5 | 0.001117 | 60.76 | NotProductionReady |
| hf:MiniMaxAI/MiniMax-M2.5 | hf:MiniMaxAI/MiniMax-M2.5 | current-no-recursion-best-of-3 | 50.00% | [9.45, 90.55] | 72.084 | 23295.5 | 0.027854 | 63.06 | Experimental |
