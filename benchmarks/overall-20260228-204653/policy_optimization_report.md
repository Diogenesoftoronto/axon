# Prompt Policy Optimization Report

Objective: pass_rate - (0.3 * format_leak_rate) - (5.0 * avg_cost_usd)

| Rank | Policy | Model | Mode | N | Pass % | Format leak % | Avg cost | Avg time (s) | Objective |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | hf:MiniMaxAI/MiniMax-M2.5 | d0-i1 | 1359 | 59.23 | 0.00 | 0.001817 | 13.335 | 0.5833 |
| 2 | baseline | hf:MiniMaxAI/MiniMax-M2.5 | d0-i3 | 1359 | 46.36 | 0.07 | 0.001581 | 11.661 | 0.4555 |
| 3 | baseline | hf:MiniMaxAI/MiniMax-M2.5 | d1-i3 | 1359 | 40.69 | 0.15 | 0.001749 | 11.573 | 0.3977 |
| 4 | baseline | hf:MiniMaxAI/MiniMax-M2.5 | d3-i1 | 1359 | 37.38 | 0.15 | 0.001695 | 11.039 | 0.3649 |
| 5 | baseline | hf:MiniMaxAI/MiniMax-M2.5 | d6-i1 | 1359 | 10.89 | 0.15 | 0.000894 | 6.889 | 0.1040 |
