# Reviewer-driven robustness analysis

This file is generated from archived prediction and summary artifacts. Retrieval values are document-level AllHit@4 means ± sample SD over seeds 13, 21, 34. Generator intervals are recomputed 2,000-draw question-level percentile-bootstrap 95% intervals; paired contrasts use 20,000 bootstrap draws.

## Retrieval robustness

AnsVis@4 excludes HotpotQA questions whose normalized reference is `yes` or `no`, because literal label presence is not evidence that the supporting passage was retrieved. Applicable per-seed sample sizes are retained in the JSON output.

| Dataset | Embedder | Retriever | Best AllHit chunker | AllHit@4 (%) | Best AnsVis chunker | AnsVis@4 (%) |
|---|---|---|---|---:|---|---:|
| SQuAD 2.0 | MiniLM | dense | `fixed_128` | 97.1 ± 1.4 | `sentence_254` | 84.7 ± 1.8 |
| SQuAD 2.0 | MiniLM | hybrid | `fixed_128` | 98.9 ± 0.4 | `sentence_254` | 93.3 ± 3.1 |
| SQuAD 2.0 | BGE-small | dense | `fixed_128` | 97.8 ± 0.4 | `sentence_254` | 89.1 ± 2.0 |
| SQuAD 2.0 | BGE-small | hybrid | `fixed_128` | 99.1 ± 1.0 | `sentence_254` | 93.3 ± 1.3 |
| HotpotQA | MiniLM | dense | `sentence_254` | 50.7 ± 4.8 | `sentence_254` | 66.1 ± 5.2 |
| HotpotQA | MiniLM | hybrid | `fixed_254` | 48.0 ± 5.8 | `sentence_254` | 64.7 ± 5.3 |
| HotpotQA | BGE-small | dense | `recursive_254`/`sentence_254` | 70.2 ± 8.9/70.2 ± 8.9 | `sentence_254` | 78.6 ± 5.3 |
| HotpotQA | BGE-small | hybrid | `recursive_254` | 58.7 ± 3.5 | `sentence_254` | 72.3 ± 3.5 |
| TechQA | MiniLM | dense | `fixed_128` | 88.0 ± 3.0 | `sentence_254` | 48.3 ± 5.3 |
| TechQA | MiniLM | hybrid | `fixed_128`/`sentence_254` | 88.7 ± 3.0/88.7 ± 2.0 | `sentence_254` | 51.0 ± 4.6 |
| TechQA | BGE-small | dense | `fixed_254` | 88.8 ± 0.8 | `sentence_254` | 48.7 ± 4.5 |
| TechQA | BGE-small | hybrid | `fixed_128` | 90.8 ± 1.3 | `sentence_254` | 49.2 ± 5.0 |

BM25 scoring uses no embeddings, and both configurations use the same pinned MiniLM chunk tokenizer. They produced exactly identical per-seed results, so the best BM25 cell for each dataset is shown once below.

| Dataset | Best AllHit chunker | AllHit@4 (%) | Best AnsVis chunker | AnsVis@4 (%) |
|---|---|---:|---|---:|
| SQuAD 2.0 | `fixed_254`/`sentence_254` | 98.4 ± 0.8/98.4 ± 0.8 | `sentence_254` | 94.0 ± 1.8 |
| HotpotQA | `fixed_128` | 33.8 ± 2.0 | `sentence_254` | 58.9 ± 8.1 |
| TechQA | `sentence_254` | 86.8 ± 1.0 | `sentence_254` | 46.2 ± 3.5 |

## Qwen2.5-1.5B generation robustness

Cells show F1 percentage points and marginal 2,000-draw 95% confidence intervals.

| Dataset | No context | `fixed_128` | `fixed_254` | `recursive_254` | `sentence_254` |
|---|---:|---:|---:|---:|---:|
| SQuAD 2.0 | 17.3 [10.0, 24.9] | 66.6 [55.0, 77.1] | 61.5 [50.1, 71.9] | 65.6 [54.5, 75.8] | 65.3 [54.3, 75.6] |
| HotpotQA | 21.9 [8.9, 35.6] | 50.3 [33.1, 68.6] | 50.3 [33.5, 67.0] | 47.0 [29.6, 64.1] | 53.4 [36.3, 70.0] |
| TechQA | 14.0 [11.7, 16.4] | 21.3 [17.0, 25.8] | 23.7 [19.7, 27.8] | 22.2 [17.9, 27.2] | 20.4 [16.5, 24.8] |

### Paired F1 contrasts

Differences are `recursive_254` minus the comparator. CIs use 20,000 paired bootstrap draws; two-sided sign-flip tests use 100,000 draws. The primary Holm correction covers all 18 contrasts across the included generators.

| Dataset | Comparator | ΔF1 (pp) | 95% CI (pp) | Raw p | Global Holm p |
|---|---|---:|---:|---:|---:|
| SQuAD 2.0 | `fixed_128` | -1.00 | [-8.72, +6.50] | 0.81173 | 1.00000 |
| SQuAD 2.0 | `fixed_254` | +4.06 | [-2.09, +10.33] | 0.20722 | 1.00000 |
| SQuAD 2.0 | `sentence_254` | +0.24 | [-8.33, +8.89] | 1.00000 | 1.00000 |
| HotpotQA | `fixed_128` | -3.33 | [-16.67, +9.33] | 0.71607 | 1.00000 |
| HotpotQA | `fixed_254` | -3.33 | [-10.00, +0.00] | 1.00000 | 1.00000 |
| HotpotQA | `sentence_254` | -6.38 | [-19.05, +5.33] | 0.37373 | 1.00000 |
| TechQA | `fixed_128` | +0.93 | [-2.51, +4.15] | 0.59445 | 1.00000 |
| TechQA | `fixed_254` | -1.50 | [-5.01, +2.01] | 0.41338 | 1.00000 |
| TechQA | `sentence_254` | +1.80 | [-1.37, +5.28] | 0.31521 | 1.00000 |

### Qwen2.5-1.5B token-budget audit

| Dataset | System | Input limit | Output limit | Context cut | Length capped | Mean full | Mean used | Mean generated | Max generated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SQuAD 2.0 | `parametric_only` | 1536 | 96 | 0/60 | 0/60 | 51.8 | 51.8 | 5.8 | 15 |
| SQuAD 2.0 | `fixed_128` | 1536 | 96 | 0/60 | 0/60 | 657.2 | 657.2 | 6.8 | 25 |
| SQuAD 2.0 | `fixed_254` | 1536 | 96 | 0/60 | 0/60 | 1181.0 | 1181.0 | 6.7 | 25 |
| SQuAD 2.0 | `recursive_254` | 1536 | 96 | 0/60 | 0/60 | 891.9 | 891.9 | 6.8 | 25 |
| SQuAD 2.0 | `sentence_254` | 1536 | 96 | 0/60 | 0/60 | 1054.9 | 1054.9 | 6.3 | 25 |
| HotpotQA | `parametric_only` | 1536 | 96 | 0/30 | 0/30 | 58.8 | 58.8 | 5.4 | 12 |
| HotpotQA | `fixed_128` | 1536 | 96 | 0/30 | 0/30 | 550.9 | 550.9 | 5.3 | 13 |
| HotpotQA | `fixed_254` | 1536 | 96 | 0/30 | 0/30 | 622.2 | 622.2 | 5.5 | 13 |
| HotpotQA | `recursive_254` | 1536 | 96 | 0/30 | 0/30 | 621.7 | 621.7 | 5.4 | 13 |
| HotpotQA | `sentence_254` | 1536 | 96 | 0/30 | 0/30 | 602.3 | 602.3 | 5.6 | 13 |
| TechQA | `parametric_only` | 1536 | 512 | 0/50 | 0/50 | 128.2 | 128.2 | 47.8 | 169 |
| TechQA | `fixed_128` | 1536 | 512 | 0/50 | 0/50 | 727.0 | 727.0 | 66.1 | 158 |
| TechQA | `fixed_254` | 1536 | 512 | 2/50 | 0/50 | 1213.1 | 1208.2 | 75.7 | 177 |
| TechQA | `recursive_254` | 1536 | 512 | 1/50 | 0/50 | 1003.9 | 1001.3 | 80.0 | 194 |
| TechQA | `sentence_254` | 1536 | 512 | 1/50 | 0/50 | 1002.9 | 1001.4 | 70.0 | 183 |

## Mistral-7B generation robustness

Cells show F1 percentage points and marginal 2,000-draw 95% confidence intervals.

| Dataset | No context | `fixed_128` | `fixed_254` | `recursive_254` | `sentence_254` |
|---|---:|---:|---:|---:|---:|
| SQuAD 2.0 | 27.8 [19.2, 36.6] | 61.6 [49.9, 72.2] | 55.8 [45.1, 66.1] | 54.4 [43.3, 65.3] | 53.7 [42.3, 64.0] |
| HotpotQA | 23.4 [13.6, 33.9] | 45.4 [29.6, 61.4] | 47.2 [31.9, 63.1] | 45.7 [30.1, 61.7] | 49.6 [33.4, 65.7] |
| TechQA | 16.0 [13.7, 18.4] | 24.2 [19.6, 29.1] | 26.5 [22.1, 31.1] | 23.8 [19.1, 28.7] | 25.0 [20.7, 29.8] |

### Paired F1 contrasts

Differences are `recursive_254` minus the comparator. CIs use 20,000 paired bootstrap draws; two-sided sign-flip tests use 100,000 draws. The primary Holm correction covers all 18 contrasts across the included generators.

| Dataset | Comparator | ΔF1 (pp) | 95% CI (pp) | Raw p | Global Holm p |
|---|---|---:|---:|---:|---:|
| SQuAD 2.0 | `fixed_128` | -7.23 | [-13.62, -1.77] | 0.01565 | 0.28170 |
| SQuAD 2.0 | `fixed_254` | -1.45 | [-8.17, +5.02] | 0.67758 | 1.00000 |
| SQuAD 2.0 | `sentence_254` | +0.66 | [-5.67, +7.46] | 0.85461 | 1.00000 |
| HotpotQA | `fixed_128` | +0.33 | [-8.83, +10.00] | 0.93834 | 1.00000 |
| HotpotQA | `fixed_254` | -1.48 | [-4.44, +0.00] | 1.00000 | 1.00000 |
| HotpotQA | `sentence_254` | -3.91 | [-15.78, +6.61] | 0.56417 | 1.00000 |
| TechQA | `fixed_128` | -0.35 | [-3.43, +2.54] | 0.82485 | 1.00000 |
| TechQA | `fixed_254` | -2.69 | [-5.61, +0.11] | 0.07211 | 1.00000 |
| TechQA | `sentence_254` | -1.22 | [-4.73, +2.57] | 0.52685 | 1.00000 |

### Mistral-7B token-budget audit

| Dataset | System | Input limit | Output limit | Context cut | Length capped | Mean full | Mean used | Mean generated | Max generated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SQuAD 2.0 | `parametric_only` | 1536 | 96 | 0/60 | 0/60 | 48.1 | 48.1 | 15.6 | 51 |
| SQuAD 2.0 | `fixed_128` | 1536 | 96 | 0/60 | 0/60 | 706.9 | 706.9 | 14.1 | 47 |
| SQuAD 2.0 | `fixed_254` | 1536 | 96 | 0/60 | 0/60 | 1275.5 | 1275.5 | 14.9 | 55 |
| SQuAD 2.0 | `recursive_254` | 1536 | 96 | 0/60 | 0/60 | 965.4 | 965.4 | 14.9 | 47 |
| SQuAD 2.0 | `sentence_254` | 1536 | 96 | 0/60 | 0/60 | 1152.7 | 1152.7 | 17.4 | 90 |
| HotpotQA | `parametric_only` | 1536 | 96 | 0/30 | 0/30 | 55.8 | 55.8 | 15.0 | 49 |
| HotpotQA | `fixed_128` | 1536 | 96 | 0/30 | 0/30 | 591.6 | 591.6 | 11.3 | 50 |
| HotpotQA | `fixed_254` | 1536 | 96 | 0/30 | 0/30 | 667.7 | 667.7 | 8.3 | 25 |
| HotpotQA | `recursive_254` | 1536 | 96 | 0/30 | 0/30 | 667.8 | 667.8 | 7.8 | 25 |
| HotpotQA | `sentence_254` | 1536 | 96 | 0/30 | 0/30 | 653.4 | 653.4 | 11.4 | 93 |
| TechQA | `parametric_only` | 1536 | 512 | 0/50 | 0/50 | 137.7 | 137.7 | 97.6 | 329 |
| TechQA | `fixed_128` | 1536 | 512 | 0/50 | 0/50 | 776.1 | 776.1 | 102.2 | 226 |
| TechQA | `fixed_254` | 1536 | 512 | 2/50 | 0/50 | 1284.0 | 1273.3 | 113.5 | 299 |
| TechQA | `recursive_254` | 1536 | 512 | 2/50 | 0/50 | 1069.2 | 1061.9 | 112.8 | 259 |
| TechQA | `sentence_254` | 1536 | 512 | 1/50 | 0/50 | 1156.4 | 1151.0 | 101.3 | 285 |
