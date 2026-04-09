# Results Summary

Primary experiment artifacts live in [`outputs/report_run/all_results.json`](/Users/assylkhan/Documents/NLP/outputs/report_run/all_results.json). The auxiliary Chonkie comparison lives in [`outputs/chonkie_report_run/all_results.json`](/Users/assylkhan/Documents/NLP/outputs/chonkie_report_run/all_results.json).

## SQuAD v2

| System | EM | F1 | Recall@4 | Precision@4 | Avg chunk tokens | # chunks | Gen latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `parametric_only` | 1.7 | 6.7 | - | - | - | - | - |
| `fixed_128` | 61.7 | 72.8 | 95.0 | 78.8 | 125.3 | 631 | 0.138 |
| `fixed_256` | 51.7 | 59.9 | 91.7 | 74.6 | 244.0 | 322 | 0.114 |
| `fixed_512` | 46.7 | 53.4 | 93.3 | 65.0 | 472.2 | 164 | 0.110 |
| `recursive_256` | 65.0 | 75.1 | 91.7 | 73.8 | 175.8 | 389 | 0.113 |
| `sentence_256` | 60.0 | 68.4 | 88.3 | 72.9 | 223.9 | 302 | 0.110 |
| `semantic_256` | 76.7 | 85.4 | 95.0 | 77.1 | 144.8 | 467 | 0.138 |

## HotpotQA

| System | EM | F1 | Recall@4 | Precision@4 | Avg chunk tokens | # chunks | Gen latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `parametric_only` | 13.3 | 17.9 | - | - | - | - | - |
| `fixed_128` | 36.7 | 43.1 | 100.0 | 47.5 | 89.6 | 427 | 0.116 |
| `fixed_256` | 36.7 | 44.8 | 100.0 | 46.7 | 117.3 | 310 | 0.104 |
| `fixed_512` | 36.7 | 44.8 | 100.0 | 46.7 | 122.0 | 294 | 0.094 |
| `recursive_256` | 36.7 | 49.7 | 100.0 | 46.7 | 116.3 | 310 | 0.114 |
| `sentence_256` | 36.7 | 49.7 | 100.0 | 46.7 | 115.0 | 310 | 0.087 |
| `semantic_256` | 36.7 | 47.2 | 100.0 | 45.8 | 96.8 | 368 | 0.107 |

## Takeaways

- On SQuAD, chunking mattered a lot. `semantic_256` was best, with `recursive_256` second.
- On HotpotQA, exact match was flat, but `recursive_256` and `sentence_256` improved token-F1 over all fixed-size baselines.
- Larger fixed chunks were consistently worse on SQuAD, suggesting that extra distractor context hurts answer generation even when recall remains high.
- The no-retrieval baseline was far behind on both datasets, so retrieval clearly mattered.

## Chonkie Comparison

| System | SQuAD EM | SQuAD F1 | HotpotQA F1 | Notes |
|---|---:|---:|---:|---|
| `recursive_256` | 65.0 | 75.1 | 49.7 | Strong non-semantic baseline |
| `chonkie_recursive_256` | 66.7 | 75.9 | 49.7 | Slight SQuAD gain, essentially same Hotpot behavior |
| `semantic_256` | 76.7 | 85.4 | 47.2 | Best SQuAD system overall |
| `chonkie_semantic_256` | 65.0 | 77.1 | 49.7 | Smaller chunks help Hotpot, but SQuAD drops noticeably |

- Recursive chunking was robust across implementations.
- Semantic chunking was more implementation-sensitive: Chonkie's variant produced more, smaller chunks and traded away SQuAD accuracy for better HotpotQA F1.
