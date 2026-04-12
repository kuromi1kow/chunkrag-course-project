# Auto-generated Report Tables

\noindent\textit{Single-run percentages. Current subsets: SQuAD v2 (n=60), HotpotQA (n=30).}

## SQuAD v2

| System | EM | F1 | Recall@4 | Precision@4 | Avg chunk tokens | # chunks |
|---|---:|---:|---:|---:|---:|---:|
| `parametric_only` | 1.7 | 6.7 | -- | -- | -- | -- |
| `fixed_128` | 61.7 | 72.8 | 95.0 | 78.8 | 125.3 | 631 |
| `fixed_256` | 51.7 | 59.9 | 91.7 | 74.6 | 244.0 | 322 |
| `fixed_512` | 46.7 | 53.4 | 93.3 | 65.0 | 472.2 | 164 |
| `recursive_256` | 65.0 | 75.1 | 91.7 | 73.8 | 175.8 | 389 |
| `sentence_256` | 60.0 | 68.4 | 88.3 | 72.9 | 223.9 | 302 |
| `semantic_256` | 76.7 | 85.4 | 95.0 | 77.1 | 144.8 | 467 |

## HotpotQA

| System | EM | F1 | Recall@4 | Precision@4 | Avg chunk tokens | # chunks |
|---|---:|---:|---:|---:|---:|---:|
| `parametric_only` | 13.3 | 17.9 | -- | -- | -- | -- |
| `fixed_128` | 36.7 | 43.1 | 100.0 | 47.5 | 89.6 | 427 |
| `fixed_256` | 36.7 | 44.8 | 100.0 | 46.7 | 117.3 | 310 |
| `fixed_512` | 36.7 | 44.8 | 100.0 | 46.7 | 122.0 | 294 |
| `recursive_256` | 36.7 | 49.7 | 100.0 | 46.7 | 116.3 | 310 |
| `sentence_256` | 36.7 | 49.7 | 100.0 | 46.7 | 115.0 | 310 |
| `semantic_256` | 36.7 | 47.2 | 100.0 | 45.8 | 96.8 | 368 |

## Chonkie Comparison

| System | SQuAD EM | SQuAD F1 | HotpotQA F1 | Avg chunk tokens |
|---|---:|---:|---:|---:|
| `recursive_256` | 65.0 | 75.1 | 49.7 | 175.8 |
| `chonkie_recursive_256` | 66.7 | 75.9 | 49.7 | 174.3 |
| `semantic_256` | 76.7 | 85.4 | 47.2 | 144.8 |
| `chonkie_semantic_256` | 65.0 | 77.1 | 49.7 | 101.5 |
