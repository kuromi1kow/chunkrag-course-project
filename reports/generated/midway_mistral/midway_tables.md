# Auto-generated Report Tables

\noindent\textit{Single-run percentages. Current subsets: SQuAD v2 (n=60), HotpotQA (n=30).}

## SQuAD v2

| System | EM | F1 | Recall@4 | Precision@4 | Avg chunk tokens | # chunks |
|---|---:|---:|---:|---:|---:|---:|
| `parametric_only` | 13.3 | 29.2 | 0.0 | 0.0 | -- | -- |
| `fixed_128` | 38.3 | 50.7 | 95.0 | 77.1 | 125.3 | 631 |
| `fixed_256` | 38.3 | 56.7 | 88.3 | 73.8 | 244.0 | 322 |
| `fixed_512` | 40.0 | 53.7 | 90.0 | 63.7 | 472.2 | 164 |
| `recursive_256` | 41.7 | 60.6 | 91.7 | 72.5 | 175.8 | 389 |
| `sentence_256` | 35.0 | 51.0 | 86.7 | 72.1 | 223.9 | 302 |
| `semantic_256` | 41.7 | 56.9 | 95.0 | 74.6 | 144.8 | 467 |

## HotpotQA

| System | EM | F1 | Recall@4 | Precision@4 | Avg chunk tokens | # chunks |
|---|---:|---:|---:|---:|---:|---:|
| `parametric_only` | 10.0 | 27.7 | 0.0 | 0.0 | -- | -- |
| `fixed_128` | 20.0 | 36.5 | 75.0 | 42.5 | 89.3 | 421 |
| `fixed_256` | 20.0 | 34.3 | 75.0 | 40.8 | 114.2 | 313 |
| `fixed_512` | 20.0 | 35.7 | 76.7 | 41.7 | 117.4 | 301 |
| `recursive_256` | 26.7 | 42.6 | 76.7 | 42.5 | 113.6 | 313 |
| `sentence_256` | 23.3 | 37.0 | 76.7 | 41.7 | 112.7 | 313 |
| `semantic_256` | 23.3 | 42.4 | 76.7 | 42.5 | 97.2 | 363 |

## Chonkie Comparison

| System | SQuAD EM | SQuAD F1 | HotpotQA F1 | Avg chunk tokens |
|---|---:|---:|---:|---:|
| `recursive_256` | 65.0 | 75.1 | 49.7 | 175.8 |
| `chonkie_recursive_256` | 66.7 | 75.9 | 49.7 | 174.3 |
| `semantic_256` | 76.7 | 85.4 | 47.2 | 144.8 |
| `chonkie_semantic_256` | 65.0 | 77.1 | 49.7 | 101.5 |
