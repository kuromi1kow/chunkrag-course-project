# Error analysis — exploratory data analysis

All numbers below are for Mistral 7B Instruct v0.3 on the v2 prediction set (60 SQuAD, 30 HotpotQA, dense retrieval, k=4).

## 1. Headline coarse buckets (best chunker per dataset)

| | SQuAD (recursive_256) | HotpotQA (recursive_256) |
|---|---|---|
| EM correct | 25 / 60 (41.7%) | 8 / 30 (26.7%) |
| Retrieval failure | 5 (14.3% of EM=0) | 0 (0.0%) |
| Format / refusal (fixable) | 23 (65.7%) | 16 (72.7%) |
| Model error | 7 (20.0%) | 6 (27.3%) |


## 2. Bucket distribution across chunkers

Does the best-F1 chunker also have the lowest fixable-error rate? (Higher fixable% = more EM=0 cases are due to format/refusal rather than genuine wrong answers.)


### squad_v2

| Chunker | EM correct | EM=0 total | Retrieval fail % | Fixable % | Model error % |
|---|---|---|---|---|---|
| recursive_256 | 25 | 35 | 14.3 | 65.7 | 20.0 |
| semantic_256 | 25 | 35 | 8.6 | 71.4 | 20.0 |
| fixed_512 | 24 | 36 | 16.7 | 55.6 | 27.8 |
| fixed_128 | 23 | 37 | 8.1 | 51.4 | 40.5 |
| fixed_256 | 23 | 37 | 18.9 | 54.1 | 27.0 |
| sentence_256 | 21 | 39 | 20.5 | 53.8 | 25.6 |

### hotpot_qa

| Chunker | EM correct | EM=0 total | Retrieval fail % | Fixable % | Model error % |
|---|---|---|---|---|---|
| recursive_256 | 8 | 22 | 0.0 | 72.7 | 27.3 |
| semantic_256 | 7 | 23 | 0.0 | 73.9 | 26.1 |
| sentence_256 | 7 | 23 | 0.0 | 65.2 | 34.8 |
| fixed_128 | 6 | 24 | 0.0 | 62.5 | 37.5 |
| fixed_256 | 6 | 24 | 0.0 | 62.5 | 37.5 |
| fixed_512 | 6 | 24 | 0.0 | 62.5 | 37.5 |


## 3. Failure types by question type

Which question types are hardest, and what kind of failure dominates each? Computed for the two top-performing chunkers (`recursive_256` is best by EM, `semantic_256` has a slightly higher fixable share).


### squad_v2 / recursive_256

| Q type | n total | EM correct | False refusal | Verbose | Terse | Paraphrase | Partial | Wrong | Retr. fail |
|---|---|---|---|---|---|---|---|---|---|
| how_many | 7 | 4 (57%) | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| when | 6 | 5 (83%) | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| where | 3 | 0 (0%) | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| who | 5 | 4 (80%) | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| what | 26 | 8 (31%) | 2 | 10 | 1 | 1 | 2 | 0 | 2 |
| which | 2 | 1 (50%) | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| yes_no | 1 | 0 (0%) | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 10 | 3 (30%) | 0 | 3 | 0 | 1 | 2 | 0 | 1 |

### squad_v2 / semantic_256

| Q type | n total | EM correct | False refusal | Verbose | Terse | Paraphrase | Partial | Wrong | Retr. fail |
|---|---|---|---|---|---|---|---|---|---|
| how_many | 7 | 4 (57%) | 1 | 0 | 0 | 1 | 0 | 1 | 0 |
| when | 6 | 5 (83%) | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| where | 3 | 0 (0%) | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| who | 5 | 4 (80%) | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| what | 26 | 9 (35%) | 3 | 10 | 0 | 1 | 1 | 0 | 2 |
| which | 2 | 1 (50%) | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| yes_no | 1 | 0 (0%) | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 10 | 2 (20%) | 2 | 3 | 0 | 0 | 2 | 1 | 0 |

### hotpot_qa / recursive_256

| Q type | n total | EM correct | False refusal | Verbose | Terse | Paraphrase | Partial | Wrong | Retr. fail |
|---|---|---|---|---|---|---|---|---|---|
| when | 5 | 1 (20%) | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| who | 1 | 1 (100%) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| what | 5 | 1 (20%) | 0 | 1 | 1 | 0 | 0 | 2 | 0 |
| which | 3 | 1 (33%) | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| yes_no | 2 | 1 (50%) | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 14 | 3 (21%) | 4 | 3 | 1 | 0 | 1 | 2 | 0 |

### hotpot_qa / semantic_256

| Q type | n total | EM correct | False refusal | Verbose | Terse | Paraphrase | Partial | Wrong | Retr. fail |
|---|---|---|---|---|---|---|---|---|---|
| when | 5 | 0 (0%) | 4 | 0 | 0 | 0 | 0 | 1 | 0 |
| who | 1 | 1 (100%) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| what | 5 | 1 (20%) | 0 | 2 | 1 | 0 | 0 | 1 | 0 |
| which | 3 | 1 (33%) | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| yes_no | 2 | 1 (50%) | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 14 | 3 (21%) | 3 | 4 | 1 | 0 | 1 | 2 | 0 |


## 4. Prediction vs gold length per bucket

Token-level lengths after `normalize_answer`. Verbose buckets should have len(pred) >> len(gold); terse buckets the reverse. The compared gold is bucket-aware: the gold variant whose tokens are a subset of pred (for verbose), or whose tokens are a superset of pred (for terse). For other buckets we use the gold with highest token recall.


### squad_v2 / recursive_256

| Bucket | n | mean(len pred) | mean(len gold) | mean(pred − gold) |
|---|---|---|---|---|
| retrieval_failure | 5 | 1.0 | 2.2 | -1.2 |
| false_refusal | 4 | 1.0 | 7.2 | -6.2 |
| format_verbose | 14 | 7.1 | 2.2 | +4.9 |
| format_terse | 2 | 7.0 | 13.5 | -6.5 |
| paraphrase | 3 | 8.3 | 6.3 | +2.0 |
| partial_answer | 4 | 9.5 | 6.5 | +3.0 |
| wrong_answer | 3 | 9.7 | 1.7 | +8.0 |

### squad_v2 / semantic_256

| Bucket | n | mean(len pred) | mean(len gold) | mean(pred − gold) |
|---|---|---|---|---|
| retrieval_failure | 3 | 1.0 | 2.3 | -1.3 |
| false_refusal | 9 | 1.0 | 5.2 | -4.2 |
| format_verbose | 14 | 8.8 | 3.0 | +5.8 |
| paraphrase | 2 | 6.0 | 7.5 | -1.5 |
| partial_answer | 3 | 7.7 | 5.7 | +2.0 |
| wrong_answer | 4 | 4.8 | 1.5 | +3.2 |

### hotpot_qa / recursive_256

| Bucket | n | mean(len pred) | mean(len gold) | mean(pred − gold) |
|---|---|---|---|---|
| false_refusal | 9 | 1.0 | 2.4 | -1.4 |
| format_verbose | 5 | 4.0 | 1.6 | +2.4 |
| format_terse | 2 | 2.0 | 3.0 | -1.0 |
| partial_answer | 1 | 5.0 | 28.0 | -23.0 |
| wrong_answer | 5 | 3.2 | 2.0 | +1.2 |

### hotpot_qa / semantic_256

| Bucket | n | mean(len pred) | mean(len gold) | mean(pred − gold) |
|---|---|---|---|---|
| false_refusal | 8 | 1.0 | 2.2 | -1.2 |
| format_verbose | 7 | 6.1 | 2.1 | +4.0 |
| format_terse | 2 | 2.0 | 3.0 | -1.0 |
| partial_answer | 1 | 5.0 | 28.0 | -23.0 |
| wrong_answer | 5 | 2.4 | 1.6 | +0.8 |


## 5. What verbose predictions add

For every `format_verbose` case (gold tokens fully contained in pred), we extract the tokens **before** and **after** the gold span. The top patterns suggest where the model is wrapping the right answer in extra words. Counts are token-frequencies, not phrase frequencies.


### squad_v2 / recursive_256 (n=14 verbose cases)

**Top prefix tokens** (model echoes question subject before gold):
- `of` × 2
- `western` × 1
- `sticky` × 1
- `y` × 1
- `pestis` × 1
- `smaller` × 1
- `number` × 1
- `electors` × 1

**Top suffix tokens** (model adds clauses after gold):
- `and` × 3
- `of` × 2
- `rhine` × 2
- `with` × 1
- `vitamin` × 1
- `d` × 1
- `that` × 1
- `essentializes` × 1

**Sample full suffix phrases** (top 5):
- `"with vitamin d"` × 1
- `"that essentializes east"` × 1
- `"forces act only at very short distances"` × 1
- `"of 1996 recognises two types public and independent"` × 1
- `"and business regulation decline"` × 1

### squad_v2 / semantic_256 (n=14 verbose cases)

**Top prefix tokens** (model echoes question subject before gold):
- `by` × 2
- `of` × 2
- `public` × 1
- `and` × 1
- `independent` × 1
- `schools` × 1
- `recognized` × 1
- `sticky` × 1

**Top suffix tokens** (model adds clauses after gold):
- `and` × 3
- `with` × 2
- `vitamin` × 1
- `d` × 1
- `of` × 1
- `1996` × 1
- `business` × 1
- `regulation` × 1

**Sample full suffix phrases** (top 5):
- `"with vitamin d"` × 1
- `"of 1996"` × 1
- `"and business regulation decline adversely affect economic mobility"` × 1
- `"that capture prey"` × 1
- `"earthquake"` × 1

### hotpot_qa / recursive_256 (n=5 verbose cases)

**Top prefix tokens** (model echoes question subject before gold):
- `bill` × 1

**Top suffix tokens** (model adds clauses after gold):
- `kentucky` × 1
- `new` × 1
- `york` × 1
- `died` × 1
- `later` × 1
- `is` × 1
- `more` × 1
- `common` × 1

**Sample full suffix phrases** (top 5):
- `"kentucky"` × 1
- `"new york"` × 1
- `"died later"` × 1
- `"is more common in temperate regions"` × 1

### hotpot_qa / semantic_256 (n=7 verbose cases)

**Top prefix tokens** (model echoes question subject before gold):
- `bill` × 1
- `twelfth` × 1
- `united` × 1
- `states` × 1
- `army` × 1
- `group` × 1
- `commander` × 1
- `was` × 1

**Top suffix tokens** (model adds clauses after gold):
- `in` × 2
- `kentucky` × 1
- `new` × 1
- `york` × 1
- `bol` × 1
- `died` × 1
- `later` × 1
- `is` × 1

**Sample full suffix phrases** (top 5):
- `"kentucky"` × 1
- `"new york"` × 1
- `"in bol"` × 1
- `"died later"` × 1
- `"is more common in temperate regions"` × 1


## 6. False-refusal retrieval quality

How often was retrieval actually adequate when Mistral refused? If most false_refusals have `recall_at_k = 1.0`, the failure is purely a prompt-tone problem.


### squad_v2 / recursive_256

- n=4 refusals
- **4/4 (100%)** had `recall_at_k = 1.0` (every gold doc was in the prompt).
- recall_at_k distribution: 1.0: 4
- supporting_doc_coverage distribution: 1.0: 4
- Question types of refusals: what: 2, yes_no: 1, where: 1

### squad_v2 / semantic_256

- n=9 refusals
- **9/9 (100%)** had `recall_at_k = 1.0` (every gold doc was in the prompt).
- recall_at_k distribution: 1.0: 9
- supporting_doc_coverage distribution: 1.0: 9
- Question types of refusals: what: 3, other: 2, yes_no: 1, where: 1, when: 1, how_many: 1

### hotpot_qa / recursive_256

- n=9 refusals
- **4/9 (44%)** had `recall_at_k = 1.0` (every gold doc was in the prompt).
- recall_at_k distribution: 0.5: 5, 1.0: 4
- supporting_doc_coverage distribution: 0.5: 5, 1.0: 4
- Question types of refusals: other: 4, when: 4, yes_no: 1

### hotpot_qa / semantic_256

- n=8 refusals
- **4/8 (50%)** had `recall_at_k = 1.0` (every gold doc was in the prompt).
- recall_at_k distribution: 0.5: 4, 1.0: 4
- supporting_doc_coverage distribution: 0.5: 4, 1.0: 4
- Question types of refusals: when: 4, other: 3, yes_no: 1


## 7. Representative examples per bucket (annotated)

Each example is followed by a one-line note explaining why it failed and what fix would help.


### squad_v2 (recursive_256)

**retrieval_failure** (5 cases). _Wrong document retrieved; chunking/embedding can't recover the correct doc._
- Q: What entity owns V/Line?
- Gold: Victorian Government | the Victorian Government | Victorian Government
- Pred: unanswerable
- F1=0.00, recall@4=0.00

**false_refusal** (4 cases). _Retrieval recovered the gold doc(s) but Mistral refused. Soften the unanswerable instruction._
- Q: Was the Use of the DATANET 1 name correct 
- Gold: use of the name was incorrect all these services were managed by the same people within one department of KPN contributed to the confusion | the name was incorrect | Dutch PTT Telecom
- Pred: unanswerable
- F1=0.00, recall@4=1.00

**format_verbose** (14 cases). _Gold span is in the prediction, surrounded by extra words. Tighten prompt to forbid subject echoing/clauses._
- Q: What is thought to have happened to the y. pestis that caused the black death?
- Gold: may no longer exist | may no longer exist | may no longer exist
- Pred: Y. pestis may no longer exist
- F1=0.80, recall@4=1.00

**format_terse** (2 cases). _Prediction is a strict subset of the gold tokens — model gave less than the gold required._
- Q: What supplanted Frame Relay and X.25 
- Gold: supplanted by the Internet Protocol (IP) at the network layer, and the Asynchronous Transfer Mode (ATM) and or versions of Multi-Protocol Label Switching | Internet Protocol (IP) | Internet Protocol
- Pred: Internet Protocol (IP), Asynchronous Transfer Mode (ATM), and versions of Multi-Protocol Label Switching (
- F1=0.79, recall@4=1.00

**paraphrase** (3 cases). _High token overlap but neither subset; close paraphrase that EM can't credit._
- Q: Contracts must be designed to ensure what?
- Gold: that each side is capable of performing the obligations set out | that each side is capable of performing the obligations set out | each side is capable of performing the obligations set out
- Pred: Contracts must ensure each side is capable of performing obligations
- F1=0.74, recall@4=1.00

**partial_answer** (4 cases). _Some token overlap with gold but not subset; model partially understood._
- Q: According to Wilson's theorem, what factorial must be divisible by p if some integer p > 1 is to be considered prime?
- Gold: (p − 1)! + 1 | (p − 1)! + 1 | (p − 1)! + 1 | (p − 1)! + 1 | (p − 1)! + 1
- Pred: (p - 1)! + 1 is divisible by p
- F1=0.55, recall@4=1.00

**wrong_answer** (3 cases). _Zero token overlap; model genuinely produced an unrelated answer._
- Q: How many counties initially made up the definition of southern California?
- Gold: seven | seven | seven
- Pred: The context does not provide a specific number of counties initially making up southern California according to the historical attempts for separate statehood or territorial status
- F1=0.00, recall@4=1.00


### hotpot_qa (recursive_256)

**false_refusal** (9 cases). _Retrieval recovered the gold doc(s) but Mistral refused. Soften the unanswerable instruction._
- Q: Robert Earl Holding owned an oil company that was originally founded by who?
- Gold: Harry F. Sinclair
- Pred: unanswerable
- F1=0.00, recall@4=1.00

**format_verbose** (5 cases). _Gold span is in the prediction, surrounded by extra words. Tighten prompt to forbid subject echoing/clauses._
- Q: What Kentucky county has a population of 60,316 and features the Lake Louisvilla neighborhood?
- Gold: Oldham County
- Pred: Oldham County, Kentucky
- F1=0.80, recall@4=1.00

**format_terse** (2 cases). _Prediction is a strict subset of the gold tokens — model gave less than the gold required._
- Q: Pacific Mozart Ensemble performed which German composer's Der Lindberghflug in 2002?
- Gold: Kurt Julian Weill
- Pred: Kurt Weill
- F1=0.80, recall@4=0.50

**partial_answer** (1 cases). _Some token overlap with gold but not subset; model partially understood._
- Q: If the Charhki  Dadri crash was less dangerous than the Tenerife airport disaster, which occured firat? 
- Gold: On March 27, 1977, two Boeing 747 passenger jets, KLM Flight 4805 and Pan Am Flight 1736, collided on the runway at Los Rodeos Airport (now Tenerife North Airport)
- Pred: The Tenerife airport disaster occurred first
- F1=0.12, recall@4=1.00

**wrong_answer** (5 cases). _Zero token overlap; model genuinely produced an unrelated answer._
- Q: Which mountain is taller, Gasherbrum II or Langtang Ri?
- Gold: Gasherbrum II
- Pred: taller than Langtang Ri (72
- F1=0.00, recall@4=1.00

