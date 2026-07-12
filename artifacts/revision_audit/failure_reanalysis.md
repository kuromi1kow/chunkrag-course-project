# Corrected failure reanalysis

For SQuAD, the audit checks whether a normalized gold answer string survives in the reconstructed post-truncation context. For HotpotQA, evidence is considered incomplete unless both supporting documents occur among fully consumed chunks. Response-form categories remain diagnostic hypotheses rather than verified fixes.

## squad_v2

### fixed_128 (EM=0: 37)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 10 | 27.0 | 15.4-43.0 |
| response_form_candidate | 14 | 37.8 | 24.1-53.9 |
| answer_content_error | 13 | 35.1 | 21.8-51.2 |

### fixed_256 (EM=0: 37)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 16 | 43.2 | 28.7-59.1 |
| response_form_candidate | 13 | 35.1 | 21.8-51.2 |
| answer_content_error | 8 | 21.6 | 11.4-37.2 |

### fixed_512 (EM=0: 36)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 15 | 41.7 | 27.1-57.8 |
| response_form_candidate | 13 | 36.1 | 22.5-52.4 |
| answer_content_error | 8 | 22.2 | 11.7-38.1 |

### recursive_256 (EM=0: 35)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 8 | 22.9 | 12.1-39.0 |
| response_form_candidate | 19 | 54.3 | 38.2-69.5 |
| answer_content_error | 8 | 22.9 | 12.1-39.0 |

### sentence_256 (EM=0: 39)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 12 | 30.8 | 18.6-46.4 |
| response_form_candidate | 18 | 46.2 | 31.6-61.4 |
| answer_content_error | 9 | 23.1 | 12.6-38.3 |

### semantic_256 (EM=0: 35)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 6 | 17.1 | 8.1-32.7 |
| response_form_candidate | 22 | 62.9 | 46.3-76.8 |
| answer_content_error | 7 | 20.0 | 10.0-35.9 |

## hotpot_qa

### fixed_128 (EM=0: 24)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 13 | 54.2 | 35.1-72.1 |
| response_form_candidate | 7 | 29.2 | 14.9-49.2 |
| answer_content_error | 4 | 16.7 | 6.7-35.9 |

### fixed_256 (EM=0: 24)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 13 | 54.2 | 35.1-72.1 |
| response_form_candidate | 7 | 29.2 | 14.9-49.2 |
| answer_content_error | 4 | 16.7 | 6.7-35.9 |

### fixed_512 (EM=0: 24)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 12 | 50.0 | 31.4-68.6 |
| response_form_candidate | 8 | 33.3 | 18.0-53.3 |
| answer_content_error | 4 | 16.7 | 6.7-35.9 |

### recursive_256 (EM=0: 22)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 13 | 59.1 | 38.7-76.7 |
| response_form_candidate | 6 | 27.3 | 13.2-48.2 |
| answer_content_error | 3 | 13.6 | 4.7-33.3 |

### sentence_256 (EM=0: 23)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 13 | 56.5 | 36.8-74.4 |
| response_form_candidate | 6 | 26.1 | 12.5-46.5 |
| answer_content_error | 4 | 17.4 | 7.0-37.1 |

### semantic_256 (EM=0: 23)

| Coarse category | n | % | Wilson 95% CI |
|---|---:|---:|---:|
| evidence_limited | 13 | 56.5 | 36.8-74.4 |
| response_form_candidate | 8 | 34.8 | 18.8-55.1 |
| answer_content_error | 2 | 8.7 | 2.4-26.8 |

