# Indicator linking results

- validation indicators (unseen): 11 of 59 candidates
- training indicators: 48, overlap: 0
- epochs: 80, seeds: [13, 42, 73, 101, 137]
- vocabulary: 3810, train OOV rate: 0.0773

## Held-out-indicator validation (11 unseen definitions, 59 candidates)

| Condition | Recall@1 | Recall@3 | MRR | Seed SD |
|---|---:|---:|---:|---:|
| rule_char2 | 19.70% | 35.61% | 0.3028 | 0.00pp |
| tfidf_cosine | 19.70% | 37.12% | 0.3149 | 0.00pp |
| tfidf_pair | 4.55% | 15.91% | 0.1496 | 0.00pp |
| cc_zeroshot | 2.27% | 4.55% | 0.0858 | 0.00pp |
| cc_maxsim | 17.42% | 29.55% | 0.2827 | 0.00pp |
| e5_zeroshot | 24.24% | 39.39% | 0.3567 | 0.00pp |
| random_frozen | 0.30% | 2.12% | 0.0526 | 0.37pp |
| random_train | 0.76% | 1.36% | 0.0437 | 0.00pp |
| cc_frozen | 1.67% | 2.42% | 0.0653 | 0.30pp |
| cc_train | 1.52% | 2.27% | 0.0589 | 0.00pp |
| shuffled_frozen | 0.15% | 0.91% | 0.0442 | 0.30pp |
| shuffled_train | 0.76% | 1.36% | 0.0474 | 0.48pp |
| e5_proj | 5.91% | 17.27% | 0.1745 | 1.30pp |

## Group cross-validation (48 held-out indicators inside the training split)

| Condition | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| rule_char2 | 12.94% | 22.73% | 0.2320 |
| tfidf_cosine | 18.71% | 28.32% | 0.2868 |
| tfidf_pair | 2.97% | 7.69% | 0.1011 |
| cc_zeroshot | 8.39% | 17.66% | 0.1818 |
| cc_maxsim | 10.66% | 19.06% | 0.2092 |
| e5_zeroshot | 23.78% | 37.76% | 0.3584 |
| random_frozen | 1.01% | 2.73% | 0.0599 |
| random_train | 0.77% | 2.24% | 0.0491 |
| cc_frozen | 3.25% | 8.74% | 0.1100 |
| cc_train | 2.66% | 4.55% | 0.0761 |
| shuffled_frozen | 0.35% | 1.64% | 0.0511 |
| shuffled_train | 1.36% | 2.27% | 0.0513 |
| e5_proj | 2.24% | 9.72% | 0.1143 |

## Paired group-bootstrap contrasts (95% percentile interval)

| Treatment - control | Protocol | Recall@1 delta | 95% interval | MRR delta | 95% interval |
|---|---|---:|---|---:|---|
| tfidf_pair - tfidf_cosine | cv | -15.73pp | [-22.54, -10.00] | -0.1857 | [-0.2549, -0.1263] |
| tfidf_pair - tfidf_cosine | val | -15.15pp | [-29.55, -3.79] | -0.1653 | [-0.2992, -0.0554] |
| cc_maxsim - cc_zeroshot | cv | +2.27pp | [-2.29, +7.04] | +0.0274 | [-0.0217, +0.0774] |
| cc_maxsim - cc_zeroshot | val | +15.15pp | [+4.55, +31.06] | +0.1969 | [+0.0895, +0.3343] |
| cc_maxsim - tfidf_cosine | cv | -8.04pp | [-12.86, -3.67] | -0.0776 | [-0.1252, -0.0348] |
| cc_maxsim - tfidf_cosine | val | -2.27pp | [-9.85, +6.06] | -0.0322 | [-0.1178, +0.0665] |
| cc_train - random_train | cv | +2.10pp | [+0.53, +3.87] | +0.0322 | [+0.0166, +0.0505] |
| cc_train - random_train | val | +0.76pp | [+0.00, +2.27] | +0.0182 | [-0.0020, +0.0492] |
| cc_frozen - random_frozen | cv | +2.10pp | [+0.35, +4.20] | +0.0521 | [+0.0283, +0.0785] |
| cc_frozen - random_frozen | val | +2.27pp | [+0.00, +6.82] | +0.0190 | [-0.0131, +0.0704] |
| cc_frozen - shuffled_frozen | cv | +2.80pp | [+0.88, +5.21] | +0.0605 | [+0.0348, +0.0899] |
| cc_frozen - shuffled_frozen | val | +2.27pp | [+0.00, +6.82] | +0.0239 | [+0.0013, +0.0629] |
| cc_train - shuffled_train | cv | +1.92pp | [+0.52, +3.82] | +0.0305 | [+0.0139, +0.0502] |
| cc_train - shuffled_train | val | +0.00pp | [+0.00, +0.00] | +0.0109 | [+0.0007, +0.0228] |
| cc_train - cc_frozen | cv | -0.17pp | [-1.92, +1.40] | -0.0313 | [-0.0537, -0.0115] |
| cc_train - cc_frozen | val | -0.76pp | [-2.27, +0.00] | -0.0074 | [-0.0153, -0.0002] |
| e5_proj - random_train | cv | +2.10pp | [-0.18, +4.89] | +0.0735 | [+0.0434, +0.1091] |
| e5_proj - random_train | val | +6.06pp | [+0.00, +14.39] | +0.1334 | [+0.0527, +0.2287] |

Seeds: 0, 101, 13, 137, 42, 73

## Max-similarity control (random token vectors)

| Protocol | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| cv | 9.09% | 18.18% | 0.1787 |
| val | 17.93% | 31.31% | 0.2824 |

Same scorer and vocabulary with random vectors; the remaining score is
shared surface tokens comparing to themselves.

## Vector scaling

- cc.ko.300 median token norm: 0.9787
- random median token norm: 1.7303
- common factor applied to cc vectors: 1.7697
- shuffle seed: 20261121

## Source model

- file: 7243669409 bytes, sha256 a021ebbd5521ca4b3b33425fc25dacd60e4a795041d6f785997800d32a58acd7
- dimension: 300
