# Area classification plus developmental level

The supplied dataset does not stop at an area label. Each KICCE row also carries a
developmental `level` 1-4. This experiment adds that second output to the predictor
and tests the risk that a score behaves like a hidden class weight rather than an
ordinal judgment.

## Data structure that shapes the design

- One `group_id` is one assessment indicator (`item_text`), and one indicator belongs
  to one of the five areas. A group has 12 vignettes: three per level 1-4.
- Levels are perfectly balanced inside every group and across areas: each area has
  mean level 2.50 in training. The level is therefore **not** a property of the area.
- Train 572 / validation 132 KICCE rows; 11 validation groups, 33 rows per level.
  Casebook rows (94 train / 26 validation) have no level.
- The 666/158 split is grouped, so a level model never sees the indicator it is
  scored on. This is the hard version of the task.

## System

Two heads over the same syllable 2-5 character TF-IDF matrix:

- **Area**: `LinearSVC(C=10)` wrapped in `CalibratedClassifierCV` (sigmoid) with
  group-aware folds. Calibration is what enables top-2 and confidence output; it also
  happens to raise validation accuracy from 68.99% to 71.52%.
- **Level**: multinomial `LogisticRegression(C=1)` trained on KICCE rows only,
  emitting a distribution over 1-4, an argmax label and an expected value.

The heads are parallel by design. The experiment tests whether that is correct.

## Results

Area, validation 158 rows:

| Setting | n | Accuracy | Macro-F1 | Top-2 |
| --- | ---: | ---: | ---: | ---: |
| SVM decision reference | 158 | 68.99% | 0.680 | 87.34% |
| Calibrated SVM (system) | 158 | **71.52%** | 0.709 | **89.24%** |
| Calibrated SVM, KICCE rows only | 132 | 71.21% | 0.697 | 90.15% |
| SVM trained on KICCE rows only | 132 | 61.36% | — | — |

Training on the casebook rows as well is worth about 10 points on the KICCE subset.
The level head is trained on KICCE only, so the stacking comparisons below use the
61.36% KICCE-only baseline, not the 71.21% all-source system.

Level, validation 132 KICCE rows:

| Setting | Exact | Within ±1 | MAE | QWK |
| --- | ---: | ---: | ---: | ---: |
| Multinomial argmax | **56.06%** | **85.61%** | **0.636** | **0.576** |
| Expected value, rounded | 41.67% | — | 0.667 | 0.466 |
| Global mean | 25.00% | — | 1.000 | 0.000 |
| Oracle area mean | 25.00% | — | 1.000 | 0.000 |
| Length only | 40.15% | — | 1.008 | 0.298 |

Per level: level 1 75.8%, level 2 66.7%, level 3 21.2%, level 4 60.6%.
Within-group Spearman between true level and predicted score: mean 0.703
(min 0.410, max 0.928) over 11 groups.

## Anti-confound checks

| Check | Result | Reading |
| --- | ---: | --- |
| Area from gold level only | 27.3% (chance 20%) | the level is not an area code |
| Level mean by area | 2.50 for all five | the score cannot act as a per-class prior |
| Area, KICCE-only baseline | 61.36% | reference for the next two rows |
| Area + level probabilities as features | 62.12% | +1 row, noise |
| Area + expected level as a feature | 62.12% | +1 row, noise |
| Area trained with `sample_weight=level` | 60.61% | the weight form *hurts* |
| Level MAE, text only | 0.811 | reference for the next row |
| Level MAE + area probabilities | 0.811 | conditioning on area adds nothing |
| Joint exact (area and level) | 40.91% | honest two-output number |

The score is orthogonal to the area in both directions. Feeding it back into the
classifier neither helps nor creates a hidden weighting effect: as a feature it moves
one item, as a training weight it loses one item. This is the concrete answer to the
"score as a learned weight" risk: **do not put the score on the classification path**.

## Interpretation

1. The level is learnable from text on unseen indicators: 56% exact and 86% within one
   level, against 25% for the mean, with QWK 0.58. Length alone explains the mode but
   not the distance (MAE 1.008), so the character n-grams carry real content.
2. The area and the level are independent targets. Classify first and score second is
   fine as a presentation order, but the implementation should be two parallel heads.
   Conditioning the level on the area buys nothing because every area contains the full
   level range by construction.
3. The weak point is level 3: 21% exact and systematically under-predicted (mean
   expected value 2.61). The 2-to-3 and 3-to-4 boundaries are fuzzy in the vignettes.
   Report the level with a ±1 band and its distribution rather than a single hard number.
4. If the product needs both outputs, 40.9% joint exact is the honest number. If it
   needs the area only, the calibrated area head is the better single output.

## Recommendation for the predictor

- Keep the area head as the classifier: calibrated SVM, top-1 plus top-2 and
  probability. Do not feed the level into it.
- Add the level head only for KICCE-schema rows (it has no training data for casebook
  rows) and return `{argmax, 1-4 distribution, expected value}`. Present the ±1 band.
- Gate the level on its own confidence: mean max probability is 0.366, so a hard label
  below roughly 0.4 should be shown as a range, not a point.
- The current scale is **1-4**, not 1-5. Do not stretch it to five points without new
  labels; the cut points would be invented.

## Limits

- 11 validation groups and 33 rows per level: one group shift moves accuracy by about
  2.5 points. Treat differences under ~3 points as noise.
- Absolute levels are indicator-relative. The model learns a cross-indicator
  sophistication axis (within-group Spearman 0.70), but level 3 of one indicator is not
  guaranteed to mean the same as level 3 of another.
- Character TF-IDF only. A fine-tuned contextual encoder is the next representation
  step; this experiment fixes the target structure, not the ceiling.

## Reproduction

```sh
export UV_PROJECT_ENVIRONMENT=.venv-semantic
uv run python hierarchical_experiment.py
```

Writes `results_hierarchical/metrics.csv`, `predictions.csv` (per-row area top-2,
confidence, level label, expected level and confidence) and `metadata.json` (split
hashes, counts, diagnostics). On a memory-constrained machine prefix the command with
`OPENBLAS_NUM_THREADS=2`.
