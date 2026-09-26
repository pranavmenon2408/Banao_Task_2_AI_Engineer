# Evidence: does it work, and how often does it not?

Every number here can be regenerated with `python scripts/evaluate.py`, `scripts/experiments.py` and
`scripts/compare_models.py`. The Streamlit screen's *How well it works* tab shows the same tables.

## 1. How it was tested

**Walk-forward validation, the way it would actually be used.** For each month from Oct 2025 to Jun 2026:
1. The model is trained only on claims from earlier months.
2. It then scores that month blind.
3. Partner-history features only use investigation outcomes that would have been known at the time: the claim is at least 14 days old, and it falls before the scored month.

No random split is used. A random split would let the model see the same partner's future outcomes and flatter the result.

- **Sample:** 6,540 scored claims, 95 confirmed frauds, over 9 monthly folds.
- **Before cleaning:** 12,029 training rows. There were 681 duplicate `claim_id`s from bounced-and-resubmitted claims, and all duplicates had matching labels. 215 claims were undecided at export.
- **Training set:** 11,146 labelled, de-duplicated claims, 141 of them fraud (1.27%).

## 2. Headline results

| | Before May 2026 (7 months) | After May 2026 (May–Jun) | All 9 months |
|---|---|---|---|
| Claims / frauds | 5,118 / 59 | 1,422 / 36 | 6,540 / 95 |
| PR-AUC (average precision) | **0.56** | **0.27** (June alone 0.51) | 0.47 |
| ROC-AUC | 0.83 | 0.61 (June alone 0.91) | 0.76 |
| Desk reviews per month (rupee rule) | 7.4 | 7.5 | 7.4 |
| Share of desk reviews that were fraud | 67% | 27% | 58% |
| Fraud rupees caught by the desk | **76%** | 10% | 65% |
| Net rupees saved per month by the desk | **₹33,300** | –₹800 | ₹25,800 |
| Recall in top 40 per month | 0.66 | 0.39 | 0.60 |
| Precision / recall / F1 at best-F1 cutoff | 0.78 / 0.59 / 0.67 | 0.40 / 0.47 / 0.43 | 0.67 / 0.50 / 0.57 |
| Accuracy if we flag **nothing** | 98.8% | 97.5% | 98.5% |

**The desk rule** sends a claim for review only when the expected saving is positive:
p × amount − (1 − p) × ₹380 goodwill − ₹260 contact > 0. The costs come from ops policy §4. The queue is capped at 40 a month.
The rule rarely fills the 40 slots, because for most claims a review costs more than the fraud it could stop.

**Precision@40 looks low for a structural reason.** Precision@40 can't be higher than (frauds that month) ÷ 40, and a normal month has 8–10 frauds, so the best possible value is 0.20–0.25. The fair ranking measure is precision within the top *n*, where n is that month's fraud count. It was 0.62–0.70 in most months.

## 3. Month by month

| Month | Claims | Frauds | Fraud ₹ | Desk reviews | Caught | ₹ caught | Net ₹ saved | Recall@40 | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| 2025-10 | 736 | 8 | 37,541 | 9 | 5 | 33,440 | 29,580 | 0.63 | 0.41 |
| 2025-11 | 717 | 10 | 38,381 | 7 | 4 | 17,891 | 14,931 | 0.40 | 0.37 |
| 2025-12 | 794 | 10 | 66,596 | 9 | 7 | 59,651 | 56,551 | 0.70 | 0.72 |
| 2026-01 | 730 | 8 | 47,614 | 8 | 5 | 33,293 | 30,073 | 0.75 | 0.57 |
| 2026-02 | 694 | 5 | 49,693 | 8 | 5 | 49,693 | 46,473 | 1.00 | 1.00 |
| 2026-03 | 738 | 9 | 56,429 | 7 | 6 | 42,284 | 40,084 | 0.67 | 0.68 |
| 2026-04 | 709 | 9 | 36,986 | 4 | 3 | 17,060 | 15,640 | 0.44 | 0.30 |
| **2026-05** | 709 | 14 | 21,450 | 5 | **0** | 0 | **–3,200** | **0.00** | **0.01** |
| 2026-06 | 713 | 22 | 45,673 | 10 | 4 | 6,539 | 1,659 | 0.77 | 0.51 |

Over the 9 months the desk reviewed 67 claims. **39 were fraud and 28 were genuine customers held unnecessarily** (42%).

## 4. Where it fails, and why

**Failure 1: fraud at a partner with no fraud on record.** This is about 40% of all fraud.

| Fraud type (at the time of the claim) | Frauds | Caught in top 40 | Median rank |
|---|---|---|---|
| Partner already had a confirmed fraud (before May) | 39 | 37 (95%) | 4th |
| Partner already had a confirmed fraud (after May) | 14 | 13 (93%) | 21st |
| Partner had no known fraud (before May) | 20 | 0 | 335th |
| Partner had no known fraud (after May) | 22 | 3 | 560th |

At partners with at most two frauds ever, fraud and genuine claims look the same on every column in the export:

| | Amount ÷ list price (median) | Photo | Inspected | Days since purchase (median) | Prior customer claims (mean) |
|---|---|---|---|---|---|
| Fraud | 0.26 | 76% | 82% | 191 | 0.53 |
| Genuine | 0.28 | 78% | 84% | 216 | 0.30 |

This is a data limit, not a tuning problem. Catching these cases needs new evidence: photo checks, invoice verification, or matching serials against sales records.

**Failure 2: the first month after a rule change.** On 1 May 2026, claims under ₹2,000 started being auto-approved without inspection. Fraud moved at once to small, uninspected claims from a handful of outlets. The May fold had no examples of that pattern to learn from, so it caught 0 of 14. One month later, the June fold had May's outcomes and scored PR-AUC 0.51.

**Failure 3: small-ticket fraud doesn't pay for a review.** After May, the typical fraud is ₹800–2,000. Even a correct flag saves little once ₹640 of review and goodwill cost comes off. In June the ranking was good (recall@40 0.77), but the desk only broke even. The answer is to act on the **outlet**, not on each claim (see section 7).

**What a wrong flag looks like:** a genuine large claim (₹5,000 or more) from a partner with a fraud history, especially if the customer has 2 or more prior claims. The desk holds it, the customer gets ₹380 goodwill, and we lose ₹640.

## 5. Are the scores honest? (Calibration, from walk-forward predictions)

| Predicted risk | Claims | Mean predicted | Actually fraud |
|---|---|---|---|
| under 0.5% | 4,982 | 0.2% | 0.6% |
| 0.5–1% | 856 | 0.7% | 0.5% |
| 1–2% | 376 | 1.4% | 1.6% |
| 2–5% | 184 | 3.0% | 0.5% |
| 5–10% | 60 | 6.9% | 8.3% |
| 10–20% | 15 | 14% | 13% |
| 20–40% | 19 | 28% | 47% |
| over 40% | 48 | 71% | 75% |

The scores behave like probabilities, which the rupee rule depends on. The model under-predicts in the 20–40% band. That part of the curve is based on only 19 claims.

## 6. What it was compared against

**Simple rules** (net ₹ per month from the top 40 each month):

| Rule | Before May | After May |
|---|---|---|
| **Model** (desk rule) | **+₹33,300** | –₹800 |
| Largest claims first | +₹3,200 | –₹16,500 |
| Near-limit, uninspected, repeat customer | –₹11,500 | –₹16,100 |
| Random | –₹24,300 | –₹23,500 |
| **Newest partners first** (the board's hypothesis) | **–₹25,600** | –₹22,200 |

"Newest partners first" is worse than random. After May, the fraud does sit in partners onboarded **3–8 months** earlier:
- **3–8 months old, auto-approved claims:** 27% fraud (29 of 107).
- **Older than 8 months, auto-approved claims:** under 0.2%.

But the fraud comes from 10 of the ~60 new partners, not the group as a whole.

**Other model families.** Same features, same folds; full table in `docs/model_comparison.md`.

| Model | PR-AUC all | Post-May | Net ₹/month |
|---|---|---|---|
| **LightGBM with monotone constraints (chosen)** | **0.460** | **0.267** | **25,700** |
| Logistic regression (C=1) | 0.417 | 0.265 | 22,600 |
| Random forest | 0.422 | 0.197 | 23,700 |
| Extra trees | 0.414 | 0.180 | 24,200 |
| SVM RBF (C=10) | 0.304 | 0.170 | 22,900 |
| SVM linear | 0.306 | 0.213 | 18,900 |
| SVM polynomial, degree 2 / 3 | 0.267 / 0.242 | 0.064 / 0.078 | 20,500 / 17,800 |
| SVM sigmoid | 0.037 | 0.055 | 1,800 |

**Ideas tried and thrown away.** Each was run with 3 seeds; `reports/experiments.csv` has the log. Seed noise on post-May PR-AUC is about ±0.005.

| Idea | Post-May PR-AUC | Kept? |
|---|---|---|
| Baseline LightGBM | 0.229 | – |
| **Monotone constraints** (risk can only rise with partner fraud, prior claims, amount ÷ price) | **0.267** | **yes** |
| Drop noisy serial and product features | 0.212 | no |
| Partner behaviour drift (last 30 days vs its own 5 months before) | 0.231 | no, within noise |
| Partner behaviour relative to the whole network | 0.224 | no |
| City-level fraud history | 0.231 | no, and it risks penalising whole cities |
| Average of 5 seeds | 0.267 | no gain |
| 45-day label lag (instead of 14) | 0.164 | no |
| Up-weight post-May rows ×3 | ≈ same | no |

## 7. What it should save in Jul–Sep 2026

These are model expectations, since there are no outcomes yet for these months. The *Jul–Sep predictions* tab shows them per month.

| | Jul | Aug | Sep | Per month |
|---|---|---|---|---|
| Expected fraud (₹) | 27,458 | 28,583 | 25,511 | ~27,200 |
| Desk: reviews | 8 | 11 | 11 | ~10 |
| Desk: expected net saving (₹) | 6,155 | 6,150 | 6,228 | ~6,200 |
| Top 7 outlets: claims | 31 | 29 | 37 | ~32 |
| Top 7 outlets: expected fraud (₹) | 13,048 | 15,231 | 14,693 | ~14,300 |
| Top 7 outlets: net after a ₹260 check on every claim | 4,988 | 7,691 | 5,073 | ~5,900 |

The 7 outlets are SP3286, SP3118, SP3160, SP3129, SP3232, SP3319 and SP3318. They file about 4% of claims but carry about 53% of the expected fraud rupees.
In May–June, **33 of their 58 claims were confirmed fraud**. Stopping these outlets also stops their future claims, which the per-claim numbers above don't count.

## 8. Service checks

- **The API scores match the batch pipeline.** Across 200 random test claims sent to `POST /score`, the largest difference from `predictions.csv` was 0.00005 (rounding). An earlier version was off by up to 0.006, because the API didn't load serial history for a partner's older claims. That bug was fixed.
- **The API is fast:** about 90 ms per request on a laptop, with no paid API calls.
- **It fails politely:**
  - Unknown partner or SKU: returns 422 with a readable message.
  - Missing model: returns 503 with the fix.
  - Missing claims history: it still scores and says in every response that history is missing.
  - The Streamlit screen shows a readable message if the API is down.
- **Streamlit was tested end to end with its test harness:** load, score, training with progress polling, and choosing a trained model.

## 9. Expected score on the hidden outcomes

**Metric: PR-AUC (average precision).** Fraud is 1–3% of claims, so accuracy and ROC-AUC mostly measure how well genuine claims are sorted among themselves. PR-AUC measures how cleanly fraud rises to the top, and that's what the desk uses.

**Estimate: about 0.40 (plausible range 0.25–0.55).**

Why that number:
- The test months (Jul–Sep 2026) are all after the May rule change.
- The closest rehearsal is the June fold. It had one post-May month to learn from and scored 0.51, with a 90% bootstrap interval of 0.36–0.70 (only 22 frauds).
- The final model has two post-May months, which helps.

Three things pull the estimate down:
1. The test months are 1–3 months past the last known outcome.
2. 49 test claims come from partners onboarded after June, with no history at all.
3. A new outlet could turn fraudulent during Jul–Sep, which is exactly failure 1.

The expected ROC-AUC is about 0.85.
