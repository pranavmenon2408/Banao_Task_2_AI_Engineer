# Kestrel Home — warranty claim review

Scores warranty claims for fraud risk before payout. It decides which claims are worth sending to the
investigation desk, which can review 40 a month, and explains each score in plain English.

No paid API or model key is used anywhere.

## Deliverables

| What | Where |
|---|---|
| Predictions for `test_unlabelled.csv` | [`predictions.csv`](predictions.csv) |
| Evidence that it works, and how often it doesn't | [`docs/evidence.md`](docs/evidence.md) |
| One-page memo to Ritu Deshpande | [`docs/memo_to_ritu.md`](docs/memo_to_ritu.md) |
| Model-family comparison (LR, SVMs, forests vs LightGBM) | [`docs/model_comparison.md`](docs/model_comparison.md) |
| Service: scoring API and Streamlit screen | `app/`, `ui/` (see *Run the service* below) |

## Setup (clean machine, Python 3.12+)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Put the client data pack in `data/`. This folder is gitignored and never committed.

```
data/train.csv  data/test_unlabelled.csv  data/partners.csv  data/products.csv  data/sample_submission.csv
```

## Run the service

Two terminals, both inside the activated venv:

```bash
uvicorn app.main:app --port 8000          # 1. the scoring API
streamlit run ui/streamlit_app.py          # 2. the screen -> http://localhost:8501
```

- **Screen (Streamlit)** has three tabs:
  - **Score a claim:** calls the API, with a choice of model.
  - **Jul-Sep predictions:** the scored test set, the desk worklist and the partner watchlist.
  - **Train a model:** use the Kestrel history or upload a CSV, which is checked with fix-and-re-upload feedback. Progress shows every 10%.
  - **How well it works:** walk-forward results, a calibration check and the model comparison.
  - If the screen runs on another host or port from the API, set `KESTREL_API_URL`.
- A fallback single-page screen is also served by the API itself at http://localhost:8000.
- API: `POST /score` with one claim as JSON. Interactive docs are at http://localhost:8000/docs.
- `GET /health` reports whether the model and the claims history loaded.
- `GET /models` lists the models available for scoring. `default` is the shipped model.
- `POST /datasets` uploads your own claims CSV for training. It must have the same columns as `train.csv`; `GET /datasets/template` downloads an example.
  - Every problem is reported at once, with CSV line numbers and how to fix it: missing columns, bad dates or numbers, Y/N fields, labels other than 1/0/blank, unknown partners or SKUs.
  - Fix the file and re-upload. A file that passes returns a `dataset_id`.
- `POST /train` trains a named model as a background job:
  - It uses the Kestrel history, an uploaded dataset (`dataset_id`), or both (`combine_with_history`).
  - The subset can be a date range, source system, partner type or product family.
  - `GET /train/{job_id}` polls the job. It returns progress plus train and hold-out log-loss and average precision at every 10% of boosting rounds.
  - The last month of the range is held out for these metrics. The model is then optionally refitted on the whole range.
  - Trained models are saved in `artifacts/models/<name>/`, which is gitignored. Pick one in the screen, or pass `"model": "<name>"` to `/score`.
  - Jobs are kept in memory, so restarting the server forgets them; saved models stay.

```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{"partner_id":"SP3160","sku":"KH-AF-02","product_serial":"kh-128015513","claim_amount_inr":1995,"days_since_purchase":150,"photo_attached":"N","partner_inspected":"N","customer_prior_claims":1,"submitted_at":"2026-09-20 11:30"}'
```

The response contains:
- the fraud probability
- a recommendation: `SEND TO INVESTIGATION DESK`, `PAY, BUT FLAG PARTNER FOR INSPECTION` or `PAY`
- the expected rupees saved if the claim is reviewed
- the reasons pushing risk up and down

The service still starts if something is missing:
- **No trained model:** `/health` and `/score` explain what is missing and how to fix it.
- **No claims history in `data/`:** it still scores, but partner-history features are empty, and every response says so.

## Reproduce the model and predictions

```bash
python scripts/evaluate.py   # walk-forward validation -> reports/evaluation.json, reports/walk_forward_*.csv
python scripts/experiments.py      # feature/regularisation ideas vs seed noise -> reports/experiments.csv
python scripts/compare_models.py   # LR / SVM (linear, RBF, poly, sigmoid) / RF / ExtraTrees vs LightGBM -> docs/model_comparison.md
python scripts/train.py      # final model -> artifacts/, predictions.csv, reports/desk_queue.csv, reports/partner_watchlist.csv
```

## How it works

| Piece | File |
|---|---|
| Cleaning: dedupe resubmitted claims, normalise hand-typed serials | `kestrel_fraud/data.py` |
| Time-safe features: each claim only sees claims before it, and outcomes known 14+ days before it | `kestrel_fraud/features.py` |
| LightGBM model; Zoho negatives down-weighted because Zoho exported undecided cases as 0 | `kestrel_fraud/model.py` |
| Costs from ops policy §4: expected saving = p × amount − (1−p) × ₹380 − ₹260 | `kestrel_fraud/metrics.py` |
| Plain-English reasons from per-feature SHAP contributions | `kestrel_fraud/explain.py` |
| Single-claim scoring, using the same feature code as training | `kestrel_fraud/scorer.py`, `app/` |

Partners are described by their behaviour (recent volume, share of small or uninspected claims, confirmed frauds),
never by their ID. This lets the model score outlets it has never seen.

## Development checks

```bash
pip install -r requirements-dev.txt
ruff check .            # lint
ruff format --check .   # formatting
mypy                    # type checks (config in pyproject.toml)
pytest -q               # tests that need no client data
```

The same four checks run in GitHub Actions on every push (`.github/workflows/ci.yml`).
The repository contains no client data, so CI also confirms that the service starts and explains itself when `data/` is missing.
