# Kestrel Home — warranty claim review

Scores warranty claims for fraud risk before payout. It decides which claims are worth sending to the
investigation desk, which can review 40 a month, and explains each score in plain English.

No paid API or model key is used anywhere.

## Setup (clean machine, Python 3.11+)

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
  - **Score a claim:** calls the API.
  - **Jul-Sep predictions:** the scored test set, the desk worklist and the partner watchlist.
  - **How well it works:** walk-forward results, a calibration check and the model comparison.
  - If the screen runs on another host or port from the API, set `KESTREL_API_URL`.
- A fallback single-page screen is also served by the API itself at http://localhost:8000.
- API: `POST /score` with one claim as JSON. Interactive docs are at http://localhost:8000/docs.
- `GET /health` reports whether the model and the claims history loaded.

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
