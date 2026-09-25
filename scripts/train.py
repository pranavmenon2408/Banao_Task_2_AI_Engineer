"""Fit the final model on all labelled claims and score test_unlabelled.csv.

Outputs:
  predictions.csv                     claim_id, score (fraud probability) in sample_submission order
  artifacts/model.txt, meta.json      the booster and what it was trained on (used by the service)
  reports/desk_queue.csv              per month, the claims worth sending to the 40-a-month desk
  reports/partner_watchlist.csv       outlets whose recent claims score highest
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kestrel_fraud import metrics, model  # noqa: E402
from kestrel_fraud.config import (ARTIFACT_DIR, DATA_DIR, DESK_CAPACITY, LABEL_LAG_DAYS,  # noqa: E402
                                  LGBM_PARAMS, REPORT_DIR, ROOT)
from kestrel_fraud.data import load_claims, load_reference  # noqa: E402
from kestrel_fraud.features import FEATURES, build_features  # noqa: E402


def main():
    partners, products = load_reference()
    train_raw, test_raw = load_claims()
    feats = build_features(pd.concat([train_raw, test_raw], ignore_index=True), partners, products)
    is_test = feats["claim_id"].isin(test_raw["claim_id"])
    train, test = feats[~is_test], feats[is_test].copy()

    m = model.fit(train)
    test["score"] = model.predict(m, test)

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    pred = sample[["claim_id"]].merge(test[["claim_id", "score"]], on="claim_id", how="left")
    assert pred["score"].notna().all() and len(pred) == len(sample), "every test claim must be scored"
    pred.to_csv(ROOT / "predictions.csv", index=False)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    m.booster_.save_model(str(ARTIFACT_DIR / "model.txt"))
    labelled = train[train["is_fraud"].notna()]
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps({
        "features": FEATURES, "params": LGBM_PARAMS, "label_lag_days": LABEL_LAG_DAYS,
        "trained_on": {"claims": len(labelled), "frauds": int(labelled["is_fraud"].sum()),
                       "from": str(labelled["ts"].min()), "to": str(labelled["ts"].max())},
        "base_rate": float(labelled["is_fraud"].mean()),
    }, indent=2))

    # The desk's actual worklist: per month, top-40 by expected rupees saved, only where positive
    test["expected_saving_inr"] = metrics.expected_value(test["score"].values, test["claim_amount_inr"].values)
    test["month"] = test["ts"].dt.to_period("M").astype(str)
    queue = []
    for _, g in test.groupby("month"):
        idx = metrics.desk_queue(g["score"].values, g["claim_amount_inr"].values, DESK_CAPACITY)
        queue.append(g.iloc[idx])
    queue = pd.concat(queue)[["month", "claim_id", "partner_id", "sku", "claim_amount_inr", "score", "expected_saving_inr"]]
    REPORT_DIR.mkdir(exist_ok=True)
    queue.round(3).to_csv(REPORT_DIR / "desk_queue.csv", index=False)

    # Outlets to take off auto-approval: where the expected fraud rupees concentrate
    test["expected_fraud_inr"] = test["score"] * test["claim_amount_inr"]
    watch = (test.groupby("partner_id")
             .agg(claims=("claim_id", "size"), mean_score=("score", "mean"),
                  expected_fraud_inr=("expected_fraud_inr", "sum"), auto_approved_share=("auto_approved", "mean"),
                  known_frauds_before_jul=("p_fraud_n", "max"), partner_age_days=("partner_age_days", "max"))
             .sort_values("expected_fraud_inr", ascending=False))
    watch = watch.merge(partners, left_index=True, right_on="partner_id").set_index("partner_id")
    watch.head(25).round(3).to_csv(REPORT_DIR / "partner_watchlist.csv")

    print(f"trained on {len(labelled)} labelled claims ({int(labelled['is_fraud'].sum())} fraud)")
    print(f"scored {len(pred)} test claims; mean score {pred['score'].mean():.4f}; "
          f"expected frauds in test ~{pred['score'].sum():.0f}")
    print(f"desk queue: {queue.groupby('month').size().to_dict()} claims, "
          f"expected saving Rs {queue['expected_saving_inr'].sum():,.0f}")
    print(watch.head(10)[["city", "partner_type", "claims", "mean_score", "expected_fraud_inr", "known_frauds_before_jul"]].round(3).to_string())


if __name__ == "__main__":
    main()
