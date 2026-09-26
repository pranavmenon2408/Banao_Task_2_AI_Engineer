"""Walk-forward validation: for each month M, train on everything labelled before M, score M.

Nothing from month M (labels or label-derived history) is visible when scoring it. Writes
reports/walk_forward_months.csv, reports/walk_forward_oof.csv and reports/evaluation.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kestrel_fraud import metrics, model
from kestrel_fraud.config import DESK_CAPACITY, POLICY_CHANGE, REPORT_DIR
from kestrel_fraud.data import load_claims, load_reference
from kestrel_fraud.features import build_features

FOLD_MONTHS = pd.period_range("2025-10", "2026-06", freq="M")


def baseline_scores(df, rng):
    """Simple alternatives the model has to beat."""
    return {
        "random": rng.random(len(df)),
        "newest_partner_first": -df["partner_age_days"].values,  # Ritu's hypothesis
        "largest_amount_first": df["claim_amount_inr"].values,
        "rule_near_limit_uninspected": (
            2 * df["near_limit"] + (1 - df["partner_inspected"]) + df["customer_prior_claims"] + df["amount_to_list"]
        ).values,
    }


def top_k_saving(y, amount, score, k=DESK_CAPACITY):
    top = np.argsort(-score)[:k]
    return metrics.realised_saving(y, amount, top), y[top].mean()


def run_fold(claims, partners, products, month, params=None, zoho="weighted"):
    start = month.start_time
    feats = build_features(claims, partners, products, label_cutoff=start)
    train = feats[(feats["ts"] < start) & feats["is_fraud"].notna()]
    if zoho == "drop":
        train = train[train["source"] != "legacy_zoho"]
        if train["is_fraud"].sum() < 10:  # not enough CRM history yet to train on
            return test_rows(feats, month), None
    test = test_rows(feats, month)
    m = model.fit(train, params) if zoho != "unweighted" else _fit_unweighted(train, params)
    return test, model.predict(m, test)


def test_rows(feats, month):
    return feats[(feats["ts"].dt.to_period("M") == month) & feats["is_fraud"].notna()]


def _fit_unweighted(train, params):
    import lightgbm as lgb

    from kestrel_fraud.config import LGBM_PARAMS
    from kestrel_fraud.features import FEATURES

    m = lgb.LGBMClassifier(**{**LGBM_PARAMS, **(params or {})})
    m.fit(train[FEATURES], train["is_fraud"].astype(int))
    return m


def main():
    partners, products = load_reference()
    train_raw, _ = load_claims()
    rng = np.random.default_rng(0)
    rows, oof, sens = [], [], []

    for month in FOLD_MONTHS:
        test, p = run_fold(train_raw, partners, products, month)
        y, amt = test["is_fraud"].astype(int).values, test["claim_amount_inr"].values
        r = {"month": str(month), **metrics.month_report(y, p, amt)}
        for name, s in baseline_scores(test, rng).items():
            r[f"base_{name}_net_inr"], r[f"base_{name}_p_at_k"] = top_k_saving(y, amt, s)
        rows.append(r)
        oof.append(
            test[
                [
                    "claim_id",
                    "ts",
                    "partner_id",
                    "claim_amount_inr",
                    "is_fraud",
                    "auto_approved",
                    "near_limit",
                    "partner_inspected",
                    "partner_age_days",
                    "p_labelled_n",
                ]
            ].assign(p=p)
        )
        for variant in ["drop", "unweighted"]:
            _, pv = run_fold(train_raw, partners, products, month, zoho=variant)
            if pv is None:
                continue
            sens.append(
                {
                    "month": str(month),
                    "variant": variant,
                    "net_inr": metrics.month_report(y, pv, amt)["desk_net_inr"],
                    "pr_auc": metrics.month_report(y, pv, amt)["pr_auc"],
                }
            )
        print(
            f"{month}: n={r['n']} fraud={r['n_fraud']} desk caught {r['desk_caught']}/{r['desk_reviewed']} "
            f"net Rs {r['desk_net_inr']:,.0f}  PR-AUC {r['pr_auc']:.3f}"
        )

    months = pd.DataFrame(rows)
    oof = pd.concat(oof, ignore_index=True)
    REPORT_DIR.mkdir(exist_ok=True)
    months.to_csv(REPORT_DIR / "walk_forward_months.csv", index=False)
    oof.to_csv(REPORT_DIR / "walk_forward_oof.csv", index=False)

    summary = {}
    for label, mask in [
        ("all_folds", np.ones(len(oof), bool)),
        ("pre_policy", oof["ts"] < POLICY_CHANGE),
        ("post_policy", oof["ts"] >= POLICY_CHANGE),
    ]:
        o = oof[mask]
        y, p = o["is_fraud"].astype(int).values, o["p"].values
        f1 = metrics.best_f1(y, p)
        mm = months[months["month"].isin(o["ts"].dt.to_period("M").astype(str).unique())]
        summary[label] = {
            "claims": len(o),
            "frauds": int(y.sum()),
            "pooled_pr_auc": float(metrics.average_precision_score(y, p)),
            "pooled_roc_auc": float(metrics.roc_auc_score(y, p)),
            "mean_monthly_pr_auc": float(mm["pr_auc"].mean()),
            "desk_net_inr_per_month": float(mm["desk_net_inr"].mean()),
            "desk_precision": float(mm["desk_caught"].sum() / max(mm["desk_reviewed"].sum(), 1)),
            "desk_recall": float(mm["desk_caught"].sum() / mm["n_fraud"].sum()),
            "desk_inr_recall": float(mm["desk_inr_caught"].sum() / mm["fraud_inr"].sum()),
            "desk_reviews_per_month": float(mm["desk_reviewed"].mean()),
            "precision_at_40": float(mm["p_at_k"].mean()),
            "recall_at_40": float(mm["r_at_k"].mean()),
            "best_f1": f1,
            "confusion_at_best_f1": metrics.confusion_at(y, p, f1["threshold"]),
            "accuracy_if_flag_nothing": float(1 - y.mean()),
            "baselines_net_inr_per_month": {
                c.replace("base_", "").replace("_net_inr", ""): float(mm[c].mean())
                for c in mm.columns
                if c.endswith("_net_inr") and c.startswith("base_")
            },
        }
    s = pd.DataFrame(sens)
    summary["zoho_sensitivity"] = {
        str(k): v for k, v in s.groupby("variant")[["net_inr", "pr_auc"]].mean().to_dict("index").items()
    }
    summary["zoho_sensitivity"]["weighted(default)"] = {
        "net_inr": float(months["desk_net_inr"].mean()),
        "pr_auc": float(months["pr_auc"].mean()),
    }
    (REPORT_DIR / "evaluation.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
