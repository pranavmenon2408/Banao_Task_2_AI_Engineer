"""Compare candidate improvements under the same walk-forward protocol as evaluate.py.

Each config runs with several seeds; a change only counts if it beats the seed-to-seed noise.
Writes reports/experiments.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kestrel_fraud import metrics, model  # noqa: E402
from kestrel_fraud.config import POLICY_CHANGE, REPORT_DIR  # noqa: E402
from kestrel_fraud.data import load_claims, load_reference  # noqa: E402
from kestrel_fraud.features import CITY, DRIFT, FEATURES, NOISY, PEER, build_features  # noqa: E402

FOLD_MONTHS = pd.period_range("2025-10", "2026-06", freq="M")
SEEDS = [7, 11, 23]


PRUNED = [f for f in FEATURES if f not in NOISY]
CONFIGS = {
    "baseline": (FEATURES, False),
    "A_prune_noisy": (PRUNED, False),
    "B_plus_drift": (FEATURES + DRIFT, False),
    "C_plus_peer": (FEATURES + PEER, False),
    "D_plus_city": (FEATURES + CITY, False),
    "E_monotone": (FEATURES, True),
    "A+B+C+E": (PRUNED + DRIFT + PEER, True),
    "E+drop_noisy": (PRUNED, True),
}


def run(folds, features, constrained, seeds, bag=False):
    """Returns out-of-fold predictions; with bag=True the seeds are averaged into one model."""
    out = []
    for month, (train, test) in folds.items():
        preds = [model.predict(model.fit(train, {"random_state": s}, features=features, constrained=constrained), test)
                 for s in seeds]
        if bag:
            out.append(test.assign(p=np.mean(preds, axis=0), seed="bag"))
        else:
            out += [test.assign(p=p, seed=s) for s, p in zip(seeds, preds)]
    return pd.concat(out)


def score(oof):
    rows = []
    for seed, o in oof.groupby("seed"):
        months = []
        for _, g in o.groupby(o["ts"].dt.to_period("M")):
            y, p, a = g["is_fraud"].astype(int).values, g["p"].values, g["claim_amount_inr"].values
            r = metrics.month_report(y, p, a)
            n = y.sum()
            r["r_precision"] = y[np.argsort(-p)[:n]].mean()
            months.append(r)
        m = pd.DataFrame(months)
        post = o["ts"] >= POLICY_CHANGE
        y = o["is_fraud"].astype(int)
        rows.append({
            "pr_auc_all": metrics.average_precision_score(y, o["p"]),
            "pr_auc_pre": metrics.average_precision_score(y[~post], o["p"][~post]),
            "pr_auc_post": metrics.average_precision_score(y[post], o["p"][post]),
            "recall_at_40": m["r_at_k"].mean(), "r_precision": m["r_precision"].mean(),
            "desk_net_inr_month": m["desk_net_inr"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    partners, products = load_reference()
    train_raw, _ = load_claims()
    folds = {}
    for month in FOLD_MONTHS:
        f = build_features(train_raw, partners, products, label_cutoff=month.start_time)
        folds[month] = (f[(f["ts"] < month.start_time) & f["is_fraud"].notna()],
                        f[(f["ts"].dt.to_period("M") == month) & f["is_fraud"].notna()])

    results = []
    for name, (features, constrained) in CONFIGS.items():
        s = score(run(folds, features, constrained, SEEDS))
        results.append({"config": name, **{f"{c}_mean": s[c].mean() for c in s}, **{f"{c}_sd": s[c].std() for c in s}})
        print(f"{name:15s} " + "  ".join(f"{c}={s[c].mean():.3f}±{s[c].std():.3f}" for c in s.columns if c != "desk_net_inr_month")
              + f"  net/mo=Rs {s['desk_net_inr_month'].mean():,.0f}")
    for name in ["E_monotone"]:
        features, constrained = CONFIGS[name]
        s = score(run(folds, features, constrained, SEEDS + [31, 47], bag=True))
        results.append({"config": f"F_bag5_{name}", **{f"{c}_mean": s[c].mean() for c in s}})
        print(f"F_bag5_{name:8s} " + "  ".join(f"{c}={s[c].mean():.3f}" for c in s.columns if c != "desk_net_inr_month")
              + f"  net/mo=Rs {s['desk_net_inr_month'].mean():,.0f}")
    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(results).round(4).to_csv(REPORT_DIR / "experiments.csv", index=False)


if __name__ == "__main__":
    main()
