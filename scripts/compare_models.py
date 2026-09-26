"""Compare model families under the same walk-forward protocol and features as the LightGBM model.

Linear models and SVMs get log-transformed counts/amounts, scaling and one-hot categoricals; their
probabilities are Platt-calibrated inside each training fold so the rupee-based desk rule is fair
to every model. Writes reports/model_comparison.csv and docs/model_comparison.md.
"""

import functools
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import FOLD_MONTHS, score

from kestrel_fraud import model as lgbm
from kestrel_fraud.config import REPORT_DIR, ROOT
from kestrel_fraud.data import load_claims, load_reference
from kestrel_fraud.features import CATEGORICAL, FEATURES, build_features

NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
SKEWED = [
    "claim_amount_inr",
    "days_since_purchase",
    "customer_prior_claims",
    "serial_prior_claims",
    "serial_prior_other_partner",
    "partner_age_days",
    "p_claims_30d",
    "p_claims_90d",
    "p_growth_30d_vs_180d",
    "p_labelled_n",
    "p_fraud_n",
    "p_fraud_90d",
]


def preprocess(scale=True):
    log_scale = [FunctionTransformer(lambda x: np.log1p(np.clip(x, 0, None)))] + ([StandardScaler()] if scale else [])
    rest = [StandardScaler()] if scale else ["passthrough"]
    return ColumnTransformer(
        [
            ("skewed", make_pipeline(*log_scale), SKEWED),
            ("numeric", make_pipeline(*rest) if scale else "passthrough", [c for c in NUMERIC if c not in SKEWED]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def calibrated(est):
    return CalibratedClassifierCV(est, method="sigmoid", cv=3)


def sk(est, scale=True):
    return lambda seed: make_pipeline(preprocess(scale), est(seed))


MODELS = {
    "Logistic regression (C=0.1)": sk(lambda s: LogisticRegression(C=0.1, max_iter=3000)),
    "Logistic regression (C=1)": sk(lambda s: LogisticRegression(C=1.0, max_iter=3000)),
    "Logistic regression, balanced (C=0.1)": sk(
        lambda s: calibrated(LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))
    ),
    "SVM linear (C=0.1)": sk(lambda s: calibrated(LinearSVC(C=0.1, class_weight="balanced", max_iter=20000))),
    "SVM RBF (C=1)": sk(
        lambda s: calibrated(SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", cache_size=1000))
    ),
    "SVM RBF (C=10)": sk(
        lambda s: calibrated(SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", cache_size=1000))
    ),
    "SVM polynomial deg 2 (C=1)": sk(
        lambda s: calibrated(SVC(kernel="poly", degree=2, C=1.0, coef0=1, class_weight="balanced", cache_size=1000))
    ),
    "SVM polynomial deg 3 (C=1)": sk(
        lambda s: calibrated(SVC(kernel="poly", degree=3, C=1.0, coef0=1, class_weight="balanced", cache_size=1000))
    ),
    "SVM sigmoid (C=1)": sk(lambda s: calibrated(SVC(kernel="sigmoid", C=1.0, class_weight="balanced", cache_size=1000))),
    "Random forest (500 trees, leaf 3)": sk(
        lambda s: RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=s), scale=False
    ),
    "Random forest (500 trees, leaf 10)": sk(
        lambda s: RandomForestClassifier(500, min_samples_leaf=10, n_jobs=-1, random_state=s), scale=False
    ),
    "Extra trees (500 trees, leaf 3)": sk(
        lambda s: ExtraTreesClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=s), scale=False
    ),
}
STOCHASTIC = {"Random forest (500 trees, leaf 3)", "Random forest (500 trees, leaf 10)", "Extra trees (500 trees, leaf 3)"}
SEEDS = [7, 11, 23]


def run_sklearn(folds, make, seeds):
    out = []
    for train, test in folds.values():
        y = train["is_fraud"].astype(int)
        for s in seeds:
            m = make(s).fit(train[FEATURES], y)
            out.append(test.assign(p=m.predict_proba(test[FEATURES])[:, 1], seed=s))
    return pd.concat(out)


def run_lgbm(folds, seeds):
    out = []
    for train, test in folds.values():
        for s in seeds:
            out.append(test.assign(p=lgbm.predict(lgbm.fit(train, {"random_state": s}), test), seed=s))
    return pd.concat(out)


def main():
    partners, products = load_reference()
    train_raw, _ = load_claims()
    folds = {}
    for month in FOLD_MONTHS:
        f = build_features(train_raw, partners, products, label_cutoff=month.start_time)
        folds[month] = (
            f[(f["ts"] < month.start_time) & f["is_fraud"].notna()],
            f[(f["ts"].dt.to_period("M") == month) & f["is_fraud"].notna()],
        )

    only = sys.argv[1:]  # optional substring filter to rerun a subset
    rows = []
    runs: dict[str, Callable[[], pd.DataFrame]] = {"LightGBM, monotone (chosen)": lambda: run_lgbm(folds, SEEDS)}
    runs.update(
        {
            name: functools.partial(run_sklearn, folds, make, SEEDS if name in STOCHASTIC else [0])
            for name, make in MODELS.items()
        }
    )
    for name, fn in runs.items():
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        t0 = time.time()
        s = score(fn())
        secs = time.time() - t0
        rows.append(
            {
                "model": name,
                "seeds": len(s),
                **{c: s[c].mean() for c in s},
                "pr_auc_post_sd": s["pr_auc_post"].std(),
                "seconds": secs,
            }
        )
        print(
            f"{name:40s} PR-AUC all={s['pr_auc_all'].mean():.3f} pre={s['pr_auc_pre'].mean():.3f} "
            f"post={s['pr_auc_post'].mean():.3f}  R@40={s['recall_at_40'].mean():.3f}  "
            f"R-prec={s['r_precision'].mean():.3f}  net/mo=Rs {s['desk_net_inr_month'].mean():,.0f}  ({secs:.0f}s)",
            flush=True,
        )

    res = pd.DataFrame(rows).sort_values("pr_auc_post", ascending=False)
    REPORT_DIR.mkdir(exist_ok=True)
    res.round(4).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    md = [
        "| Model | PR-AUC all | PR-AUC pre-May | PR-AUC post-May | Recall@40 | R-precision "
        "| Desk net Rs/month | Train+score time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in res.itertuples():
        md.append(
            f"| {r.model} | {r.pr_auc_all:.3f} | {r.pr_auc_pre:.3f} | {r.pr_auc_post:.3f} | {r.recall_at_40:.3f} | "
            f"{r.r_precision:.3f} | {r.desk_net_inr_month:,.0f} | {r.seconds:.0f}s |"
        )
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "model_comparison.md").write_text(
        "# Model comparison (walk-forward, 9 monthly folds Oct 2025 - Jun 2026)\n\n"
        "Same features and folds for every model. Stochastic models averaged over 3 seeds. "
        "Sorted by post-May PR-AUC, the regime the test set is in.\n\n" + "\n".join(md) + "\n"
    )


if __name__ == "__main__":
    main()
