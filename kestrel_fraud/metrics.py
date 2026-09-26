"""Business and statistical metrics for a month of scored claims."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .config import CONTACT_INR, DESK_CAPACITY, GOODWILL_INR


def expected_value(p, amount):
    """Expected rupees saved by sending a claim to the desk rather than paying it (policy s4)."""
    return p * amount - (1 - p) * GOODWILL_INR - CONTACT_INR


def desk_queue(p, amount, k=DESK_CAPACITY):
    """Indices the desk should review this month: top-k by expected value, only if EV > 0."""
    ev = expected_value(p, amount)
    order = np.argsort(-ev)[:k]
    return order[ev[order] > 0]


def realised_saving(y, amount, picked):
    """Rupees actually saved by reviewing `picked`: fraud stopped minus goodwill and contact costs."""
    y, amount = np.asarray(y), np.asarray(amount)
    caught = amount[picked][y[picked] == 1].sum()
    false_holds = (y[picked] == 0).sum()
    return caught - false_holds * GOODWILL_INR - len(picked) * CONTACT_INR


def month_report(y, p, amount, k=DESK_CAPACITY):
    y, p, amount = np.asarray(y, int), np.asarray(p, float), np.asarray(amount, float)
    n_fraud, fraud_inr = int(y.sum()), float(amount[y == 1].sum())
    r = {"n": len(y), "n_fraud": n_fraud, "fraud_inr": fraud_inr}

    # desk as it would actually run (EV-ranked, capped at k)
    q = desk_queue(p, amount, k)
    r["desk_reviewed"] = len(q)
    r["desk_caught"] = int(y[q].sum())
    r["desk_precision"] = y[q].mean() if len(q) else np.nan
    r["desk_recall"] = y[q].sum() / n_fraud if n_fraud else np.nan
    r["desk_inr_caught"] = float(amount[q][y[q] == 1].sum())
    r["desk_inr_recall"] = r["desk_inr_caught"] / fraud_inr if fraud_inr else np.nan
    r["desk_net_inr"] = realised_saving(y, amount, q)

    # plain top-k by probability (what predictions.csv ranks by)
    top = np.argsort(-p)[:k]
    r["p_at_k"] = y[top].mean()
    r["r_at_k"] = y[top].sum() / n_fraud if n_fraud else np.nan

    # threshold-free and standard classification numbers
    if 0 < n_fraud < len(y):
        r["pr_auc"] = average_precision_score(y, p)
        r["roc_auc"] = roc_auc_score(y, p)
    else:
        r["pr_auc"] = r["roc_auc"] = np.nan
    return r


def best_f1(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return {"threshold": float(thr[i]), "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}


def confusion_at(y, p, threshold):
    y, pred = np.asarray(y, int), (np.asarray(p) >= threshold).astype(int)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "accuracy": float((pred == y).mean()),
    }
