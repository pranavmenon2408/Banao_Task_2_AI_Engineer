"""Time-safe feature engineering.

Every feature for a claim at time t is computed only from claims submitted before t, and
label-based features only from outcomes that would have been known then (claims older than
LABEL_LAG_DAYS and before `label_cutoff`). Raw partner_id and city are deliberately NOT
model inputs: partners are described by their behaviour so the model works for outlets it
has never seen.
"""
import numpy as np
import pandas as pd

from .config import (AUTO_APPROVE_LIMIT, LABEL_LAG_DAYS, METRO_CITIES, NEAR_LIMIT_BAND,
                     POLICY_CHANGE)

DAY = np.timedelta64(1, "D")

CATEGORICAL = ["partner_type", "family"]
FEATURES = [
    # the claim itself
    "claim_amount_inr", "amount_to_list", "days_since_purchase", "warranty_used_frac",
    "customer_prior_claims", "photo_attached", "partner_inspected", "inspector_note_missing",
    "serial_messy", "auto_approved", "near_limit", "family",
    # the unit
    "serial_prior_claims", "serial_prior_other_partner",
    # the partner (behaviour, not identity)
    "partner_type", "partner_metro", "partner_age_days",
    "p_claims_30d", "p_claims_90d", "p_growth_30d_vs_180d",
    "p_small_share_30d", "p_near_limit_share_30d", "p_uninspected_share_30d",
    "p_mean_amount_to_list_90d", "p_serial_repeat_share_90d", "p_prior_claims_mean_90d",
    "p_labelled_n", "p_fraud_n", "p_fraud_90d", "p_fraud_rate_smoothed",
]


def _claim_level(df, partners, products):
    df = df.merge(products[["sku", "family", "list_price_inr", "warranty_months"]], on="sku", how="left")
    df = df.merge(partners[["partner_id", "city", "onboarded_date", "partner_type"]], on="partner_id", how="left")
    amt = df["claim_amount_inr"]
    df["amount_to_list"] = amt / df["list_price_inr"]
    df["warranty_used_frac"] = df["days_since_purchase"] / (df["warranty_months"] * 30.44)
    df["photo_attached"] = (df["photo_attached"] == "Y").astype(int)
    df["partner_inspected"] = (df["partner_inspected"] == "Y").astype(int)
    df["inspector_note_missing"] = df["inspector_note"].isna().astype(int)
    post = df["ts"] >= pd.Timestamp(POLICY_CHANGE)
    df["auto_approved"] = (post & (amt < AUTO_APPROVE_LIMIT)).astype(int)
    df["near_limit"] = ((amt >= AUTO_APPROVE_LIMIT - NEAR_LIMIT_BAND) & (amt < AUTO_APPROVE_LIMIT)).astype(int)
    df["partner_metro"] = df["city"].isin(METRO_CITIES).astype(int)
    df["partner_age_days"] = (df["ts"] - df["onboarded_date"]) / DAY
    df["is_small"] = (amt < AUTO_APPROVE_LIMIT).astype(int)
    return df


# candidate features evaluated in scripts/experiments.py; only promoted into FEATURES if they help
DRIFT = ["drift_small_share", "drift_near_limit_share", "drift_uninspected_share", "drift_amount_to_list"]
PEER = ["rel_small_share", "rel_near_limit_share", "rel_uninspected_share"]
CITY = ["city_fraud_rate"]
NOISY = ["serial_prior_claims", "serial_prior_other_partner", "p_serial_repeat_share_90d", "serial_messy", "family"]


def _window_sum(times, values, t_end, days):
    """Sum of values over (t_end - days, t_end) for sorted times - strictly before t_end."""
    csum = np.concatenate([[0.0], np.cumsum(values)])
    hi = np.searchsorted(times, t_end, side="left")
    lo = np.searchsorted(times, t_end - np.timedelta64(days, "D"), side="left")
    return csum[hi] - csum[lo]


def _partner_history(df, label_cutoff, prior_rate, label_lag_days, prior_weight=20.0):
    out = {c: np.zeros(len(df)) for c in [
        "p_claims_30d", "p_claims_90d", "p_claims_180d", "p_small_30d", "p_near_30d",
        "p_uninsp_30d", "p_ratio_sum_90d", "p_serial_rep_90d", "p_prior_sum_90d",
        "p_labelled_n", "p_fraud_n", "p_fraud_90d",
        "p_small_180d", "p_near_180d", "p_uninsp_180d", "p_ratio_sum_180d"]}
    cutoff = np.datetime64(label_cutoff) if label_cutoff is not None else None
    for _, idx in df.groupby("partner_id").indices.items():
        g = df.iloc[idx]
        t = g["ts"].values
        ones = np.ones(len(g))
        out["p_claims_30d"][idx] = _window_sum(t, ones, t, 30)
        out["p_claims_90d"][idx] = _window_sum(t, ones, t, 90)
        out["p_claims_180d"][idx] = _window_sum(t, ones, t, 180)
        out["p_small_30d"][idx] = _window_sum(t, g["is_small"].values, t, 30)
        out["p_near_30d"][idx] = _window_sum(t, g["near_limit"].values, t, 30)
        out["p_uninsp_30d"][idx] = _window_sum(t, 1 - g["partner_inspected"].values, t, 30)
        out["p_ratio_sum_90d"][idx] = _window_sum(t, g["amount_to_list"].values, t, 90)
        out["p_serial_rep_90d"][idx] = _window_sum(t, (g["serial_prior_claims"].values > 0).astype(float), t, 90)
        out["p_prior_sum_90d"][idx] = _window_sum(t, g["customer_prior_claims"].values.astype(float), t, 90)
        out["p_small_180d"][idx] = _window_sum(t, g["is_small"].values, t, 180)
        out["p_near_180d"][idx] = _window_sum(t, g["near_limit"].values, t, 180)
        out["p_uninsp_180d"][idx] = _window_sum(t, 1 - g["partner_inspected"].values, t, 180)
        out["p_ratio_sum_180d"][idx] = _window_sum(t, g["amount_to_list"].values, t, 180)

        # outcomes known by the time of each claim
        lab = g["is_fraud"].notna().to_numpy(copy=True)
        if cutoff is not None:
            lab &= t < cutoff
        lt, ly = t[lab], g["is_fraud"].values[lab]
        known_until = t - np.timedelta64(label_lag_days, "D")
        if cutoff is not None:
            known_until = np.minimum(known_until, cutoff)
        k = np.searchsorted(lt, known_until, side="left")
        out["p_labelled_n"][idx] = k
        cum_fraud = np.concatenate([[0.0], np.cumsum(ly)])
        out["p_fraud_n"][idx] = cum_fraud[k]
        # frauds among outcomes known in the last 90 days: catches partners that turned recently
        k90 = np.searchsorted(lt, known_until - np.timedelta64(90, "D"), side="left")
        out["p_fraud_90d"][idx] = cum_fraud[k] - cum_fraud[k90]

    f = pd.DataFrame(out, index=df.index)
    n30, n90 = f["p_claims_30d"], f["p_claims_90d"]
    f["p_growth_30d_vs_180d"] = n30 / (f["p_claims_180d"] / 6 + 1)
    f["p_small_share_30d"] = f["p_small_30d"] / n30.clip(lower=1)
    f["p_near_limit_share_30d"] = f["p_near_30d"] / n30.clip(lower=1)
    f["p_uninspected_share_30d"] = f["p_uninsp_30d"] / n30.clip(lower=1)
    f["p_mean_amount_to_list_90d"] = f["p_ratio_sum_90d"] / n90.clip(lower=1)
    f["p_serial_repeat_share_90d"] = f["p_serial_rep_90d"] / n90.clip(lower=1)
    f["p_prior_claims_mean_90d"] = f["p_prior_sum_90d"] / n90.clip(lower=1)
    f["p_fraud_rate_smoothed"] = (f["p_fraud_n"] + prior_rate * prior_weight) / (f["p_labelled_n"] + prior_weight)

    # drift: last 30 days vs the partner's own previous 5 months
    older = (f["p_claims_180d"] - n30).clip(lower=1)
    f["drift_small_share"] = f["p_small_share_30d"] - (f["p_small_180d"] - f["p_small_30d"]) / older
    f["drift_near_limit_share"] = f["p_near_limit_share_30d"] - (f["p_near_180d"] - f["p_near_30d"]) / older
    f["drift_uninspected_share"] = f["p_uninspected_share_30d"] - (f["p_uninsp_180d"] - f["p_uninsp_30d"]) / older
    f["drift_amount_to_list"] = f["p_mean_amount_to_list_90d"] - f["p_ratio_sum_180d"] / f["p_claims_180d"].clip(lower=1)
    return f


def _market_and_city(df, label_cutoff, prior_rate, label_lag_days, prior_weight=50.0):
    """Network-wide 30-day shares (peer baseline) and time-safe city fraud rate."""
    t = df["ts"].values
    n = _window_sum(t, np.ones(len(df)), t, 30).clip(1)
    f = pd.DataFrame(index=df.index)
    f["rel_small_share"] = _window_sum(t, df["is_small"].values, t, 30) / n
    f["rel_near_limit_share"] = _window_sum(t, df["near_limit"].values, t, 30) / n
    f["rel_uninspected_share"] = _window_sum(t, 1 - df["partner_inspected"].values, t, 30) / n
    f["city_fraud_rate"] = 0.0
    cutoff = np.datetime64(label_cutoff) if label_cutoff is not None else None
    for _, idx in df.groupby("city").indices.items():
        g = df.iloc[idx]
        tt = g["ts"].values
        lab = g["is_fraud"].notna().to_numpy(copy=True)
        if cutoff is not None:
            lab &= tt < cutoff
        known_until = tt - np.timedelta64(label_lag_days, "D")
        if cutoff is not None:
            known_until = np.minimum(known_until, cutoff)
        k = np.searchsorted(tt[lab], known_until, side="left")
        cf = np.concatenate([[0.0], np.cumsum(g["is_fraud"].values[lab])])[k]
        f.loc[f.index[idx], "city_fraud_rate"] = (cf + prior_rate * prior_weight) / (k + prior_weight)
    return f


def build_features(claims, partners, products, label_cutoff=None, prior_rate=0.013,
                   label_lag_days=LABEL_LAG_DAYS):
    """claims: every claim available as history AND to be scored (train + test, cleaned).

    label_cutoff: outcomes of claims submitted on/after this date are treated as unknown
    (used by walk-forward validation so a fold never sees its own month's labels).
    Returns a frame aligned to `claims` (sorted by time) with FEATURES plus id/label columns.
    """
    df = claims.sort_values("ts", kind="stable").reset_index(drop=True)
    df = _claim_level(df, partners, products)
    # earlier claims on the same physical unit (after serial normalisation)
    df["serial_prior_claims"] = df.groupby("serial_norm").cumcount()
    df["serial_prior_other_partner"] = df["serial_prior_claims"] - df.groupby(["serial_norm", "partner_id"]).cumcount()
    hist = _partner_history(df, label_cutoff, prior_rate, label_lag_days)
    df = pd.concat([df, hist], axis=1)
    mkt = _market_and_city(df, label_cutoff, prior_rate, label_lag_days)
    for c in ["small_share", "near_limit_share", "uninspected_share"]:
        base = "p_near_limit_share_30d" if c == "near_limit_share" else f"p_{c}_30d"
        df[f"rel_{c}"] = df[base] - mkt[f"rel_{c}"]
    df["city_fraud_rate"] = mkt["city_fraud_rate"]
    # fixed category sets so batch training and single-record scoring encode identically
    df["partner_type"] = pd.Categorical(df["partner_type"], categories=sorted(partners["partner_type"].unique()))
    df["family"] = pd.Categorical(df["family"], categories=sorted(products["family"].unique()))
    return df
