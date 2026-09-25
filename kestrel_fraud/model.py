"""LightGBM fraud model: training and scoring on the shared feature frame."""
import lightgbm as lgb
import numpy as np

from .config import LGBM_PARAMS, POLICY_CHANGE, POST_POLICY_WEIGHT, ZOHO_NEGATIVE_WEIGHT
from .features import FEATURES


def sample_weights(df, post_policy_weight=POST_POLICY_WEIGHT):
    """Zoho exported undecided cases as 0, so its negatives are partly unknowns - trust them less.
    Claims after the May-2026 approval change are the regime we score in - trust them more."""
    w = np.ones(len(df))
    w[((df["source"] == "legacy_zoho") & (df["is_fraud"] == 0)).values] = ZOHO_NEGATIVE_WEIGHT
    w[(df["ts"] >= POLICY_CHANGE).values] *= post_policy_weight
    return w


def fit(train_df, params=None, post_policy_weight=POST_POLICY_WEIGHT):
    train_df = train_df[train_df["is_fraud"].notna()]
    model = lgb.LGBMClassifier(**{**LGBM_PARAMS, **(params or {})})
    model.fit(train_df[FEATURES], train_df["is_fraud"].astype(int), sample_weight=sample_weights(train_df, post_policy_weight))
    return model


def predict(model, df):
    return model.predict_proba(df[FEATURES])[:, 1]


def contributions(model, df):
    """Per-feature SHAP contributions in log-odds (last column is the base value)."""
    return model.booster_.predict(df[FEATURES], pred_contrib=True)
