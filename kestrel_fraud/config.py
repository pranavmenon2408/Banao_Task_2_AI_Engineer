"""Constants taken from the data pack (ops policy v4.1, email thread)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

# Policy s4 - costs used for decisions
GOODWILL_INR = 380          # genuine claim held for review
CONTACT_INR = 260           # blended cost of a service contact (every review)

# Policy s5 - claim approval
POLICY_CHANGE = "2026-05-01"   # claims under the threshold auto-approved from here
AUTO_APPROVE_LIMIT = 2000
NEAR_LIMIT_BAND = 200          # claims in [limit - band, limit) look like threshold gaming
DESK_CAPACITY = 40             # investigations per month

# Policy s9 - legacy Zoho exported undecided cases as 0, so its negatives are less trustworthy
ZOHO_NEGATIVE_WEIGHT = 0.8
# Rows after the policy change describe the regime we will score in
POST_POLICY_WEIGHT = 1.0

# Investigation outcomes are not known the moment a claim lands. Label-based history
# features only use outcomes of claims submitted at least this many days earlier.
LABEL_LAG_DAYS = 14

METRO_CITIES = {"Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"}

# Common-sense direction: more confirmed partner fraud, more prior customer claims, or a claim
# closer to list price can only raise risk. Chosen by walk-forward (post-May PR-AUC 0.229 -> 0.267).
MONOTONE_UP = ["p_fraud_rate_smoothed", "p_fraud_n", "p_fraud_90d", "customer_prior_claims", "amount_to_list"]

LGBM_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    n_estimators=250,
    num_leaves=7,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    verbose=-1,
    random_state=7,
)
