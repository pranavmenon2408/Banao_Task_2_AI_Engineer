"""Loading and cleaning the raw export."""

import re

import pandas as pd

from .config import DATA_DIR

SERIAL_RE = re.compile(r"^KH\d{9}$")


def normalise_serial(raw: str) -> str:
    """Partners type serials by hand: 'kh123', ' KH-123' and 'KH123' are the same unit."""
    return re.sub(r"[^A-Z0-9]", "", str(raw).upper())


def load_reference(data_dir=DATA_DIR):
    partners = pd.read_csv(data_dir / "partners.csv", parse_dates=["onboarded_date"])
    products = pd.read_csv(data_dir / "products.csv")
    return partners, products


def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["submitted_at"])
    # Bounced claims are re-submitted under the same claim_id (email thread). Labels agree
    # across copies, so keep the first submission - that is when the claim was really made.
    df = df.sort_values("ts").drop_duplicates("claim_id", keep="first")
    df["serial_norm"] = df["product_serial"].map(normalise_serial)
    df["serial_messy"] = (~df["product_serial"].astype(str).str.match(SERIAL_RE)).astype(int)
    return df.reset_index(drop=True)


def load_claims(data_dir=DATA_DIR):
    """Returns (train, test). train.is_fraud is NaN where the case was undecided at export."""
    train = clean_claims(pd.read_csv(data_dir / "train.csv"))
    test = clean_claims(pd.read_csv(data_dir / "test_unlabelled.csv"))
    test["is_fraud"] = float("nan")
    return train, test
