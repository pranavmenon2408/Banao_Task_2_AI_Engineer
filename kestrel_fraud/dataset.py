"""Validate a user-uploaded claims CSV before it can be used for training.

The file must have the same shape as the data pack's train.csv. Every problem is reported at once,
with example row numbers, so the user can fix the file in one pass and re-upload.
"""

import io
from typing import Any

import numpy as np
import pandas as pd

REQUIRED = [
    "claim_id",
    "submitted_at",
    "partner_id",
    "sku",
    "product_serial",
    "days_since_purchase",
    "claim_amount_inr",
    "photo_attached",
    "partner_inspected",
    "customer_prior_claims",
    "is_fraud",
]
OPTIONAL = {"claim_description": "", "inspector_note": None, "source": "crm"}
MAX_EXAMPLES = 5
MIN_FRAUDS = 15


def template_csv():
    """Two example rows in the expected shape."""
    return pd.DataFrame(
        [
            dict(
                claim_id="WC900001",
                submitted_at="2026-07-03 10:15",
                partner_id="SP3160",
                sku="KH-AF-02",
                product_serial="KH128015513",
                days_since_purchase=150,
                claim_amount_inr=1995.0,
                photo_attached="N",
                partner_inspected="N",
                claim_description="not heating",
                inspector_note="",
                customer_prior_claims=1,
                source="crm",
                is_fraud=1,
            ),
            dict(
                claim_id="WC900002",
                submitted_at="2026-07-03 11:40",
                partner_id="SP3265",
                sku="KH-CF-01",
                product_serial="KH835948289",
                days_since_purchase=300,
                claim_amount_inr=900.0,
                photo_attached="Y",
                partner_inspected="Y",
                claim_description="motor not running",
                inspector_note="Unit inspected, fault confirmed",
                customer_prior_claims=0,
                source="crm",
                is_fraud=0,
            ),
        ]
    ).to_csv(index=False)


def _rows(mask):
    """1-based CSV line numbers (header is line 1) of the first failing rows."""
    idx = np.flatnonzero(np.asarray(mask))
    return {"count": len(idx), "example_lines": [int(i) + 2 for i in idx[:MAX_EXAMPLES]]}


def validate(raw: bytes, partners, products):
    """Returns (df or None, report). report = {ok, errors[], warnings[], summary}."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    report: dict[str, Any] = {"ok": False, "errors": errors, "warnings": warnings, "summary": None}
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, na_values=[""])
    except Exception as e:
        errors.append({"problem": "file is not a readable CSV", "detail": str(e)[:200]})
        return None, report
    if df.empty:
        errors.append(
            {
                "problem": "file has no data rows",
                "fix": "upload a CSV with a header row and at least one claim (see the template)",
            }
        )
        return None, report

    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        errors.append(
            {
                "problem": "missing required columns",
                "columns": missing,
                "fix": f"the header must contain: {', '.join(REQUIRED)}",
            }
        )
    extra = [c for c in df.columns if c not in REQUIRED and c not in OPTIONAL]
    if extra:
        warnings.append({"problem": "unrecognised columns will be ignored", "columns": extra})
    if missing:
        return None, report  # cell checks are meaningless without the columns

    for c, default in OPTIONAL.items():
        if c not in df.columns:
            df[c] = default
    df = df[REQUIRED + list(OPTIONAL)].copy()

    def check(col, bad, fix):
        if bad.any():
            errors.append({"problem": f"invalid values in '{col}'", **_rows(bad), "fix": fix})

    for col in REQUIRED:
        if col != "is_fraud":
            check(col, df[col].isna(), "must not be blank")
    ts = pd.to_datetime(df["submitted_at"], errors="coerce", format="mixed")
    check("submitted_at", df["submitted_at"].notna() & ts.isna(), "use a date-time like 2026-07-03 10:15 (IST)")
    for col, lo, integer in [
        ("days_since_purchase", 0, True),
        ("claim_amount_inr", 0.01, False),
        ("customer_prior_claims", 0, True),
    ]:
        num = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & (num.isna() | (num < lo) | ((num % 1 != 0) if integer else False))
        check(col, bad, f"must be a number >= {lo}" + (" with no decimals" if integer else ""))
        df[col] = num
    for col in ["photo_attached", "partner_inspected"]:
        df[col] = df[col].str.strip().str.upper()
        check(col, df[col].notna() & ~df[col].isin(["Y", "N"]), "must be Y or N")
    label = pd.to_numeric(df["is_fraud"], errors="coerce")
    check("is_fraud", df["is_fraud"].notna() & ~label.isin([0, 1]), "must be 1 (fraud), 0 (not fraud) or blank (undecided)")
    df["is_fraud"] = label
    df["source"] = df["source"].fillna("crm").str.strip()
    check("source", ~df["source"].isin(["crm", "legacy_zoho"]), "must be crm or legacy_zoho (blank = crm)")
    check(
        "partner_id",
        df["partner_id"].notna() & ~df["partner_id"].isin(set(partners["partner_id"])),
        "partner must exist in partners.csv",
    )
    check("sku", df["sku"].notna() & ~df["sku"].isin(set(products["sku"])), "SKU must exist in products.csv")

    if errors:
        return None, report

    df["submitted_at"] = ts.dt.strftime("%Y-%m-%d %H:%M")
    dup = df["claim_id"].duplicated(keep=False)
    if dup.any():
        warnings.append({"problem": "repeated claim_id treated as resubmissions (first submission kept)", **_rows(dup)})
    labelled = df["is_fraud"].notna()
    frauds = int((df["is_fraud"] == 1).sum())
    months = ts[labelled].dt.to_period("M").nunique()
    # Not fatal here: these are fine if the upload is combined with the Kestrel history.
    # The training job enforces its own minimums on whatever data it finally trains on.
    if frauds < MIN_FRAUDS:
        warnings.append(
            {
                "problem": f"only {frauds} fraud labels; at least {MIN_FRAUDS} are needed to train on this file alone",
                "fix": "add more labelled claims, or tick 'combine with the Kestrel history'",
            }
        )
    if months < 2:
        warnings.append(
            {
                "problem": "labelled claims span fewer than 2 months",
                "fix": "the last month is held out to measure the model; combine with the history or add another month",
            }
        )
    report["summary"] = {
        "rows": len(df),
        "labelled": int(labelled.sum()),
        "frauds": frauds,
        "undecided": int((~labelled).sum()),
        "from": str(ts.min()),
        "to": str(ts.max()),
        "months": int(months),
    }
    report["ok"] = not errors
    return df, report
