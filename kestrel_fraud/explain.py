"""Turn per-feature SHAP contributions into sentences a claims officer can act on."""

import numpy as np

from .config import AUTO_APPROVE_LIMIT


def _pct(x):
    return f"{100 * x:.0f}%"


# feature -> function(row) returning the sentence shown when that feature pushes the score
TEMPLATES = {
    "p_fraud_rate_smoothed": lambda r: (
        f"This partner has {int(r.p_fraud_n)} confirmed fraudulent claims out of {int(r.p_labelled_n)} investigated"
    ),
    "p_fraud_n": lambda r: f"This partner has {int(r.p_fraud_n)} confirmed fraudulent claims on record",
    "p_fraud_90d": lambda r: f"{int(r.p_fraud_90d)} of this partner's claims in the last 3 months were confirmed fraud",
    "p_labelled_n": lambda r: f"Partner has {int(r.p_labelled_n)} investigated claims on record",
    "amount_to_list": lambda r: f"Claim is {_pct(r.amount_to_list)} of the product's list price",
    "claim_amount_inr": lambda r: f"Claim amount Rs {r.claim_amount_inr:,.0f}",
    "customer_prior_claims": lambda r: f"Customer has {int(r.customer_prior_claims)} earlier warranty claims",
    "near_limit": lambda r: (
        f"Amount sits just under the Rs {AUTO_APPROVE_LIMIT:,} auto-approval limit"
        if r.near_limit
        else f"Amount is not near the Rs {AUTO_APPROVE_LIMIT:,} auto-approval limit"
    ),
    "auto_approved": lambda r: (
        "Under Rs 2,000 after May 2026, so eligible for auto-approval without inspection"
        if r.auto_approved
        else "Not eligible for auto-approval (inspection route)"
    ),
    "partner_inspected": lambda r: "Partner inspected the unit" if r.partner_inspected else "No partner inspection",
    "inspector_note_missing": lambda r: "No inspector note" if r.inspector_note_missing else "Inspector note present",
    "photo_attached": lambda r: "Photo attached" if r.photo_attached else "No photo attached",
    "serial_prior_claims": lambda r: (
        f"This serial number has been claimed {int(r.serial_prior_claims)} time(s) before"
        if r.serial_prior_claims
        else "First claim on this serial number"
    ),
    "serial_prior_other_partner": lambda r: (
        f"Serial previously claimed through {int(r.serial_prior_other_partner)} other partner claim(s)"
    ),
    "serial_messy": lambda r: (
        "Serial number typed in a non-standard format" if r.serial_messy else "Serial number well-formed"
    ),
    "partner_age_days": lambda r: f"Partner onboarded {r.partner_age_days:.0f} days before this claim",
    "partner_type": lambda r: f"Partner type: {str(r.partner_type).replace('_', ' ')}",
    "partner_metro": lambda r: "Partner in a metro city" if r.partner_metro else "Partner in a tier-2/3 city",
    "p_claims_30d": lambda r: f"Partner filed {int(r.p_claims_30d)} claims in the previous 30 days",
    "p_claims_90d": lambda r: f"Partner filed {int(r.p_claims_90d)} claims in the previous 90 days",
    "p_growth_30d_vs_180d": lambda r: f"Partner's claim volume is {r.p_growth_30d_vs_180d:.1f}x its 6-month monthly average",
    "p_small_share_30d": lambda r: f"{_pct(r.p_small_share_30d)} of partner's recent claims are under Rs 2,000",
    "p_near_limit_share_30d": lambda r: (
        f"{_pct(r.p_near_limit_share_30d)} of partner's recent claims sit just under Rs 2,000"
    ),
    "p_uninspected_share_30d": lambda r: f"{_pct(r.p_uninspected_share_30d)} of partner's recent claims were not inspected",
    "p_mean_amount_to_list_90d": lambda r: f"Partner's claims average {_pct(r.p_mean_amount_to_list_90d)} of list price",
    "p_serial_repeat_share_90d": lambda r: f"{_pct(r.p_serial_repeat_share_90d)} of partner's recent claims reuse a serial",
    "p_prior_claims_mean_90d": lambda r: f"Partner's customers average {r.p_prior_claims_mean_90d:.1f} earlier claims",
    "days_since_purchase": lambda r: f"Claimed {int(r.days_since_purchase)} days after purchase",
    "warranty_used_frac": lambda r: f"{_pct(r.warranty_used_frac)} of the warranty period used",
    "family": lambda r: f"Product family: {r.family}",
}


def reasons(row, contrib, features, top=4):
    """Returns (risk_up, risk_down): the strongest sentences pushing the score up and down.
    Features describing the same fact are collapsed so each point is made once."""
    groups = {
        "p_fraud_rate_smoothed": "partner_fraud",
        "p_fraud_n": "partner_fraud",
        "p_fraud_90d": "partner_fraud",
        "p_labelled_n": "partner_fraud",
        "claim_amount_inr": "amount",
        "amount_to_list": "amount",
        "auto_approved": "inspection",
        "partner_inspected": "inspection",
        "inspector_note_missing": "inspection",
    }
    order = np.argsort(-np.abs(contrib))
    up: list[dict] = []
    down: list[dict] = []
    seen: set[str] = set()
    for i in order:
        f, c = features[i], contrib[i]
        g = groups.get(f, f)
        if abs(c) < 0.05 or g in seen:
            continue
        seen.add(g)
        item = {"feature": f, "effect": round(float(c), 3), "text": TEMPLATES[f](row)}
        (up if c > 0 else down).append(item)
    return up[:top], down[:top]
