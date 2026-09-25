"""Score one claim against the claims history - the same feature code as training."""
import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import explain, metrics
from .config import ARTIFACT_DIR, DATA_DIR
from .data import clean_claims, load_claims, load_reference, normalise_serial
from .features import FEATURES, build_features

log = logging.getLogger(__name__)

# ~8x the base fraud rate: worth acting on at partner level even when a desk review is not
FLAG_PROBABILITY = 0.10

REQUIRED = ["partner_id", "sku", "product_serial", "days_since_purchase", "claim_amount_inr",
            "photo_attached", "partner_inspected", "customer_prior_claims"]


class ClaimScorer:
    def __init__(self, artifact_dir=ARTIFACT_DIR, data_dir=DATA_DIR):
        self.booster = lgb.Booster(model_file=str(artifact_dir / "model.txt"))
        self.meta = json.loads((artifact_dir / "meta.json").read_text())
        self.partners, self.products = load_reference(data_dir)
        try:
            train, test = load_claims(data_dir)
            self.history = pd.concat([train, test], ignore_index=True)
            self.history_note = f"{len(self.history):,} past claims loaded as partner history"
        except FileNotFoundError:
            # Service still runs, but partner-history features are empty: say so in every answer.
            self.history = None
            self.history_note = "claims history not found in data/ - partner history features are empty"
        log.info(self.history_note)

    def validate(self, rec):
        missing = [k for k in REQUIRED if rec.get(k) in (None, "")]
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        if rec["partner_id"] not in set(self.partners["partner_id"]):
            raise ValueError(f"unknown partner_id {rec['partner_id']!r} (not in partners.csv)")
        if rec["sku"] not in set(self.products["sku"]):
            raise ValueError(f"unknown sku {rec['sku']!r} (not in products.csv)")

    def score(self, rec):
        self.validate(rec)
        defaults = {"claim_id": "LIVE-REQUEST", "submitted_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "claim_description": "", "source": "crm"}
        rec = {**rec, **{k: v for k, v in defaults.items() if rec.get(k) in (None, "")}}
        rec.setdefault("inspector_note", None)
        new = clean_claims(pd.DataFrame([rec]))
        new["is_fraud"] = np.nan

        # Features need this partner's claims plus the full history of every serial they touch
        # (a partner's serial-repeat share depends on claims made through other partners too)
        if self.history is not None:
            h = self.history
            serials = set(h.loc[h["partner_id"] == rec["partner_id"], "serial_norm"])
            serials.add(normalise_serial(rec["product_serial"]))
            h = h[(h["partner_id"] == rec["partner_id"]) | h["serial_norm"].isin(serials)]
            h = h[h["claim_id"] != rec["claim_id"]]
            claims = pd.concat([h, new], ignore_index=True)
        else:
            claims = new
        feats = build_features(claims, self.partners, self.products)
        row = feats[feats["claim_id"] == rec["claim_id"]].iloc[[0]]

        contrib = self.booster.predict(row[FEATURES], pred_contrib=True)[0]
        p = float(self.booster.predict(row[FEATURES])[0])
        amount = float(row["claim_amount_inr"].iloc[0])
        ev = float(metrics.expected_value(p, amount))
        up, down = explain.reasons(row.iloc[0], contrib[:-1], FEATURES)

        if ev > 0:
            action = "SEND TO INVESTIGATION DESK"
            why = f"Expected saving from reviewing is Rs {ev:,.0f} (fraud risk x amount, minus Rs 640 review and goodwill cost)."
        elif p >= FLAG_PROBABILITY:
            action = "PAY, BUT FLAG PARTNER FOR INSPECTION"
            why = ("Risk is high but the amount is too small for a desk review to pay for itself; "
                   "the fix is to require inspection for this partner's claims.")
        else:
            action = "PAY"
            why = "Risk is low relative to the cost of holding a genuine customer."
        return {
            "fraud_probability": round(p, 4),
            "risk_band": "high" if p >= FLAG_PROBABILITY else "medium" if p >= 0.04 else "low",
            "base_rate": round(self.meta["base_rate"], 4),
            "claim_amount_inr": amount,
            "expected_saving_if_reviewed_inr": round(ev),
            "recommendation": action,
            "recommendation_reason": why,
            "reasons_risk_up": up,
            "reasons_risk_down": down,
            "history_note": self.history_note,
        }
