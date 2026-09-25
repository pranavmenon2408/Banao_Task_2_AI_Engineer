"""Kestrel warranty-claim risk service.

POST /score   one claim as JSON -> fraud probability, recommendation, reasons
GET  /        the screen that calls it
GET  /health  model + history status
"""
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kestrel_fraud.scorer import ClaimScorer  # noqa: E402

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Kestrel warranty claim review")
STATIC = Path(__file__).parent / "static"

try:
    scorer = ClaimScorer()
    startup_error = None
except Exception as e:  # keep the server up and explain what is missing
    scorer, startup_error = None, f"{type(e).__name__}: {e}"
    logging.exception("scorer failed to load")


class Claim(BaseModel):
    partner_id: str = Field(examples=["SP3160"])
    sku: str = Field(examples=["KH-AF-02"])
    product_serial: str = Field(examples=["KH-128015513"])
    days_since_purchase: int = Field(ge=0, examples=[150])
    claim_amount_inr: float = Field(gt=0, examples=[1995])
    photo_attached: str = Field(pattern="^[YN]$", examples=["N"])
    partner_inspected: str = Field(pattern="^[YN]$", examples=["N"])
    customer_prior_claims: int = Field(ge=0, examples=[1])
    submitted_at: str | None = Field(None, examples=["2026-09-20 11:30"], description="IST; defaults to now")
    claim_id: str | None = None
    claim_description: str | None = None
    inspector_note: str | None = None


@app.get("/health")
def health():
    if scorer is None:
        return {"ok": False, "error": startup_error,
                "fix": "Run `python scripts/train.py` with the data pack in data/ (see README)."}
    return {"ok": True, "model": scorer.meta["trained_on"], "history": scorer.history_note}


@app.post("/score")
def score(claim: Claim):
    if scorer is None:
        raise HTTPException(503, f"Model not loaded ({startup_error}). Run `python scripts/train.py` first - see README.")
    try:
        return scorer.score(claim.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
