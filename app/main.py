"""Kestrel warranty-claim risk service.

POST   /score              one claim as JSON -> fraud probability, recommendation, reasons
GET    /models             available models ('default' is the shipped one)
POST   /train              train a named model on a subset of the history (background job)
GET    /train/{job_id}     poll a training job: status, % done, metrics every 10% of boosting rounds
DELETE /models/{name}      remove a user-trained model
GET    /health             model + history status
GET    /                   fallback single-page screen (the main screen is Streamlit, see README)
"""
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kestrel_fraud.registry import DEFAULT  # noqa: E402
from kestrel_fraud.scorer import ClaimScorer  # noqa: E402
from kestrel_fraud.training_jobs import JobManager  # noqa: E402

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Kestrel warranty claim review")
STATIC = Path(__file__).parent / "static"

try:
    scorer = ClaimScorer()
    jobs = JobManager(scorer, scorer.registry)
    startup_error = None
except Exception as e:  # keep the server up and explain what is missing
    scorer, jobs, startup_error = None, None, f"{type(e).__name__}: {e}"
    logging.exception("scorer failed to load")


def need_scorer():
    if scorer is None:
        raise HTTPException(503, f"Model not loaded ({startup_error}). Run `python scripts/train.py` first - see README.")


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
    model: str = Field(DEFAULT, description="which trained model to score with")


class TrainBody(BaseModel):
    name: str = Field(examples=["crm_only_2026"], description="letters, digits, _ or -; 'default' is reserved")
    date_from: str = Field("2025-04-01", description="first claim date in the training subset")
    date_to: str = Field("2026-06-30", description="last claim date; its month is held out to measure progress")
    sources: list[str] = ["crm", "legacy_zoho"]
    partner_types: list[str] = ["authorised_service_centre", "franchise", "freelance_technician"]
    families: list[str] | None = None
    n_estimators: int = Field(250, ge=10, le=3000)
    learning_rate: float = Field(0.03, gt=0, le=1)
    num_leaves: int = Field(7, ge=2, le=255)
    monotone: bool = True
    zoho_negative_weight: float = Field(0.8, ge=0, le=1)
    refit_full: bool = Field(True, description="after measuring on the hold-out month, refit on the whole range")


@app.get("/health")
def health():
    if scorer is None:
        return {"ok": False, "error": startup_error,
                "fix": "Run `python scripts/train.py` with the data pack in data/ (see README)."}
    return {"ok": True, "model": scorer.meta["trained_on"], "history": scorer.history_note,
            "models": scorer.registry.names()}


@app.post("/score")
def score(claim: Claim):
    need_scorer()
    body = claim.model_dump()
    try:
        return scorer.score(body, model_name=body.pop("model"))
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/models")
def models():
    need_scorer()
    return scorer.registry.list()


@app.delete("/models/{name}")
def delete_model(name: str):
    need_scorer()
    try:
        scorer.registry.delete(name)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"deleted": name}


@app.post("/train", status_code=202)
def train(body: TrainBody):
    need_scorer()
    try:
        job_id = jobs.start(**body.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"job_id": job_id, "poll": f"/train/{job_id}"}


@app.get("/train/{job_id}")
def train_status(job_id: str):
    need_scorer()
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no training job {job_id!r} (jobs are kept in memory until the server restarts)")
    return job


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
