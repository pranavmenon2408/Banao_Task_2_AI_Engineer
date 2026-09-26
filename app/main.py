"""Kestrel warranty-claim risk service.

POST   /score              one claim as JSON -> fraud probability, recommendation, reasons
GET    /models             available models ('default' is the shipped one)
POST   /datasets           upload a claims CSV for training; returns every problem to fix, or a dataset_id
GET    /datasets/template  a two-row CSV in the expected shape
POST   /train              train a named model on the history and/or an upload (background job)
GET    /train/{job_id}     poll a training job: status, % done, metrics every 10% of boosting rounds
DELETE /models/{name}      remove a user-trained model
GET    /health             model + history status
GET    /                   fallback single-page screen (the main screen is Streamlit, see README)
"""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kestrel_fraud import dataset
from kestrel_fraud.registry import DEFAULT
from kestrel_fraud.scorer import ClaimScorer
from kestrel_fraud.training_jobs import JobManager

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Kestrel warranty claim review")
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_MB = 50

_scorer: ClaimScorer | None = None
_jobs: JobManager | None = None
startup_error: str | None = None
try:
    _scorer = ClaimScorer()
    _jobs = JobManager(_scorer, _scorer.registry)
except Exception as e:  # keep the server up and explain what is missing
    startup_error = f"{type(e).__name__}: {e}"
    logging.exception("scorer failed to load")


def need_scorer() -> ClaimScorer:
    if _scorer is None:
        raise HTTPException(503, f"Model not loaded ({startup_error}). Run `python scripts/train.py` first - see README.")
    return _scorer


def need_jobs() -> JobManager:
    need_scorer()
    assert _jobs is not None  # created together with the scorer
    return _jobs


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
    dataset_id: str | None = Field(None, description="an upload from POST /datasets; omit to use the Kestrel history")
    combine_with_history: bool = Field(True, description="with an upload: also train on the Kestrel history")
    date_from: str | None = Field(None, description="first claim date in the training subset (blank = earliest)")
    date_to: str | None = Field(None, description="last claim date; its month is held out (blank = latest)")
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
    if _scorer is None:
        return {
            "ok": False,
            "error": startup_error,
            "fix": "Run `python scripts/train.py` with the data pack in data/ (see README).",
        }
    return {
        "ok": True,
        "model": _scorer.meta["trained_on"],
        "history": _scorer.history_note,
        "models": _scorer.registry.names(),
    }


@app.post("/score")
def score(claim: Claim):
    scorer = need_scorer()
    body = claim.model_dump()
    try:
        return scorer.score(body, model_name=body.pop("model"))
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\"")) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/models")
def models():
    return need_scorer().registry.list()


@app.delete("/models/{name}")
def delete_model(name: str):
    scorer = need_scorer()
    try:
        scorer.registry.delete(name)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"deleted": name}


@app.get("/datasets/template", response_class=PlainTextResponse)
def dataset_template():
    return PlainTextResponse(
        dataset.template_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=claims_template.csv"},
    )


@app.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    """Validate an uploaded claims CSV. 422 lists every problem so the file can be fixed in one pass."""
    scorer, jobs = need_scorer(), need_jobs()
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file is larger than {MAX_UPLOAD_MB} MB")
    df, report = dataset.validate(raw, scorer.partners, scorer.products)
    if not report["ok"]:
        raise HTTPException(
            422,
            {
                "message": "the file does not match the expected shape - fix these and re-upload",
                "required_columns": dataset.REQUIRED,
                "optional_columns": list(dataset.OPTIONAL),
                **report,
            },
        )
    return {"dataset_id": jobs.add_dataset(df, file.filename, report["summary"]), "filename": file.filename, **report}


@app.post("/train", status_code=202)
def train(body: TrainBody):
    jobs = need_jobs()
    try:
        job_id = jobs.start(**body.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"job_id": job_id, "poll": f"/train/{job_id}"}


@app.get("/train/{job_id}")
def train_status(job_id: str):
    job = need_jobs().get(job_id)
    if job is None:
        raise HTTPException(404, f"no training job {job_id!r} (jobs are kept in memory until the server restarts)")
    return job


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
