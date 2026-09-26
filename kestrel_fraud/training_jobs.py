"""Background training on a user-chosen subset, with progress reported every 10% of boosting rounds.

The last month of the chosen date range is held out: progress metrics are measured on it, so the
curve shows whether the model generalises forward in time rather than how well it memorises.
Optionally the model is then refitted on the whole range (the saved model is that refit).
"""

import threading
import traceback
import uuid
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import model as mdl
from .config import ARTIFACT_DIR, LGBM_PARAMS
from .data import clean_claims
from .features import FEATURES, build_features
from .registry import validate_name

MIN_TRAIN_FRAUDS = 15
UPLOAD_DIR = ARTIFACT_DIR / "uploads"


# Defaults for a training request; validated in JobManager.start()
TRAIN_DEFAULTS: dict[str, Any] = dict(
    date_from=None,
    date_to=None,
    dataset_id=None,
    combine_with_history=True,
    sources=["crm", "legacy_zoho"],
    partner_types=["authorised_service_centre", "franchise", "freelance_technician"],
    families=None,
    n_estimators=LGBM_PARAMS["n_estimators"],
    learning_rate=LGBM_PARAMS["learning_rate"],
    num_leaves=LGBM_PARAMS["num_leaves"],
    monotone=True,
    zoho_negative_weight=0.8,
    refit_full=True,
)


class JobManager:
    def __init__(self, scorer, registry):
        self.scorer, self.registry = scorer, registry
        self.jobs, self.datasets, self._lock = {}, {}, threading.Lock()

    def add_dataset(self, df, filename, summary):
        """Keep a validated upload (and a copy on disk for traceability); returns its id."""
        dataset_id = uuid.uuid4().hex[:10]
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(UPLOAD_DIR / f"{dataset_id}.csv", index=False)
        with self._lock:
            self.datasets[dataset_id] = {"df": df, "filename": filename, "summary": summary}
        return dataset_id

    def _claims_for(self, req):
        """History, the upload, or both. Uploaded rows replace history rows with the same claim_id."""
        parts = []
        if req["dataset_id"]:
            with self._lock:
                ds = self.datasets.get(req["dataset_id"])
            if ds is None:
                raise ValueError(f"no uploaded dataset {req['dataset_id']!r} - upload the CSV again")
            up = clean_claims(ds["df"])
            if req["combine_with_history"] and self.scorer.history is not None:
                parts.append(self.scorer.history[~self.scorer.history["claim_id"].isin(up["claim_id"])])
            parts.append(up)
        else:
            parts.append(self.scorer.history)
        return pd.concat(parts, ignore_index=True)

    def start(self, name, **kwargs):
        validate_name(name)
        req = {**TRAIN_DEFAULTS, **{k: v for k, v in kwargs.items() if v is not None}}
        if req["date_from"] and req["date_to"] and pd.Timestamp(req["date_from"]) >= pd.Timestamp(req["date_to"]):
            raise ValueError("date_from must be before date_to")
        if not 10 <= int(req["n_estimators"]) <= 3000:
            raise ValueError("n_estimators must be between 10 and 3000")
        if req["dataset_id"] and req["dataset_id"] not in self.datasets:
            raise ValueError(f"no uploaded dataset {req['dataset_id']!r} - upload the CSV again")
        if self.scorer.history is None and not req["dataset_id"]:
            raise ValueError("claims history not loaded - put the data pack in data/ or upload a CSV")
        job_id = uuid.uuid4().hex[:10]
        job = {
            "job_id": job_id,
            "name": name,
            "request": req,
            "status": "queued",
            "stage": "",
            "progress_pct": 0,
            "history": [],
            "error": None,
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        with self._lock:
            self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job_id

    def get(self, job_id):
        with self._lock:
            job = self.jobs.get(job_id)
            return None if job is None else {**job, "history": list(job["history"])}

    def _update(self, job, **kw):
        with self._lock:
            job.update(kw)

    def _run(self, job):
        try:
            self._train(job)
        except Exception as e:  # report, never crash the server
            self._update(job, status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=3))

    def _subset(self, feats, req, start, end):
        m = (feats["ts"] >= start) & (feats["ts"] < end) & feats["is_fraud"].notna()
        m &= feats["source"].isin(req["sources"]) & feats["partner_type"].isin(req["partner_types"])
        if req.get("families"):
            m &= feats["family"].isin(req["families"])
        return feats[m], end

    def _train(self, job):
        req = job["request"]
        self._update(job, status="running", stage="building time-safe features")
        claims = self._claims_for(req)
        labelled_ts = claims.loc[claims["is_fraud"].notna(), "ts"]
        if labelled_ts.empty:
            raise ValueError("no labelled claims to train on")
        # blank dates mean "the whole span of the chosen data"
        start = pd.Timestamp(req["date_from"]) if req["date_from"] else labelled_ts.min().normalize()
        end = (pd.Timestamp(req["date_to"]) if req["date_to"] else labelled_ts.max().normalize()) + pd.Timedelta(days=1)
        feats = build_features(claims, self.scorer.partners, self.scorer.products, label_cutoff=end)
        data, end = self._subset(feats, req, start, end)
        if data.empty:
            raise ValueError("no labelled claims match this subset")
        valid_start = (end - pd.Timedelta(days=1)).to_period("M").start_time
        train, valid = data[data["ts"] < valid_start], data[data["ts"] >= valid_start]
        if train["is_fraud"].sum() < MIN_TRAIN_FRAUDS:
            raise ValueError(
                f"only {int(train['is_fraud'].sum())} frauds before the hold-out month; "
                f"need at least {MIN_TRAIN_FRAUDS} - widen the date range or filters"
            )

        params: dict[str, Any] = {
            **LGBM_PARAMS,
            "n_estimators": int(req["n_estimators"]),
            "learning_rate": float(req["learning_rate"]),
            "num_leaves": int(req["num_leaves"]),
        }
        if req["monotone"]:
            params.update(mdl.monotone(FEATURES))

        def weights(df):
            w = np.ones(len(df))
            w[((df["source"] == "legacy_zoho") & (df["is_fraud"] == 0)).values] = float(req["zoho_negative_weight"])
            return w

        n = params["n_estimators"]
        step = max(1, n // 10)
        has_valid_fraud = 0 < valid["is_fraud"].sum() < len(valid)

        def progress(env):
            it = env.iteration + 1
            if it % step and it != n:
                return
            rec = {"iteration": it, "pct": round(100 * it / n)}
            for data_name, metric, value, _ in env.evaluation_result_list:
                rec[f"{data_name}_{metric}"] = round(float(value), 5)
            with self._lock:
                job["history"].append(rec)
                job["progress_pct"] = rec["pct"] if not req["refit_full"] else round(0.9 * rec["pct"])

        self._update(
            job,
            stage=f"training on {len(train):,} claims, measuring on hold-out {valid_start:%b %Y} ({len(valid):,} claims)",
        )
        clf = lgb.LGBMClassifier(**params)
        eval_set = [(train[FEATURES], train["is_fraud"].astype(int))]
        names = ["train"]
        if has_valid_fraud:
            eval_set.append((valid[FEATURES], valid["is_fraud"].astype(int)))
            names.append("holdout")
        clf.fit(
            train[FEATURES],
            train["is_fraud"].astype(int),
            sample_weight=weights(train),
            eval_set=eval_set,
            eval_names=names,
            eval_metric=["binary_logloss", "average_precision"],
            callbacks=[progress],
        )

        validation: dict[str, Any] = {
            "holdout_month": f"{valid_start:%Y-%m}",
            "holdout_claims": len(valid),
            "holdout_frauds": int(valid["is_fraud"].sum()),
        }
        if has_valid_fraud:
            p = np.asarray(clf.predict_proba(valid[FEATURES]))[:, 1]
            validation.update(
                pr_auc=round(average_precision_score(valid["is_fraud"], p), 4),
                roc_auc=round(roc_auc_score(valid["is_fraud"], p), 4),
            )

        final, trained = clf, train
        if req["refit_full"]:
            self._update(job, stage="refitting on the full date range", progress_pct=92)
            final = lgb.LGBMClassifier(**params).fit(
                data[FEATURES], data["is_fraud"].astype(int), sample_weight=weights(data)
            )
            trained = data

        meta = {
            "features": FEATURES,
            "params": {k: v for k, v in params.items() if k != "monotone_constraints"},
            "monotone": bool(req["monotone"]),
            "created": datetime.now().isoformat(timespec="seconds"),
            "subset": {
                **{
                    k: req[k]
                    for k in [
                        "sources",
                        "partner_types",
                        "families",
                        "zoho_negative_weight",
                        "refit_full",
                        "combine_with_history",
                    ]
                },
                "date_from": str(start.date()),
                "date_to": str((end - pd.Timedelta(days=1)).date()),
                "data": (
                    f"uploaded {self.datasets[req['dataset_id']]['filename']}"
                    + (" + Kestrel history" if req["combine_with_history"] else " only")
                )
                if req["dataset_id"]
                else "Kestrel history",
            },
            "trained_on": {
                "claims": len(trained),
                "frauds": int(trained["is_fraud"].sum()),
                "from": str(trained["ts"].min()),
                "to": str(trained["ts"].max()),
            },
            "base_rate": float(trained["is_fraud"].mean()),
            "validation": validation,
            "training_history": job["history"],
        }
        self.registry.save(job["name"], final.booster_, meta)
        self._update(
            job,
            status="done",
            stage="saved",
            progress_pct=100,
            validation=validation,
            finished=datetime.now().isoformat(timespec="seconds"),
        )
