"""Background training on a user-chosen subset, with progress reported every 10% of boosting rounds.

The last month of the chosen date range is held out: progress metrics are measured on it, so the
curve shows whether the model generalises forward in time rather than how well it memorises.
Optionally the model is then refitted on the whole range (the saved model is that refit).
"""
import threading
import time
import traceback
import uuid
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import model as mdl
from .config import LGBM_PARAMS
from .features import FEATURES, build_features
from .registry import validate_name

MIN_TRAIN_FRAUDS = 15


class TrainRequest(dict):
    """Plain dict with defaults; validated in start()."""
    DEFAULTS = dict(date_from="2025-04-01", date_to="2026-06-30", sources=["crm", "legacy_zoho"],
                    partner_types=["authorised_service_centre", "franchise", "freelance_technician"],
                    families=None, n_estimators=LGBM_PARAMS["n_estimators"], learning_rate=LGBM_PARAMS["learning_rate"],
                    num_leaves=LGBM_PARAMS["num_leaves"], monotone=True, zoho_negative_weight=0.8, refit_full=True)


class JobManager:
    def __init__(self, scorer, registry):
        self.scorer, self.registry = scorer, registry
        self.jobs, self._lock = {}, threading.Lock()

    def start(self, name, **kwargs):
        validate_name(name)
        req = {**TrainRequest.DEFAULTS, **{k: v for k, v in kwargs.items() if v is not None}}
        if pd.Timestamp(req["date_from"]) >= pd.Timestamp(req["date_to"]):
            raise ValueError("date_from must be before date_to")
        if not 10 <= int(req["n_estimators"]) <= 3000:
            raise ValueError("n_estimators must be between 10 and 3000")
        if self.scorer.history is None:
            raise ValueError("claims history not loaded - put the data pack in data/")
        job_id = uuid.uuid4().hex[:10]
        job = {"job_id": job_id, "name": name, "request": req, "status": "queued", "stage": "",
               "progress_pct": 0, "history": [], "error": None, "started": datetime.now().isoformat(timespec="seconds")}
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

    def _subset(self, feats, req):
        start, end = pd.Timestamp(req["date_from"]), pd.Timestamp(req["date_to"]) + pd.Timedelta(days=1)
        m = (feats["ts"] >= start) & (feats["ts"] < end) & feats["is_fraud"].notna()
        m &= feats["source"].isin(req["sources"]) & feats["partner_type"].isin(req["partner_types"])
        if req.get("families"):
            m &= feats["family"].isin(req["families"])
        return feats[m], end

    def _train(self, job):
        req = job["request"]
        self._update(job, status="running", stage="building time-safe features")
        end = pd.Timestamp(req["date_to"]) + pd.Timedelta(days=1)
        feats = build_features(self.scorer.history, self.scorer.partners, self.scorer.products, label_cutoff=end)
        data, end = self._subset(feats, req)
        if data.empty:
            raise ValueError("no labelled claims match this subset")
        valid_start = (end - pd.Timedelta(days=1)).to_period("M").start_time
        train, valid = data[data["ts"] < valid_start], data[data["ts"] >= valid_start]
        if train["is_fraud"].sum() < MIN_TRAIN_FRAUDS:
            raise ValueError(f"only {int(train['is_fraud'].sum())} frauds before the hold-out month; "
                             f"need at least {MIN_TRAIN_FRAUDS} - widen the date range or filters")

        params = {**LGBM_PARAMS, "n_estimators": int(req["n_estimators"]), "learning_rate": float(req["learning_rate"]),
                  "num_leaves": int(req["num_leaves"])}
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

        self._update(job, stage=f"training on {len(train):,} claims, measuring on hold-out {valid_start:%b %Y} ({len(valid):,} claims)")
        clf = lgb.LGBMClassifier(**params)
        eval_set = [(train[FEATURES], train["is_fraud"].astype(int))]
        names = ["train"]
        if has_valid_fraud:
            eval_set.append((valid[FEATURES], valid["is_fraud"].astype(int)))
            names.append("holdout")
        clf.fit(train[FEATURES], train["is_fraud"].astype(int), sample_weight=weights(train),
                eval_set=eval_set, eval_names=names, eval_metric=["binary_logloss", "average_precision"],
                callbacks=[progress])

        validation = {"holdout_month": f"{valid_start:%Y-%m}", "holdout_claims": len(valid),
                      "holdout_frauds": int(valid["is_fraud"].sum())}
        if has_valid_fraud:
            p = clf.predict_proba(valid[FEATURES])[:, 1]
            validation.update(pr_auc=round(average_precision_score(valid["is_fraud"], p), 4),
                              roc_auc=round(roc_auc_score(valid["is_fraud"], p), 4))

        final, trained = clf, train
        if req["refit_full"]:
            self._update(job, stage="refitting on the full date range", progress_pct=92)
            final = lgb.LGBMClassifier(**params).fit(data[FEATURES], data["is_fraud"].astype(int), sample_weight=weights(data))
            trained = data

        meta = {"features": FEATURES, "params": {k: v for k, v in params.items() if k != "monotone_constraints"},
                "monotone": bool(req["monotone"]), "created": datetime.now().isoformat(timespec="seconds"),
                "subset": {k: req[k] for k in ["date_from", "date_to", "sources", "partner_types", "families",
                                               "zoho_negative_weight", "refit_full"]},
                "trained_on": {"claims": len(trained), "frauds": int(trained["is_fraud"].sum()),
                               "from": str(trained["ts"].min()), "to": str(trained["ts"].max())},
                "base_rate": float(trained["is_fraud"].mean()), "validation": validation,
                "training_history": job["history"]}
        self.registry.save(job["name"], final.booster_, meta)
        self._update(job, status="done", stage="saved", progress_pct=100, validation=validation,
                     finished=datetime.now().isoformat(timespec="seconds"))
