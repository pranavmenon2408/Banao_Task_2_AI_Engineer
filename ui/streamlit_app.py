"""Kestrel claim review screen. Calls the FastAPI service for scoring (start it first - see README)."""
import json
import time
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kestrel_fraud.config import CONTACT_INR, DESK_CAPACITY, GOODWILL_INR  # noqa: E402
from kestrel_fraud.metrics import desk_queue, expected_value  # noqa: E402

API_URL = os.environ.get("KESTREL_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Kestrel Claim Review", layout="wide")
st.title("Warranty claim review")
st.caption(f"The investigation desk can review {DESK_CAPACITY} claims a month. A review costs Rs {CONTACT_INR} "
           f"plus Rs {GOODWILL_INR} goodwill if the claim turns out genuine, so only claims where the expected "
           "fraud stopped exceeds that cost go to the desk.")

PRESETS = {
    "Small uninspected claim (flagged outlet)": dict(partner_id="SP3160", sku="KH-AF-02", product_serial="kh-128015513",
        claim_amount_inr=1995, days_since_purchase=150, photo_attached="N", partner_inspected="N", customer_prior_claims=1),
    "Large claim, repeat customer": dict(partner_id="SP3207", sku="KH-RV-03", product_serial="KH222151083",
        claim_amount_inr=18500, days_since_purchase=40, photo_attached="Y", partner_inspected="Y", customer_prior_claims=3),
    "Routine claim": dict(partner_id="SP3265", sku="KH-CF-01", product_serial="KH835948289",
        claim_amount_inr=900, days_since_purchase=300, photo_attached="Y", partner_inspected="Y", customer_prior_claims=0),
}


@st.cache_data
def load_csv(path):
    p = ROOT / path
    return pd.read_csv(p) if p.exists() else None


def api_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except requests.RequestException:
        return None


def api_models():
    try:
        r = requests.get(f"{API_URL}/models", timeout=5)
        return r.json() if r.status_code == 200 else []
    except requests.RequestException:
        return []


def monthly_savings(df, watch, top_outlets=7):
    """Expected rupees per test month from (a) the desk queue and (b) verifying the top watchlist outlets."""
    df = df.assign(month=pd.to_datetime(df["submitted_at"]).dt.to_period("M").astype(str))
    outlets = set(watch["partner_id"].head(top_outlets)) if watch is not None else set()
    rows = []
    for month, g in df.groupby("month"):
        p, a = g["score"].values, g["claim_amount_inr"].values
        q = desk_queue(p, a)
        o = g[g["partner_id"].isin(outlets)]
        outlet_fraud = (o["score"] * o["claim_amount_inr"]).sum()
        rows.append({
            "month": month, "claims": len(g), "expected frauds": round(p.sum(), 1),
            "expected fraud Rs": round((p * a).sum()),
            "desk reviews": len(q), "desk: expected fraud Rs caught": round((p[q] * a[q]).sum()),
            "desk: expected net saving Rs": round(expected_value(p[q], a[q]).sum()),
            f"top-{top_outlets} outlets: claims": len(o),
            "outlets: expected fraud Rs": round(outlet_fraud),
            f"outlets: net after Rs {CONTACT_INR} check each": round(outlet_fraud - CONTACT_INR * len(o)),
        })
    out = pd.DataFrame(rows)
    total = out.drop(columns="month").sum(numeric_only=True).round(1)
    return pd.concat([out, pd.DataFrame([{"month": "Total", **total.to_dict()}])], ignore_index=True)


tab_score, tab_preds, tab_train, tab_perf = st.tabs(["Score a claim", "Jul-Sep predictions", "Train a model", "How well it works"])

# --------------------------------------------------------------------------- score one claim
with tab_score:
    health = api_health()
    if health is None:
        st.error(f"The scoring service is not running at {API_URL}. Start it with "
                 "`uvicorn app.main:app --port 8000` (see README), then refresh this page.")
    elif not health.get("ok"):
        st.error(f"The scoring service is up but the model is not loaded: {health.get('error')}. {health.get('fix', '')}")

    models = api_models()
    c1, c2 = st.columns([1, 2])
    model_name = c1.selectbox("Model", [m["name"] for m in models] or ["default"],
                              help="'default' is the shipped model; others were trained in the 'Train a model' tab")
    chosen = next((m for m in models if m["name"] == model_name), None)
    if chosen and chosen.get("trained_on"):
        t, v = chosen["trained_on"], chosen.get("validation") or {}
        c2.caption(f"Trained on {t['claims']:,} claims ({t['frauds']} fraud), {t['from'][:10]} to {t['to'][:10]}"
                   + (f"; hold-out {v['holdout_month']} PR-AUC {v.get('pr_auc', 'n/a')}" if v else ""))
    preset = st.selectbox("Start from an example", ["-"] + list(PRESETS))
    d = PRESETS.get(preset, dict(partner_id="", sku="", product_serial="", claim_amount_inr=1000, days_since_purchase=100,
                                 photo_attached="Y", partner_inspected="Y", customer_prior_claims=0))
    with st.form("claim"):
        c1, c2, c3 = st.columns(3)
        partner_id = c1.text_input("Partner ID", d["partner_id"])
        sku = c2.text_input("SKU", d["sku"])
        serial = c3.text_input("Product serial (as typed)", d["product_serial"])
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Claim amount (Rs)", min_value=1.0, value=float(d["claim_amount_inr"]), step=50.0)
        days = c2.number_input("Days since purchase", min_value=0, value=int(d["days_since_purchase"]))
        prior = c3.number_input("Customer's earlier claims", min_value=0, value=int(d["customer_prior_claims"]))
        c1, c2, c3 = st.columns(3)
        photo = c1.selectbox("Photo attached", ["Y", "N"], index=["Y", "N"].index(d["photo_attached"]))
        inspected = c2.selectbox("Partner inspected", ["Y", "N"], index=["Y", "N"].index(d["partner_inspected"]))
        submitted = c3.text_input("Submitted at (IST, blank = now)", "2026-09-20 11:30" if preset != "-" else "")
        go = st.form_submit_button("Score claim", type="primary")

    if go:
        body = dict(partner_id=partner_id.strip(), sku=sku.strip(), product_serial=serial, claim_amount_inr=amount,
                    days_since_purchase=int(days), photo_attached=photo, partner_inspected=inspected,
                    customer_prior_claims=int(prior), submitted_at=submitted.strip() or None, model=model_name)
        try:
            r = requests.post(f"{API_URL}/score", json=body, timeout=15)
        except requests.RequestException as e:
            st.error(f"Could not reach the scoring service at {API_URL}: {e}")
        else:
            if r.status_code != 200:
                detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
                st.error(f"Could not score this claim: {detail if isinstance(detail, str) else json.dumps(detail)}")
            else:
                res = r.json()
                rec = res["recommendation"]
                st.caption(f"Scored with model **{res['model']}**")
                (st.error if rec.startswith("SEND") else st.warning if "FLAG" in rec else st.success)(f"**{rec}**  \n{res['recommendation_reason']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Fraud risk", f"{100 * res['fraud_probability']:.1f}%",
                          f"{res['fraud_probability'] / res['base_rate']:.1f}x a typical claim", delta_color="inverse")
                m2.metric("Claim amount", f"Rs {res['claim_amount_inr']:,.0f}")
                m3.metric("Expected saving if reviewed", f"Rs {res['expected_saving_if_reviewed_inr']:,}")
                c1, c2 = st.columns(2)
                c1.markdown("**Why it looks risky**\n" + "\n".join(f"- {x['text']}" for x in res["reasons_risk_up"]) if res["reasons_risk_up"] else "**Why it looks risky**\n- Nothing significant")
                c2.markdown("**What counts in its favour**\n" + "\n".join(f"- {x['text']}" for x in res["reasons_risk_down"]) if res["reasons_risk_down"] else "**What counts in its favour**\n- Nothing significant")
                st.caption(f"{res['history_note']}. The risk is a model estimate from 15 months of investigated claims: "
                           "a prompt for review, not a finding of fraud.")
                with st.expander("Raw API response"):
                    st.json(res)

# --------------------------------------------------------------------------- batch predictions
with tab_preds:
    preds = load_csv("predictions.csv")
    test = load_csv("data/test_unlabelled.csv")
    if preds is None:
        st.info("predictions.csv not found - run `python scripts/train.py`.")
    else:
        df = preds if test is None else test.merge(preds, on="claim_id")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Claims scored", f"{len(df):,}")
        c2.metric("Expected frauds (sum of risk)", f"{df['score'].sum():.0f}")
        c3.metric("Risk above 10%", int((df["score"] > 0.10).sum()))
        c4.metric("Median risk", f"{100 * df['score'].median():.2f}%")
        st.markdown("**Score distribution** - most claims are near zero because fraud is rare (1-3% of claims); "
                    "the useful signal is the ranking and the long tail.")
        bands = pd.cut(df["score"], [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0],
                       labels=["<1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-40%", ">40%"])
        st.bar_chart(bands.value_counts().sort_index())
        min_score = st.slider("Show claims with risk at least", 0.0, 1.0, 0.10, 0.01)
        show = df[df["score"] >= min_score].sort_values("score", ascending=False)
        st.dataframe(show, width="stretch", hide_index=True,
                     column_config={"score": st.column_config.ProgressColumn("risk", min_value=0, max_value=1, format="%.3f")})
        if test is not None:
            st.markdown("**Month by month: what acting on these scores is expected to save** "
                        "(model estimates - these months have no outcomes yet)")
            st.dataframe(monthly_savings(df, load_csv("reports/partner_watchlist.csv")), width="stretch", hide_index=True)
            st.caption(f"Desk: claims ranked by expected saving, at most {DESK_CAPACITY} a month and only where a review pays "
                       f"for itself. Outlets: expected fraud rupees at the top watchlist outlets, minus Rs {CONTACT_INR} to verify "
                       "each of their claims. These are expectations (risk x amount); a real month will vary around them.")
        c1, c2 = st.columns(2)
        q, w = load_csv("reports/desk_queue.csv"), load_csv("reports/partner_watchlist.csv")
        if q is not None:
            c1.markdown("**Desk worklist** (per month, only claims where review pays for itself)")
            c1.dataframe(q, width="stretch", hide_index=True)
        if w is not None:
            c2.markdown("**Outlets to take off auto-approval** (expected fraud rupees, Jul-Sep)")
            c2.dataframe(w[["partner_id", "city", "partner_type", "claims", "mean_score", "expected_fraud_inr",
                            "known_frauds_before_jul"]], width="stretch", hide_index=True)

# --------------------------------------------------------------------------- train
with tab_train:
    st.markdown("Train a model on a **subset** of the labelled history, name it, then pick it in *Score a claim*. "
                "The last month of the date range is **held out**: progress is measured on it, so the curve shows "
                "whether the model generalises forward in time rather than how well it memorises.")
    with st.form("train"):
        name = st.text_input("Model name", placeholder="e.g. crm_only_2026 (letters, digits, _ or -)")
        c1, c2 = st.columns(2)
        lo, hi = pd.Timestamp("2025-04-01").date(), pd.Timestamp("2026-06-30").date()
        d_from = c1.date_input("Claims from", lo, min_value=lo, max_value=hi)
        d_to = c2.date_input("Claims to (its month is the hold-out)", hi, min_value=lo, max_value=hi)
        c1, c2 = st.columns(2)
        sources = c1.multiselect("Source systems", ["crm", "legacy_zoho"], ["crm", "legacy_zoho"])
        ptypes = c2.multiselect("Partner types", ["authorised_service_centre", "franchise", "freelance_technician"],
                                ["authorised_service_centre", "franchise", "freelance_technician"])
        families = st.multiselect("Product families (empty = all)", ["Air Fryer", "Ceiling Fan", "Induction Cooktop",
                                                                     "Mixer Grinder", "Robot Vacuum", "Room Heater", "Water Purifier"])
        with st.expander("Model settings"):
            c1, c2, c3 = st.columns(3)
            n_est = c1.number_input("Boosting rounds", 10, 3000, 250, 10)
            lr = c2.number_input("Learning rate", 0.001, 1.0, 0.03, 0.005, format="%.3f")
            leaves = c3.number_input("Leaves per tree", 2, 255, 7)
            c1, c2, c3 = st.columns(3)
            mono = c1.checkbox("Monotone constraints", True,
                               help="risk can only rise with confirmed partner fraud, prior claims and amount/price")
            zw = c2.slider("Weight on Zoho 'not fraud' labels", 0.0, 1.0, 0.8, 0.05, help="Zoho exported undecided cases as 0")
            refit = c3.checkbox("Refit on the full range after measuring", True)
        start = st.form_submit_button("Start training", type="primary")

    if start:
        body = dict(name=name.strip(), date_from=str(d_from), date_to=str(d_to), sources=sources, partner_types=ptypes,
                    families=families or None, n_estimators=int(n_est), learning_rate=float(lr), num_leaves=int(leaves),
                    monotone=mono, zoho_negative_weight=float(zw), refit_full=refit)
        try:
            r = requests.post(f"{API_URL}/train", json=body, timeout=10)
        except requests.RequestException as e:
            st.error(f"Could not reach the API at {API_URL}: {e}")
        else:
            if r.status_code != 202:
                d = r.json().get("detail", r.text)
                st.error(f"Training not started: {d if isinstance(d, str) else json.dumps(d)}")
            else:
                job_id = r.json()["job_id"]
                bar, chart, table = st.progress(0, "queued"), st.empty(), st.empty()
                while True:  # poll GET /train/{job_id}
                    j = requests.get(f"{API_URL}/train/{job_id}", timeout=10).json()
                    bar.progress(j["progress_pct"] / 100, f"{j['progress_pct']}% - {j['stage'] or j['status']}")
                    if j["history"]:
                        h = pd.DataFrame(j["history"])
                        chart.line_chart(h.set_index("pct")[[c for c in h.columns if "logloss" in c]],
                                         x_label="% of boosting rounds", y_label="log-loss (lower is better)")
                        table.dataframe(h, width="stretch", hide_index=True)
                    if j["status"] in ("done", "failed"):
                        break
                    time.sleep(1)
                if j["status"] == "done":
                    v = j.get("validation", {})
                    st.success(f"Saved as **{j['name']}**. Hold-out {v.get('holdout_month')}: {v.get('holdout_frauds')} frauds in "
                               f"{v.get('holdout_claims')} claims, PR-AUC {v.get('pr_auc', 'n/a')}, ROC-AUC {v.get('roc_auc', 'n/a')}. "
                               "Pick it in the *Score a claim* tab.")
                else:
                    st.error(f"Training failed: {j['error']}")

    saved = api_models()
    if saved:
        st.markdown("**Saved models**")
        st.dataframe(pd.DataFrame([{"name": m["name"], "claims": (m.get("trained_on") or {}).get("claims"),
                                    "frauds": (m.get("trained_on") or {}).get("frauds"),
                                    "hold-out PR-AUC": (m.get("validation") or {}).get("pr_auc"),
                                    "created": m.get("created")} for m in saved]), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- evidence
with tab_perf:
    st.markdown("Every number here comes from **walk-forward validation**: for each month from Oct 2025 to Jun 2026 "
                "the model was trained only on earlier months and scored that month blind.")
    ev_path = ROOT / "reports" / "evaluation.json"
    if ev_path.exists():
        ev = json.loads(ev_path.read_text())
        rows = []
        for k, label in [("pre_policy", "Before May 2026"), ("post_policy", "After May 2026 (like Jul-Sep)"), ("all_folds", "All 9 months")]:
            e = ev[k]
            rows.append({"period": label, "claims": e["claims"], "frauds": e["frauds"], "PR-AUC": round(e["pooled_pr_auc"], 3),
                         "recall in top 40/month": round(e["recall_at_40"], 2), "precision at best-F1": round(e["best_f1"]["precision"], 2),
                         "recall at best-F1": round(e["best_f1"]["recall"], 2), "F1": round(e["best_f1"]["f1"], 2),
                         "desk net Rs/month": round(e["desk_net_inr_per_month"]),
                         "accuracy if we flag nothing": f"{100 * e['accuracy_if_flag_nothing']:.1f}%"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    months = load_csv("reports/walk_forward_months.csv")
    if months is not None:
        st.markdown("**Month by month**: realised desk saving when the model ranked each month blind")
        st.bar_chart(months.set_index("month")["desk_net_inr"], y_label="net Rs saved by the desk")
        st.dataframe(months[["month", "n", "n_fraud", "desk_reviewed", "desk_caught", "desk_net_inr", "p_at_k", "r_at_k", "pr_auc"]]
                     .rename(columns={"n": "claims", "n_fraud": "frauds", "p_at_k": "precision@40", "r_at_k": "recall@40"}).round(3),
                     width="stretch", hide_index=True)
    oof = load_csv("reports/walk_forward_oof.csv")
    if oof is not None:
        st.markdown("**Are the scores honest?** Claims grouped by predicted risk vs how many actually turned out fraudulent.")
        oof["band"] = pd.cut(oof["p"], [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1]).astype(str)
        st.dataframe(oof.groupby("band", sort=False).agg(claims=("p", "size"), predicted=("p", "mean"), actual=("is_fraud", "mean"))
                     .round(3).reset_index().sort_values("predicted"), width="stretch", hide_index=True)
    mc = ROOT / "docs" / "model_comparison.md"
    if mc.exists():
        st.markdown(mc.read_text())
