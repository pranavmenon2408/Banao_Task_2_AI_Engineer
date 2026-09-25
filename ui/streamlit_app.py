"""Kestrel claim review screen. Calls the FastAPI service for scoring (start it first - see README)."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kestrel_fraud.config import CONTACT_INR, DESK_CAPACITY, GOODWILL_INR  # noqa: E402

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


tab_score, tab_preds, tab_perf = st.tabs(["Score a claim", "Jul-Sep predictions", "How well it works"])

# --------------------------------------------------------------------------- score one claim
with tab_score:
    health = api_health()
    if health is None:
        st.error(f"The scoring service is not running at {API_URL}. Start it with "
                 "`uvicorn app.main:app --port 8000` (see README), then refresh this page.")
    elif not health.get("ok"):
        st.error(f"The scoring service is up but the model is not loaded: {health.get('error')}. {health.get('fix', '')}")

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
                    customer_prior_claims=int(prior), submitted_at=submitted.strip() or None)
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
        c1, c2 = st.columns(2)
        q, w = load_csv("reports/desk_queue.csv"), load_csv("reports/partner_watchlist.csv")
        if q is not None:
            c1.markdown("**Desk worklist** (per month, only claims where review pays for itself)")
            c1.dataframe(q, width="stretch", hide_index=True)
        if w is not None:
            c2.markdown("**Outlets to take off auto-approval** (expected fraud rupees, Jul-Sep)")
            c2.dataframe(w[["partner_id", "city", "partner_type", "claims", "mean_score", "expected_fraud_inr",
                            "known_frauds_before_jul"]], width="stretch", hide_index=True)

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
        st.markdown("**Month by month**")
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
