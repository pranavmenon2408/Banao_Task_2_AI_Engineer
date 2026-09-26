"""Tests that need no client data, so they run in CI where data/ is absent."""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kestrel_fraud import dataset, metrics
from kestrel_fraud.data import normalise_serial

PARTNERS = pd.DataFrame({"partner_id": ["SP3160", "SP3265"]})
PRODUCTS = pd.DataFrame({"sku": ["KH-AF-02", "KH-CF-01"]})


def test_expected_value_matches_policy_costs():
    # Rs 1,995 claim at 29.4% risk: 586.5 fraud avoided - 268.3 goodwill - 260 contact
    assert metrics.expected_value(0.294, 1995) == pytest.approx(58.2, abs=0.1)
    # a certain fraud saves the claim minus the contact cost; a certain genuine claim costs 640
    assert metrics.expected_value(1.0, 5000) == 5000 - 260
    assert metrics.expected_value(0.0, 5000) == -640


def test_desk_queue_only_takes_claims_worth_reviewing():
    p = np.array([0.9, 0.01, 0.5, 0.3])
    amount = np.array([10000, 50000, 400, 20000])
    picked = metrics.desk_queue(p, amount, k=40)
    assert set(picked) == {0, 3}  # the Rs 400 claim at 50% and the 1% claim are not worth Rs 640


@pytest.mark.parametrize("raw", ["kh-128015513", " KH128015513", "KH-128015513", "kh128015513"])
def test_serials_normalise_to_one_form(raw):
    assert normalise_serial(raw) == "KH128015513"


def test_template_passes_its_own_validation_apart_from_size_warnings():
    df, report = dataset.validate(dataset.template_csv().encode(), PARTNERS, PRODUCTS)
    assert report["ok"] and df is not None and len(df) == 2
    assert any("fraud labels" in w["problem"] for w in report["warnings"])  # 1 fraud is too few alone


def test_missing_column_is_reported_by_name():
    csv = pd.read_csv(io.StringIO(dataset.template_csv())).drop(columns=["sku"]).to_csv(index=False)
    df, report = dataset.validate(csv.encode(), PARTNERS, PRODUCTS)
    assert df is None and not report["ok"]
    assert report["errors"][0]["columns"] == ["sku"]


def test_bad_values_are_all_reported_with_line_numbers():
    rows = pd.read_csv(io.StringIO(dataset.template_csv()))
    rows.loc[0, "photo_attached"] = "yes"
    rows.loc[1, "partner_id"] = "SP0000"
    rows.loc[1, "is_fraud"] = 2
    _, report = dataset.validate(rows.to_csv(index=False).encode(), PARTNERS, PRODUCTS)
    problems = {e["problem"]: e for e in report["errors"]}
    assert problems["invalid values in 'photo_attached'"]["example_lines"] == [2]
    assert problems["invalid values in 'partner_id'"]["example_lines"] == [3]
    assert "invalid values in 'is_fraud'" in problems


def test_non_csv_is_rejected_politely():
    df, report = dataset.validate(b"", PARTNERS, PRODUCTS)
    assert df is None and report["errors"]


def test_service_starts_and_explains_itself():
    """With or without data/, the API must start; without it, it says what to do instead of crashing."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    if not body["ok"]:
        assert "fix" in body
        r = client.post("/score", json={"partner_id": "SP3160"})
        assert r.status_code in (422, 503)
