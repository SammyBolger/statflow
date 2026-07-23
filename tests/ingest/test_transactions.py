import json
from datetime import date

import pandas as pd
import responses

from statflow.ingest.transactions import ingest_transactions

TXN_URL = "https://statsapi.mlb.com/api/v1/transactions"


@responses.activate
def test_ingest_transactions_writes_parquet(tmp_path):
    responses.add(
        responses.GET,
        TXN_URL,
        json={
            "transactions": [
                {"id": 1001, "typeCode": "SC", "description": "IL10 placement"},
                {"id": 1002, "typeCode": "SC", "description": "IL10 activation"},
            ]
        },
        status=200,
    )

    path = ingest_transactions(date(2025, 8, 1), out_dir=tmp_path)

    assert path == tmp_path / "transactions.parquet"
    df = pd.read_parquet(path)
    assert list(df.columns) == ["transaction_id", "date", "payload", "ingested_at"]
    assert df["transaction_id"].tolist() == [1001, 1002]
    payload_0 = json.loads(df["payload"].iloc[0])
    assert payload_0["typeCode"] == "SC"


@responses.activate
def test_ingest_transactions_returns_none_when_empty(tmp_path):
    responses.add(responses.GET, TXN_URL, json={"transactions": []}, status=200)
    assert ingest_transactions(date(2025, 8, 1), out_dir=tmp_path) is None
    assert list(tmp_path.iterdir()) == []


@responses.activate
def test_ingest_transactions_sends_date_range(tmp_path):
    """The API needs startDate and endDate; we send both equal to the target date."""
    responses.add(
        responses.GET,
        TXN_URL,
        json={"transactions": [{"id": 1, "typeCode": "SC"}]},
        status=200,
    )

    ingest_transactions(date(2025, 8, 1), out_dir=tmp_path)

    url = responses.calls[0].request.url
    assert "startDate=2025-08-01" in url
    assert "endDate=2025-08-01" in url


@responses.activate
def test_ingest_transactions_is_idempotent(tmp_path):
    responses.add(
        responses.GET,
        TXN_URL,
        json={"transactions": [{"id": 1, "typeCode": "SC"}]},
        status=200,
    )
    responses.add(
        responses.GET,
        TXN_URL,
        json={"transactions": [{"id": 1, "typeCode": "SC"}]},
        status=200,
    )

    ingest_transactions(date(2025, 8, 1), out_dir=tmp_path)
    ingest_transactions(date(2025, 8, 1), out_dir=tmp_path)

    df = pd.read_parquet(tmp_path / "transactions.parquet")
    assert len(df) == 1
