import json
from datetime import date

import pandas as pd
import responses

from statflow.ingest.schedule import ingest_schedule

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def _schedule_payload(target: str, games: list[dict]) -> dict:
    return {"dates": [{"date": target, "games": games}]}


@responses.activate
def test_ingest_schedule_writes_parquet(tmp_path):
    responses.add(
        responses.GET,
        SCHEDULE_URL,
        json=_schedule_payload(
            "2025-08-01",
            [
                {"gamePk": 111, "gameDate": "2025-08-01T23:05:00Z"},
                {"gamePk": 222, "gameDate": "2025-08-01T23:10:00Z"},
            ],
        ),
        status=200,
    )

    path = ingest_schedule(date(2025, 8, 1), out_dir=tmp_path)

    assert path == tmp_path / "schedule.parquet"
    df = pd.read_parquet(path)
    assert list(df.columns) == ["game_pk", "game_date", "payload", "ingested_at"]
    assert len(df) == 2
    assert df["game_pk"].tolist() == [111, 222]
    assert (df["game_date"] == "2025-08-01").all()
    # payload column holds the raw per-game JSON
    payload_0 = json.loads(df["payload"].iloc[0])
    assert payload_0["gamePk"] == 111


@responses.activate
def test_ingest_schedule_returns_none_when_no_games(tmp_path):
    responses.add(responses.GET, SCHEDULE_URL, json={"dates": []}, status=200)

    result = ingest_schedule(date(2025, 12, 25), out_dir=tmp_path)

    assert result is None
    assert list(tmp_path.iterdir()) == []


@responses.activate
def test_ingest_schedule_is_idempotent(tmp_path):
    """Re-running for the same date overwrites the file rather than appending."""
    payload = _schedule_payload("2025-08-01", [{"gamePk": 111, "gameDate": "..."}])
    responses.add(responses.GET, SCHEDULE_URL, json=payload, status=200)
    responses.add(responses.GET, SCHEDULE_URL, json=payload, status=200)

    ingest_schedule(date(2025, 8, 1), out_dir=tmp_path)
    ingest_schedule(date(2025, 8, 1), out_dir=tmp_path)

    df = pd.read_parquet(tmp_path / "schedule.parquet")
    assert len(df) == 1


@responses.activate
def test_ingest_schedule_sends_hydrate_param(tmp_path):
    """We must request probablePitcher — it's the main injury-adjacent signal."""
    responses.add(
        responses.GET,
        SCHEDULE_URL,
        json=_schedule_payload("2025-08-01", [{"gamePk": 111, "gameDate": "..."}]),
        status=200,
    )

    ingest_schedule(date(2025, 8, 1), out_dir=tmp_path)

    call = responses.calls[0]
    assert "probablePitcher" in call.request.url
