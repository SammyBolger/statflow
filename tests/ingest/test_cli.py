from datetime import date

import pandas as pd
import pytest
import responses

from statflow.ingest.cli import _parse_args, run_daily_ingest

BASE = "https://statsapi.mlb.com/api/v1"


def test_parse_args_requires_date():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_parse_args_parses_iso_date():
    args = _parse_args(["--date", "2025-08-01"])
    assert args.date == date(2025, 8, 1)


def test_parse_args_rejects_bad_date():
    with pytest.raises(SystemExit):
        _parse_args(["--date", "not-a-date"])


@responses.activate
def test_run_daily_ingest_writes_all_four_partitions(tmp_path):
    responses.add(
        responses.GET,
        f"{BASE}/schedule",
        json={
            "dates": [
                {
                    "date": "2025-08-01",
                    "games": [
                        {"gamePk": 111, "gameDate": "2025-08-01T23:05:00Z"},
                        {"gamePk": 222, "gameDate": "2025-08-01T23:10:00Z"},
                    ],
                }
            ]
        },
        status=200,
    )
    for pk in (111, 222):
        responses.add(responses.GET, f"{BASE}/game/{pk}/boxscore", json={"teams": {}}, status=200)
        responses.add(
            responses.GET, f"{BASE}/game/{pk}/playByPlay", json={"allPlays": []}, status=200
        )
    responses.add(
        responses.GET,
        f"{BASE}/transactions",
        json={"transactions": [{"id": 1, "typeCode": "SC"}]},
        status=200,
    )

    run_daily_ingest(date(2025, 8, 1), base_dir=tmp_path)

    schedule_path = tmp_path / "schedule" / "date=2025-08-01" / "schedule.parquet"
    box_path = tmp_path / "boxscores" / "date=2025-08-01" / "boxscores.parquet"
    plays_path = tmp_path / "plays" / "date=2025-08-01" / "plays.parquet"
    txn_path = tmp_path / "transactions" / "date=2025-08-01" / "transactions.parquet"
    assert schedule_path.exists()
    assert box_path.exists()
    assert plays_path.exists()
    assert txn_path.exists()

    assert pd.read_parquet(schedule_path)["game_pk"].tolist() == [111, 222]
    assert pd.read_parquet(box_path)["game_pk"].tolist() == [111, 222]
    assert pd.read_parquet(plays_path)["game_pk"].tolist() == [111, 222]


@responses.activate
def test_run_daily_ingest_off_day_skips_games_but_fetches_transactions(tmp_path):
    """No games on this date but transactions can still exist (e.g., IL moves)."""
    responses.add(responses.GET, f"{BASE}/schedule", json={"dates": []}, status=200)
    responses.add(
        responses.GET,
        f"{BASE}/transactions",
        json={"transactions": [{"id": 42, "typeCode": "SC"}]},
        status=200,
    )

    run_daily_ingest(date(2025, 12, 25), base_dir=tmp_path)

    assert not (tmp_path / "schedule").exists()
    assert not (tmp_path / "boxscores").exists()
    assert not (tmp_path / "plays").exists()
    txn_path = tmp_path / "transactions" / "date=2025-12-25" / "transactions.parquet"
    assert txn_path.exists()
