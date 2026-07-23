from datetime import date
from unittest.mock import patch

import pytest
import responses

from statflow.ingest.backfill import backfill, dates_with_games, main
from statflow.ingest.mlb_api import MLBStatsAPI

BASE = "https://statsapi.mlb.com/api/v1"


@responses.activate
def test_dates_with_games_returns_unique_sorted_dates():
    responses.add(
        responses.GET,
        f"{BASE}/schedule",
        json={
            "dates": [
                {"date": "2024-04-01", "games": [{"gamePk": 1}]},
                {"date": "2024-04-02", "games": []},
                {"date": "2024-04-03", "games": [{"gamePk": 2}, {"gamePk": 3}]},
            ]
        },
        status=200,
    )

    result = dates_with_games(MLBStatsAPI(), 2024)
    assert result == [date(2024, 4, 1), date(2024, 4, 3)]


@responses.activate
def test_dates_with_games_requests_regular_and_postseason():
    responses.add(responses.GET, f"{BASE}/schedule", json={"dates": []}, status=200)
    dates_with_games(MLBStatsAPI(), 2024)

    url = responses.calls[0].request.url
    assert "season=2024" in url
    assert "gameType=R%2CP" in url or "gameType=R,P" in url


@responses.activate
def test_backfill_skips_existing_partitions(tmp_path):
    # Two dates in the season schedule.
    responses.add(
        responses.GET,
        f"{BASE}/schedule",
        json={
            "dates": [
                {"date": "2024-04-01", "games": [{"gamePk": 1}]},
                {"date": "2024-04-02", "games": [{"gamePk": 2}]},
            ]
        },
        status=200,
    )
    # Pretend 2024-04-01 was already ingested.
    (tmp_path / "schedule" / "date=2024-04-01").mkdir(parents=True)
    (tmp_path / "schedule" / "date=2024-04-01" / "schedule.parquet").touch()

    with patch("statflow.ingest.backfill.run_daily_ingest") as mock_ingest:
        backfill(2024, 2024, bronze_dir=tmp_path, sleep_between_dates=0)

    # Only 2024-04-02 got ingested.
    assert mock_ingest.call_count == 1
    assert mock_ingest.call_args.args[0] == date(2024, 4, 2)


@responses.activate
def test_backfill_continues_on_single_date_error(tmp_path):
    responses.add(
        responses.GET,
        f"{BASE}/schedule",
        json={
            "dates": [
                {"date": "2024-04-01", "games": [{"gamePk": 1}]},
                {"date": "2024-04-02", "games": [{"gamePk": 2}]},
                {"date": "2024-04-03", "games": [{"gamePk": 3}]},
            ]
        },
        status=200,
    )

    def flaky(target_date, api=None, base_dir=None):
        if target_date == date(2024, 4, 2):
            raise RuntimeError("boom")

    with patch("statflow.ingest.backfill.run_daily_ingest", side_effect=flaky) as mock_ingest:
        backfill(2024, 2024, bronze_dir=tmp_path, sleep_between_dates=0)

    # Still called for all three dates — the failure didn't abort the loop.
    assert mock_ingest.call_count == 3


@responses.activate
def test_backfill_walks_multiple_seasons(tmp_path):
    responses.add(
        responses.GET,
        f"{BASE}/schedule",
        json={"dates": [{"date": "2024-04-01", "games": [{"gamePk": 1}]}]},
        status=200,
    )
    responses.add(
        responses.GET,
        f"{BASE}/schedule",
        json={"dates": [{"date": "2025-04-01", "games": [{"gamePk": 2}]}]},
        status=200,
    )

    with patch("statflow.ingest.backfill.run_daily_ingest") as mock_ingest:
        backfill(2024, 2025, bronze_dir=tmp_path, sleep_between_dates=0)

    assert mock_ingest.call_count == 2
    assert mock_ingest.call_args_list[0].args[0] == date(2024, 4, 1)
    assert mock_ingest.call_args_list[1].args[0] == date(2025, 4, 1)


def test_main_requires_season_range():
    with pytest.raises(SystemExit):
        main([])


def test_main_parses_season_range():
    with patch("statflow.ingest.backfill.backfill") as mock_backfill:
        main(["--start-season", "2024", "--end-season", "2026"])
    mock_backfill.assert_called_once()
    call = mock_backfill.call_args
    assert call.args == (2024, 2026)
    assert call.kwargs["skip_existing"] is True


def test_main_force_disables_skip_existing():
    with patch("statflow.ingest.backfill.backfill") as mock_backfill:
        main(["--start-season", "2024", "--end-season", "2024", "--force"])
    assert mock_backfill.call_args.kwargs["skip_existing"] is False
