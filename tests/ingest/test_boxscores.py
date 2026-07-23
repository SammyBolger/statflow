import json
from datetime import date

import pandas as pd
import responses

from statflow.ingest.boxscores import ingest_boxscores


@responses.activate
def test_ingest_boxscores_writes_parquet(tmp_path):
    responses.add(
        responses.GET,
        "https://statsapi.mlb.com/api/v1/game/111/boxscore",
        json={"teams": {"home": {"team": {"id": 1}}, "away": {"team": {"id": 2}}}},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://statsapi.mlb.com/api/v1/game/222/boxscore",
        json={"teams": {"home": {"team": {"id": 3}}, "away": {"team": {"id": 4}}}},
        status=200,
    )

    path = ingest_boxscores(date(2025, 8, 1), game_pks=[111, 222], out_dir=tmp_path)

    assert path == tmp_path / "boxscores.parquet"
    df = pd.read_parquet(path)
    assert list(df.columns) == ["game_pk", "payload", "ingested_at"]
    assert df["game_pk"].tolist() == [111, 222]
    payload_0 = json.loads(df["payload"].iloc[0])
    assert payload_0["teams"]["home"]["team"]["id"] == 1


def test_ingest_boxscores_returns_none_when_no_games(tmp_path):
    result = ingest_boxscores(date(2025, 8, 1), game_pks=[], out_dir=tmp_path)
    assert result is None
    assert list(tmp_path.iterdir()) == []


@responses.activate
def test_ingest_boxscores_is_idempotent(tmp_path):
    responses.add(
        responses.GET,
        "https://statsapi.mlb.com/api/v1/game/111/boxscore",
        json={"teams": {}},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://statsapi.mlb.com/api/v1/game/111/boxscore",
        json={"teams": {}},
        status=200,
    )

    ingest_boxscores(date(2025, 8, 1), game_pks=[111], out_dir=tmp_path)
    ingest_boxscores(date(2025, 8, 1), game_pks=[111], out_dir=tmp_path)

    df = pd.read_parquet(tmp_path / "boxscores.parquet")
    assert len(df) == 1
