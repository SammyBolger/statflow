import json
from datetime import date

import pandas as pd
import responses

from statflow.ingest.plays import ingest_plays


@responses.activate
def test_ingest_plays_writes_parquet(tmp_path):
    responses.add(
        responses.GET,
        "https://statsapi.mlb.com/api/v1/game/111/playByPlay",
        json={"allPlays": [{"about": {"inning": 1}}]},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://statsapi.mlb.com/api/v1/game/222/playByPlay",
        json={"allPlays": [{"about": {"inning": 1}}, {"about": {"inning": 2}}]},
        status=200,
    )

    path = ingest_plays(date(2025, 8, 1), game_pks=[111, 222], out_dir=tmp_path)

    assert path == tmp_path / "plays.parquet"
    df = pd.read_parquet(path)
    assert list(df.columns) == ["game_pk", "payload", "ingested_at"]
    assert df["game_pk"].tolist() == [111, 222]
    payload_1 = json.loads(df["payload"].iloc[1])
    assert len(payload_1["allPlays"]) == 2


def test_ingest_plays_returns_none_when_no_games(tmp_path):
    assert ingest_plays(date(2025, 8, 1), game_pks=[], out_dir=tmp_path) is None
    assert list(tmp_path.iterdir()) == []


@responses.activate
def test_ingest_plays_is_idempotent(tmp_path):
    responses.add(
        responses.GET,
        "https://statsapi.mlb.com/api/v1/game/111/playByPlay",
        json={"allPlays": []},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://statsapi.mlb.com/api/v1/game/111/playByPlay",
        json={"allPlays": []},
        status=200,
    )

    ingest_plays(date(2025, 8, 1), game_pks=[111], out_dir=tmp_path)
    ingest_plays(date(2025, 8, 1), game_pks=[111], out_dir=tmp_path)

    df = pd.read_parquet(tmp_path / "plays.parquet")
    assert len(df) == 1
