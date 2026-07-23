import pytest
import requests
import responses

from statflow.ingest.mlb_api import USER_AGENT, MLBStatsAPI


@responses.activate
def test_get_returns_parsed_json():
    responses.add(
        responses.GET,
        "https://statsapi.mlb.com/api/v1/schedule",
        json={"dates": [{"date": "2025-08-01", "games": []}]},
        status=200,
    )

    api = MLBStatsAPI()
    data = api.get("schedule", params={"sportId": 1, "date": "2025-08-01"})

    assert data == {"dates": [{"date": "2025-08-01", "games": []}]}


@responses.activate
def test_get_raises_on_client_error():
    responses.add(
        responses.GET,
        "https://statsapi.mlb.com/api/v1/nope",
        status=404,
    )

    api = MLBStatsAPI()
    with pytest.raises(requests.HTTPError):
        api.get("nope")


@responses.activate
def test_user_agent_header_is_set():
    responses.add(
        responses.GET,
        "https://statsapi.mlb.com/api/v1/schedule",
        json={},
        status=200,
    )

    api = MLBStatsAPI()
    api.get("schedule")

    assert responses.calls[0].request.headers["User-Agent"] == USER_AGENT


def test_leading_slash_in_path_is_tolerated():
    """`api.get("/schedule")` and `api.get("schedule")` should both work."""
    api = MLBStatsAPI(base_url="https://example.test/api/v1")
    # Just check URL construction — we don't actually make a request.
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, "https://example.test/api/v1/schedule", json={}, status=200)
        api.get("/schedule")
        api.get("schedule")
