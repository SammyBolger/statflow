import pytest

from statflow.models.cli import _parse_seasons, main


def test_parse_seasons_single():
    assert _parse_seasons("2024") == [2024]


def test_parse_seasons_multiple():
    assert _parse_seasons("2024,2025,2026") == [2024, 2025, 2026]


def test_main_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])


def test_main_rejects_unknown_subcommand():
    with pytest.raises(SystemExit):
        main(["not-a-command"])
