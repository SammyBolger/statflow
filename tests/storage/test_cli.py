from unittest.mock import patch

import pytest

from statflow.storage.cli import main


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])


def test_cli_push_calls_push():
    with patch("statflow.storage.cli.push", return_value=3) as mock_push:
        main(["push"])
    mock_push.assert_called_once()


def test_cli_pull_calls_pull():
    with patch("statflow.storage.cli.pull", return_value=5) as mock_pull:
        main(["pull"])
    mock_pull.assert_called_once()
