import pytest

from statflow.features.cli import main


def test_main_rejects_unknown_args():
    with pytest.raises(SystemExit):
        main(["--unknown"])
