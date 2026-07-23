from statflow.transform.cli import main


def test_main_rejects_unknown_args():
    """The CLI has no positional args yet — passing one should error out."""
    import pytest

    with pytest.raises(SystemExit):
        main(["--unknown"])
