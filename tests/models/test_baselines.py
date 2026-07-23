import pandas as pd
import pytest

from statflow.models.baselines import HomeAlwaysWinsBaseline, MeanRunsBaseline


def test_home_baseline_learns_training_rate():
    X = pd.DataFrame({"foo": range(10)})
    y = pd.Series([1] * 6 + [0] * 4)  # 60% home wins
    m = HomeAlwaysWinsBaseline().fit(X, y)

    probs = m.predict_proba(pd.DataFrame({"foo": range(5)}))
    assert len(probs) == 5
    assert (probs == 0.6).all()


def test_home_baseline_errors_before_fit():
    with pytest.raises(RuntimeError):
        HomeAlwaysWinsBaseline().predict_proba(pd.DataFrame({"x": [1]}))


def test_mean_runs_baseline_learns_mean():
    X = pd.DataFrame({"foo": range(4)})
    y = pd.Series([4, 6, 8, 10])
    m = MeanRunsBaseline().fit(X, y)

    preds = m.predict(pd.DataFrame({"foo": range(3)}))
    assert (preds == 7.0).all()


def test_mean_runs_baseline_errors_before_fit():
    with pytest.raises(RuntimeError):
        MeanRunsBaseline().predict(pd.DataFrame({"x": [1]}))
