import pandas as pd
import pytest

from backtest import calibration, score


def predictions():
    return pd.DataFrame({
        "actual": ["H", "H", "D", "A"],
        "pH": [0.7, 0.7, 0.3, 0.3],
        "pD": [0.2, 0.2, 0.4, 0.3],
        "pA": [0.1, 0.1, 0.3, 0.4],
    })


def test_calibration_bins_cover_every_probability():
    table = calibration(predictions(), "p", bins=5)
    assert table["count"].sum() == 12
    top = table[table["bin"] == "60%-80%"].iloc[0]
    assert top["predicted"] == pytest.approx(0.7)
    assert top["observed"] == pytest.approx(1.0)


def test_score_perfect_accuracy():
    assert score(predictions(), "p")["accuracy"] == pytest.approx(1.0)
