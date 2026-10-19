import numpy as np
import pandas as pd
import pytest

from football_predictor.model import DixonColes
from football_predictor.simulation import remaining_fixtures, simulate_season
from tests.test_model import synthetic_results


def half_season():
    results = synthetic_results(rounds=1)
    return results.iloc[: len(results) // 2]


def test_remaining_fixtures():
    played = half_season()
    remaining = remaining_fixtures(played)
    assert len(remaining) + len(played) == 4 * 3
    assert not set(remaining) & set(zip(played["HomeTeam"], played["AwayTeam"]))


def test_position_probabilities_are_valid():
    model = DixonColes(l2=0.1).fit(synthetic_results())
    summary, positions = simulate_season(model, half_season(), n_sims=2000, seed=1)
    assert positions.sum(axis=0) == pytest.approx(np.ones(4))  # every position is filled once
    assert positions.sum(axis=1) == pytest.approx(np.ones(4))  # every team finishes somewhere
    assert (summary["expected_points"] >= summary["points"]).all()
    title = dict(zip(summary.index, positions[:, 0]))
    assert title["A"] > title["D"]


def test_finished_season_is_deterministic():
    results = pd.DataFrame({
        "Date": pd.to_datetime(["2025-08-01", "2025-08-02"]),
        "HomeTeam": ["X", "Y"], "AwayTeam": ["Y", "X"],
        "FTHG": [2, 0], "FTAG": [0, 1],
    })
    model = DixonColes().fit(results)
    summary, positions = simulate_season(model, results, n_sims=100, seed=0)
    assert summary.loc["X", "points"] == 6
    assert summary.loc["X", "expected_points"] == 6
    assert positions[list(summary.index).index("X"), 0] == 1.0


def test_plot_positions_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from football_predictor.plot import plot_positions

    played = half_season()
    model = DixonColes().fit(synthetic_results())
    summary, positions = simulate_season(model, played, n_sims=200, seed=0)
    path = plot_positions(summary, positions, tmp_path / "chart.png", "test")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_rejects_misaligned_positions(tmp_path):
    pytest.importorskip("matplotlib")
    from football_predictor.plot import plot_positions

    model = DixonColes().fit(synthetic_results())
    summary, positions = simulate_season(model, half_season(), n_sims=50, seed=0)
    with pytest.raises(ValueError):
        plot_positions(summary, positions[:-1], tmp_path / "chart.png", "test")
