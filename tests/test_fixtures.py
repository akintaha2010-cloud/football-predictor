import pandas as pd
import pytest

from fixtures import predict_fixtures, unplayed
from football_predictor.model import DixonColes
from tests.test_model import synthetic_results


def test_predict_fixtures_skips_unknown_teams():
    model = DixonColes().fit(synthetic_results())
    fixtures = pd.DataFrame({
        "Date": pd.to_datetime(["2026-10-24", "2026-10-25"]),
        "HomeTeam": ["A", "Z"],
        "AwayTeam": ["D", "B"],
    })
    table = predict_fixtures(model, fixtures)
    assert list(table["HomeTeam"]) == ["A"]
    row = table.iloc[0]
    assert row.home_win + row.draw + row.away_win == pytest.approx(1, abs=1e-6)
    assert row.home_win > row.away_win


def test_unplayed_drops_fixtures_with_results():
    fixtures = pd.DataFrame({
        "Date": pd.to_datetime(["2026-10-17 19:00", "2026-10-24 19:00"]),
        "HomeTeam": ["A", "B"],
        "AwayTeam": ["D", "C"],
    })
    results = pd.DataFrame({"Date": pd.to_datetime(["2026-10-17"]), "HomeTeam": ["A"], "AwayTeam": ["D"]})
    assert list(unplayed(fixtures, results, today="2026-10-01")["HomeTeam"]) == ["B"]


def test_unplayed_drops_past_fixtures():
    fixtures = pd.DataFrame({
        "Date": pd.to_datetime(["2026-10-17 19:00", "2026-10-24 19:00"]),
        "HomeTeam": ["A", "B"],
        "AwayTeam": ["D", "C"],
    })
    results = pd.DataFrame({"Date": pd.to_datetime([]), "HomeTeam": [], "AwayTeam": []})
    assert list(unplayed(fixtures, results, today="2026-10-19")["HomeTeam"]) == ["B"]
