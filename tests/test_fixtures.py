import pandas as pd
import pytest

from fixtures import predict_fixtures
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
