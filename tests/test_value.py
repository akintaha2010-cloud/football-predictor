import pandas as pd
import pytest

from football_predictor.data import load_fixtures
from value import edge, kelly_fraction


def test_edge():
    assert edge(0.5, 2.0) == pytest.approx(0.0)
    assert edge(0.6, 2.0) == pytest.approx(0.2)
    assert edge(0.4, 2.0) == pytest.approx(-0.2)


def test_kelly_fraction():
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    assert kelly_fraction(0.4, 2.0) == 0.0
    assert kelly_fraction(0.9, 1.0) == 0.0


def test_load_fixtures_filters_league_and_falls_back_to_bet365(tmp_path):
    csv = tmp_path / "fixtures.csv"
    csv.write_text(
        "\ufeffDiv,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,AvgH,AvgD,AvgA\n"
        "E0,17/10/2026,12:30,Everton,Chelsea,3.1,3.4,2.3,3.0,3.5,2.25\n"
        "T1,18/10/2026,19:00,Galatasaray ,Konyaspor,1.3,5.5,9.0,,,\n",
        encoding="utf-8",
    )
    df = load_fixtures("T1", path=csv)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["HomeTeam"] == "Galatasaray"
    assert row["Date"] == pd.Timestamp("2026-10-18")
    assert (row["OddsH"], row["OddsD"], row["OddsA"]) == (1.3, 5.5, 9.0)
