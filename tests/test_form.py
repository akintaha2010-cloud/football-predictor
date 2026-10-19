import pandas as pd

from football_predictor.form import recent_form


def test_recent_form_newest_first_from_team_view():
    results = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-01", "2026-08-08", "2026-08-15", "2026-08-22"]),
        "HomeTeam": ["A", "B", "A", "C"],
        "AwayTeam": ["B", "A", "C", "B"],
        "FTHG": [2, 1, 0, 3],
        "FTAG": [0, 1, 1, 3],
    })
    form = recent_form(results, "A", n=2)
    assert list(form["opponent"]) == ["C", "B"]
    assert list(form["venue"]) == ["H", "A"]
    assert list(form["score"]) == ["0-1", "1-1"]
    assert "".join(form["result"]) == "LD"
    assert "".join(recent_form(results, "A")["result"]) == "LDW"


def test_head_to_head_both_grounds_newest_first():
    from football_predictor.form import head_to_head

    results = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-01", "2026-08-08", "2026-08-15", "2026-08-22"]),
        "HomeTeam": ["A", "B", "A", "C"],
        "AwayTeam": ["B", "A", "C", "B"],
        "FTHG": [2, 1, 0, 3],
        "FTAG": [0, 1, 1, 3],
    })
    meetings = head_to_head(results, "A", "B")
    assert list(meetings["HomeTeam"]) == ["B", "A"]
    assert list(meetings["score"]) == ["1-1", "2-0"]
    assert head_to_head(results, "A", "B", n=1).shape[0] == 1
