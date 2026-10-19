import json
import numpy as np
import pandas as pd
import pytest

from football_predictor.data import read_results, recent_seasons, season_code
from football_predictor.model import DixonColes


def synthetic_results(seed=0, rounds=6):
    """Double round-robin where team strength decreases A > B > C > D."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 0.5, "B": 0.2, "C": -0.2, "D": -0.5}
    rows = []
    day = pd.Timestamp("2025-08-01")
    for _ in range(rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lam = np.exp(0.25 + strength[home] - strength[away] * 0.5)
                mu = np.exp(strength[away] - strength[home] * 0.5)
                rows.append({"Date": day, "HomeTeam": home, "AwayTeam": away,
                             "FTHG": rng.poisson(lam), "FTAG": rng.poisson(mu)})
                day += pd.Timedelta(days=3)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def model():
    return DixonColes(l2=0.1).fit(synthetic_results())


def test_outcome_probabilities_sum_to_one(model):
    p = model.predict("A", "D")
    assert p.home_win + p.draw + p.away_win == pytest.approx(1.0)
    assert p.score_matrix.sum() == pytest.approx(1.0)


def test_stronger_team_is_favourite(model):
    p = model.predict("A", "D")
    assert p.home_win > p.away_win
    reverse = model.predict("D", "A")
    assert reverse.away_win > reverse.home_win


def test_home_advantage_is_positive(model):
    assert model.home_adv > 0
    assert model.predict("B", "C").home_win > model.predict("C", "B").away_win


def test_neutral_venue_is_symmetric(model):
    p = model.predict("B", "C", neutral=True)
    q = model.predict("C", "B", neutral=True)
    assert p.home_win == pytest.approx(q.away_win)
    assert p.draw == pytest.approx(q.draw)
    assert p.home_win < model.predict("B", "C").home_win


def test_ratings_order(model):
    assert list(model.ratings().index) == ["A", "B", "C", "D"]


def test_attack_ratings_sum_to_zero(model):
    assert sum(model.attack.values()) == pytest.approx(0.0, abs=1e-6)


def test_markets_are_consistent(model):
    p = model.predict("A", "B")
    assert 0 < p.over(2.5) < 1
    assert p.over(0.5) > p.over(2.5) > p.over(4.5)
    assert 0 < p.both_teams_score < 1
    scores = p.top_scores(3)
    assert scores[0][2] >= scores[1][2] >= scores[2][2]


def test_l2_shrinks_ratings():
    results = synthetic_results()
    loose = DixonColes(l2=0.0).fit(results)
    tight = DixonColes(l2=50.0).fit(results)
    assert np.abs(list(tight.attack.values())).sum() < np.abs(list(loose.attack.values())).sum()


def test_matches_after_as_of_are_ignored():
    results = synthetic_results()
    cutoff = results["Date"].iloc[len(results) // 2]
    a = DixonColes().fit(results, as_of=cutoff)
    b = DixonColes().fit(results[results["Date"] <= cutoff])
    assert a.attack == pytest.approx(b.attack)


def test_team_name_lookup(model):
    assert model.resolve_team("a") == "A"
    with pytest.raises(KeyError, match="Unknown team"):
        model.resolve_team("Zebra")


def test_season_helpers():
    assert season_code(2025) == "2526"
    assert season_code(1999) == "9900"
    assert recent_seasons(2, today=pd.Timestamp("2026-10-19").date()) == ["2526", "2627"]
    assert recent_seasons(1, today=pd.Timestamp("2026-03-01").date()) == ["2526"]


def test_read_results_skips_unplayed_matches(tmp_path):
    csv = tmp_path / "T1_2627.csv"
    csv.write_text(
        "\ufeffDiv,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,AvgH,AvgD,AvgA\n"
        "T1,14/08/2026,19:30,Galatasaray ,Corum,2,2,1.3,5.5,9\n"
        "T1,20/08/2026,19:30,Corum,Galatasaray,,,6,4.5,1.5\n",
        encoding="utf-8",
    )
    df = read_results(csv)
    assert len(df) == 1
    assert df.iloc[0]["HomeTeam"] == "Galatasaray"
    assert df.iloc[0]["Date"] == pd.Timestamp("2026-08-14")


def test_handicap_lines(model):
    p = model.predict("A", "D")
    assert p.handicap(0) == pytest.approx((p.home_win, p.draw, p.away_win))
    win, push, lose = p.handicap(-0.5)
    assert push == 0
    assert win == pytest.approx(p.home_win)
    win, push, lose = p.handicap(-1)
    assert win + push + lose == pytest.approx(p.score_matrix.sum())
    assert win < p.home_win
    assert p.handicap(-1.5)[0] == pytest.approx(win)
    assert p.handicap(0.5)[0] == pytest.approx(p.home_win + p.draw)


def test_to_dict_is_json_ready(model):
    p = model.predict("A", "D")
    data = json.loads(json.dumps(p.to_dict(top_n=3)))
    assert data["result"]["home_win"] == pytest.approx(p.home_win)
    assert data["over"]["2.5"] == pytest.approx(p.over(2.5))
    assert data["handicap"]["-1.0"]["push"] == pytest.approx(p.handicap(-1)[1])
    assert len(data["top_scores"]) == 3


def test_double_chance_and_draw_no_bet(model):
    p = model.predict("A", "D")
    assert p.double_chance["1X"] == pytest.approx(1 - p.away_win, abs=1e-6)
    assert sum(p.double_chance.values()) == pytest.approx(2 * (p.home_win + p.draw + p.away_win))
    home, away = p.draw_no_bet
    assert home + away == pytest.approx(1)
    assert home > p.home_win


def test_every_league_has_a_name():
    from football_predictor.data import LEAGUES

    assert len(LEAGUES) == 18
    assert all(code.isalnum() and name for code, name in LEAGUES.items())


def test_clean_sheets(model):
    p = model.predict("A", "D")
    cs_home, cs_away = p.clean_sheet
    assert cs_home == pytest.approx(p.score_matrix[:, 0].sum())
    assert cs_home > cs_away
    # Either side keeps a clean sheet exactly when both teams do not score.
    assert cs_home + cs_away - p.score_matrix[0, 0] == pytest.approx(1 - p.both_teams_score)


def test_fair_odds_are_inverse_probabilities(model):
    p = model.predict("A", "D")
    odds = p.fair_odds
    assert odds["home"] * p.home_win == pytest.approx(1.0)
    assert sum(1 / o for o in odds.values()) == pytest.approx(1.0)
    assert odds["home"] < odds["away"]


def test_score_probability(model):
    p = model.predict("A", "D")
    h, a, prob = p.top_scores(1)[0]
    assert p.score_probability(h, a) == pytest.approx(prob)
    assert p.score_probability(99, 0) == 0.0
    assert p.score_probability(-1, 0) == 0.0


def test_top_scores_count_and_order(model):
    p = model.predict("A", "D")
    top = p.to_dict(top_n=3)["top_scores"]
    assert len(top) == 3
    probs = [s["probability"] for s in top]
    assert probs == sorted(probs, reverse=True)
