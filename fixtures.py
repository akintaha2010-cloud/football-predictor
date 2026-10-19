"""Predict every upcoming match of a league.

    python fixtures.py --league T1
    python fixtures.py --league E0 --csv predictions.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from football_predictor.data import LEAGUES, load_fixtures, load_results, recent_seasons
from football_predictor.model import DEFAULT_L2, DEFAULT_XI, DixonColes


def predict_fixtures(model: DixonColes, fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture with outcome probabilities; matches with unknown teams are dropped."""
    rows = []
    for match in fixtures.itertuples():
        try:
            p = model.predict(match.HomeTeam, match.AwayTeam)
        except KeyError:
            continue
        home_goals, away_goals, _ = p.top_scores(1)[0]
        rows.append({
            "Date": match.Date, "HomeTeam": p.home, "AwayTeam": p.away,
            "xG_home": p.home_goals, "xG_away": p.away_goals,
            "home_win": p.home_win, "draw": p.draw, "away_win": p.away_win,
            "over_2_5": p.over(2.5), "btts": p.both_teams_score,
            "likely_score": f"{home_goals}-{away_goals}",
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", default="T1", choices=LEAGUES)
    parser.add_argument("--seasons", type=int, default=3)
    parser.add_argument("--csv", metavar="FILE", help="Also save the predictions as CSV")
    parser.add_argument("--refresh", action="store_true", help="Re-download results")
    parser.add_argument("--xi", type=float, default=DEFAULT_XI)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    args = parser.parse_args(argv)

    results = load_results(args.league, recent_seasons(args.seasons), refresh=args.refresh)
    model = DixonColes(xi=args.xi, l2=args.l2).fit(results)
    table = predict_fixtures(model, load_fixtures(args.league))
    if table.empty:
        print(f"No upcoming {LEAGUES[args.league]} fixtures listed right now.")
        return 0

    print(f"{LEAGUES[args.league]}: {len(table)} upcoming matches\n")
    print(f"{'Date':<6} {'Match':<38} {'1':>6} {'X':>6} {'2':>6} {'O2.5':>6} {'Score':>6}")
    for row in table.itertuples():
        match = f"{row.HomeTeam} vs {row.AwayTeam}"
        print(f"{row.Date:%d %b} {match:<38} {row.home_win:6.1%} {row.draw:6.1%} {row.away_win:6.1%} "
              f"{row.over_2_5:6.1%} {row.likely_score:>6}")
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nSaved {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
