"""Predict a football match.

    python predict.py Galatasaray Fenerbahce
    python predict.py Arsenal Chelsea --league E0
    python predict.py Galatasaray Fenerbahce --neutral
    python predict.py --league T1 --table
    python predict.py --league T1 --table --csv ratings.csv
    python predict.py Galatasaray Fenerbahce --json
    python predict.py Galatasaray Fenerbahce --form
"""

from __future__ import annotations

import argparse
import json
import sys

from football_predictor.data import LEAGUES, load_results, recent_seasons
from football_predictor.form import recent_form
from football_predictor.model import DEFAULT_L2, DEFAULT_XI, DixonColes


def pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("home", nargs="?", help="Home team")
    parser.add_argument("away", nargs="?", help="Away team")
    parser.add_argument("--league", default="T1", choices=LEAGUES, help="League code (default: T1, Turkey Super Lig)")
    parser.add_argument("--seasons", type=int, default=3, help="Number of recent seasons to train on")
    parser.add_argument("--refresh", action="store_true", help="Re-download the data files")
    parser.add_argument("--table", action="store_true", help="Show team ratings instead of a match")
    parser.add_argument("--neutral", action="store_true", help="Neutral venue: no home advantage")
    parser.add_argument("--form", action="store_true", help="Also show each team's last 5 results")
    parser.add_argument("--csv", metavar="FILE", help="With --table: also save the ratings as CSV")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--xi", type=float, default=DEFAULT_XI, help="Time decay per day")
    parser.add_argument("--l2", type=float, default=DEFAULT_L2, help="Rating shrinkage")
    args = parser.parse_args(argv)

    if not args.table and not (args.home and args.away):
        parser.error("give HOME and AWAY teams, or --table")
    if args.csv and not args.table:
        parser.error("--csv only works with --table")

    results = load_results(args.league, recent_seasons(args.seasons), refresh=args.refresh)
    model = DixonColes(xi=args.xi, l2=args.l2).fit(results)
    trained_on = f"{LEAGUES[args.league]}: trained on {len(results)} matches up to {results['Date'].max():%Y-%m-%d}"
    if args.json:
        print(trained_on, file=sys.stderr)
    else:
        print(trained_on + "\n")

    if args.table:
        current = set(load_results(args.league, recent_seasons(1))[["HomeTeam", "AwayTeam"]].stack())
        table = model.ratings().loc[lambda df: df.index.isin(current)]
        if args.json:
            print(json.dumps(table.rename_axis("team").reset_index().to_dict(orient="records"), indent=2))
            return 0
        print(f"{'Team':<20} {'Goals for':>9} {'Goals against':>13} {'Strength':>8}")
        for team, row in table.iterrows():
            print(f"{team:<20} {row.goals_for:9.2f} {row.goals_against:13.2f} {row.strength:+8.2f}")
        if args.csv:
            table.rename_axis("team").to_csv(args.csv, float_format="%.4f")
            print(f"\nSaved {args.csv}")
        return 0

    try:
        p = model.predict(args.home, args.away, neutral=args.neutral)
    except KeyError as error:
        print(error.args[0], file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"league": args.league, "neutral": args.neutral, **p.to_dict()}, indent=2, ensure_ascii=False))
        return 0

    print(f"{p.home} vs {p.away}" + (" (neutral venue)" if args.neutral else ""))
    print(f"  Expected goals:   {p.home_goals:.2f} - {p.away_goals:.2f}\n")
    odds = p.fair_odds
    print(f"  {p.home} win:".ljust(28) + pct(p.home_win) + f"   fair odds {odds['home']:.2f}")
    print("  Draw:".ljust(28) + pct(p.draw) + f"   fair odds {odds['draw']:.2f}")
    print(f"  {p.away} win:".ljust(28) + pct(p.away_win) + f"   fair odds {odds['away']:.2f}")
    print()
    for label, prob in p.double_chance.items():
        print(f"  Double chance {label}:".ljust(28) + pct(prob))
    dnb_home, dnb_away = p.draw_no_bet
    print("  Draw no bet:".ljust(28) + f"{pct(dnb_home)} - {pct(dnb_away).strip()}")
    print()
    for line in (1.5, 2.5, 3.5):
        print(f"  Over {line} goals:".ljust(28) + pct(p.over(line)))
    print("  Both teams score:".ljust(28) + pct(p.both_teams_score))
    cs_home, cs_away = p.clean_sheet
    print(f"  {p.home} clean sheet:".ljust(28) + pct(cs_home))
    print(f"  {p.away} clean sheet:".ljust(28) + pct(cs_away))
    print(f"\n  Asian handicap for {p.home}:")
    print("    Line".ljust(28) + f"{'win':>6} {'push':>6} {'lose':>6}")
    centre = -round(2 * (p.home_goals - p.away_goals)) / 2
    for line in (centre - 0.5, centre, centre + 0.5):
        win, push, lose = p.handicap(line)
        print(f"    {line:+.1f}".ljust(28) + f"{pct(win)} {pct(push)} {pct(lose)}")
    print("\n  Most likely scores:")
    for home_goals, away_goals, prob in p.top_scores(5):
        print(f"    {home_goals}-{away_goals}  {pct(prob)}")
    if args.form:
        for team in (p.home, p.away):
            form = recent_form(results, team)
            print(f"\n  {team} last {len(form)}: {''.join(form['result'])}")
            for row in form.itertuples():
                print(f"    {row.Date:%d %b %Y}  {row.venue}  {row.score:>5}  {row.opponent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
