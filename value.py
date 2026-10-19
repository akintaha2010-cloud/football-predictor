"""Compare model probabilities with bookmaker odds for upcoming matches.

    python value.py --league E0
    python value.py --league T1 --min-edge 0.10

A bet has positive expected value when model_probability * odds > 1. The backtest shows the model is
somewhat less accurate than closing odds, so most "value" it finds is model error, not free money.
"""

from __future__ import annotations

import argparse

from football_predictor.data import LEAGUES, load_fixtures, load_results, recent_seasons
from football_predictor.model import DEFAULT_L2, DEFAULT_XI, DixonColes


def edge(probability: float, odds: float) -> float:
    """Expected profit per unit staked."""
    return probability * odds - 1


def kelly_fraction(probability: float, odds: float) -> float:
    """Share of bankroll the Kelly criterion would stake (0 when there is no edge)."""
    if odds <= 1:
        return 0.0
    return max(0.0, edge(probability, odds) / (odds - 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", default="T1", choices=LEAGUES)
    parser.add_argument("--seasons", type=int, default=3)
    parser.add_argument("--min-edge", type=float, default=0.05, help="Only flag bets with at least this edge")
    parser.add_argument("--refresh", action="store_true", help="Re-download results")
    parser.add_argument("--xi", type=float, default=DEFAULT_XI)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    args = parser.parse_args(argv)

    results = load_results(args.league, recent_seasons(args.seasons), refresh=args.refresh)
    model = DixonColes(xi=args.xi, l2=args.l2).fit(results)
    fixtures = load_fixtures(args.league)
    if fixtures.empty:
        print(f"No upcoming {LEAGUES[args.league]} fixtures with odds right now.")
        return 0

    print(f"{LEAGUES[args.league]}: {len(fixtures)} upcoming matches (model vs average odds)\n")
    flagged = 0
    for match in fixtures.itertuples():
        try:
            p = model.predict(match.HomeTeam, match.AwayTeam)
        except KeyError:
            print(f"{match.Date:%d %b}  {match.HomeTeam} vs {match.AwayTeam}: unknown team, skipped")
            continue
        print(f"{match.Date:%d %b}  {p.home} vs {p.away}")
        for label, prob, odds in [("Home", p.home_win, match.OddsH), ("Draw", p.draw, match.OddsD), ("Away", p.away_win, match.OddsA)]:
            e = edge(prob, odds)
            mark = ""
            if e >= args.min_edge:
                flagged += 1
                mark = f"  <- value, Kelly {kelly_fraction(prob, odds):.1%}"
            print(f"    {label}: model {prob:6.1%}  fair odds {1 / prob:5.2f}  market {odds:5.2f}  edge {e:+6.1%}{mark}")
    print(f"\n{flagged} outcome(s) with edge >= {args.min_edge:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
