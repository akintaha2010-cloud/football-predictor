"""Simulate the rest of the current season: title, top-N and relegation chances.

    python simulate.py --league T1
    python simulate.py --league E0 --top 4 --relegated 3 --sims 20000
    python simulate.py --league T1 --plot season.png
"""

from __future__ import annotations

import argparse

from football_predictor.data import LEAGUES, load_results, recent_seasons
from football_predictor.model import DEFAULT_L2, DEFAULT_XI, DixonColes
from football_predictor.simulation import remaining_fixtures, simulate_season


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", default="T1", choices=LEAGUES)
    parser.add_argument("--seasons", type=int, default=3, help="Seasons to train the model on")
    parser.add_argument("--sims", type=int, default=10_000, help="Number of simulated seasons")
    parser.add_argument("--top", type=int, default=4, help="Places that count as 'top' (e.g. European spots)")
    parser.add_argument("--relegated", type=int, default=3, help="Number of relegation places")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot", metavar="FILE", help="Save a heatmap of finishing positions (needs matplotlib)")
    parser.add_argument("--refresh", action="store_true", help="Re-download the data files")
    parser.add_argument("--xi", type=float, default=DEFAULT_XI)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    args = parser.parse_args(argv)

    seasons = recent_seasons(args.seasons)
    results = load_results(args.league, seasons, refresh=args.refresh)
    current = load_results(args.league, seasons[-1:])
    model = DixonColes(xi=args.xi, l2=args.l2).fit(results)

    summary, positions = simulate_season(model, current, n_sims=args.sims, seed=args.seed)
    n_teams = len(summary)
    summary["title"] = positions[:, 0]
    summary["top"] = positions[:, : args.top].sum(axis=1)
    summary["relegated"] = positions[:, n_teams - args.relegated :].sum(axis=1)
    table = summary.sort_values("expected_points", ascending=False)

    remaining = len(remaining_fixtures(current))
    print(f"{LEAGUES[args.league]}: {len(current)} matches played, {remaining} left, {args.sims} simulations\n")
    print(f"{'Team':<20} {'P':>3} {'Pts':>4} {'xPts':>6} {'Title':>7} {'Top ' + str(args.top):>7} {'Releg.':>7}")
    for team, row in table.iterrows():
        print(f"{team:<20} {int(row.played):3d} {int(row.points):4d} {row.expected_points:6.1f} "
              f"{row.title:7.1%} {row.top:7.1%} {row.relegated:7.1%}")

    if args.plot:
        from football_predictor.plot import plot_positions

        title = f"{LEAGUES[args.league]}: chance of each final position (%)"
        print(f"\nSaved {plot_positions(summary, positions, args.plot, title)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
