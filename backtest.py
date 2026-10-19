"""Walk-forward backtest: refit before each matchweek, predict it, score the predictions.

    python backtest.py --league T1 --season 2526
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from football_predictor.data import LEAGUES, load_results, season_code
from football_predictor.model import DEFAULT_L2, DEFAULT_XI, DixonColes

OUTCOMES = ["H", "D", "A"]


def run(results: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp, xi: float, l2: float, step_days: int = 7):
    rows = []
    cursor = test_start
    while cursor <= test_end:
        window_end = cursor + pd.Timedelta(days=step_days)
        upcoming = results[(results["Date"] >= cursor) & (results["Date"] < window_end)]
        if not upcoming.empty:
            model = DixonColes(xi=xi, l2=l2).fit(results, as_of=cursor - pd.Timedelta(days=1))
            for match in upcoming.itertuples():
                if match.HomeTeam not in model.attack or match.AwayTeam not in model.attack:
                    continue
                p = model.predict(match.HomeTeam, match.AwayTeam)
                actual = "H" if match.FTHG > match.FTAG else "A" if match.FTHG < match.FTAG else "D"
                odds = np.array([match.AvgH, match.AvgD, match.AvgA], dtype=float)
                market = (1 / odds) / (1 / odds).sum() if np.all(odds > 1) else np.full(3, np.nan)
                rows.append({
                    "Date": match.Date, "HomeTeam": match.HomeTeam, "AwayTeam": match.AwayTeam,
                    "actual": actual,
                    "pH": p.home_win, "pD": p.draw, "pA": p.away_win,
                    "mH": market[0], "mD": market[1], "mA": market[2],
                })
        cursor = window_end
    return pd.DataFrame(rows)


def score(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    probs = df[[f"{prefix}{o}" for o in OUTCOMES]].to_numpy()
    actual = np.array([[a == o for o in OUTCOMES] for a in df["actual"]], dtype=float)
    return {
        "log_loss": float(-np.log((probs * actual).sum(axis=1)).mean()),
        "brier": float(np.square(probs - actual).sum(axis=1).mean()),
        "accuracy": float((probs.argmax(axis=1) == actual.argmax(axis=1)).mean()),
    }


def calibration(df: pd.DataFrame, prefix: str, bins: int = 5) -> pd.DataFrame:
    """Groups every predicted outcome probability into bins and compares it with how often it happened."""
    probs = df[[f"{prefix}{o}" for o in OUTCOMES]].to_numpy().ravel()
    hits = np.array([[a == o for o in OUTCOMES] for a in df["actual"]], dtype=float).ravel()
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if mask.any():
            rows.append({
                "bin": f"{edges[b]:.0%}-{edges[b + 1]:.0%}",
                "count": int(mask.sum()),
                "predicted": float(probs[mask].mean()),
                "observed": float(hits[mask].mean()),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", default="T1", choices=LEAGUES)
    parser.add_argument("--season", default="2526", help='Season to test, e.g. "2526" for 2025/26')
    parser.add_argument("--history", type=int, default=2, help="Earlier seasons used for training")
    parser.add_argument("--xi", type=float, default=DEFAULT_XI)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    args = parser.parse_args()

    start_year = 2000 + int(args.season[:2])
    seasons = [season_code(y) for y in range(start_year - args.history, start_year + 1)]
    results = load_results(args.league, seasons)
    test = load_results(args.league, [args.season])
    # Skip the first month of the season: the model needs a few rounds to rate promoted teams.
    test_start = test["Date"].min() + pd.Timedelta(days=28)

    df = run(results, test_start, test["Date"].max(), args.xi, args.l2)
    print(f"{LEAGUES[args.league]} {args.season}: {len(df)} matches (xi={args.xi}, l2={args.l2})")
    print(f"  model:     {score(df, 'p')}")
    with_odds = df.dropna(subset=["mH"])
    if not with_odds.empty:
        print(f"  model*:    {score(with_odds, 'p')}  (matches with odds: {len(with_odds)})")
        print(f"  bookmaker: {score(with_odds, 'm')}")
    base = df.assign(bH=(df["actual"] == "H").mean(), bD=(df["actual"] == "D").mean(), bA=(df["actual"] == "A").mean())
    print(f"  base rate: {score(base, 'b')}")

    print("\nCalibration (all home/draw/away probabilities):")
    print(f"  {'Predicted':<10} {'Count':>6} {'Avg predicted':>14} {'Happened':>9}")
    for row in calibration(df, "p").itertuples():
        print(f"  {row.bin:<10} {row.count:6d} {row.predicted:14.1%} {row.observed:9.1%}")


if __name__ == "__main__":
    main()
