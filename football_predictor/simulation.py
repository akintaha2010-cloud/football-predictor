"""Monte Carlo simulation of the rest of a league season."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import DixonColes


def remaining_fixtures(played: pd.DataFrame) -> list[tuple[str, str]]:
    """Double round-robin fixtures (each pair home and away) that haven't been played yet."""
    teams = sorted(set(played["HomeTeam"]) | set(played["AwayTeam"]))
    done = set(zip(played["HomeTeam"], played["AwayTeam"]))
    return [(h, a) for h in teams for a in teams if h != a and (h, a) not in done]


def simulate_season(model: DixonColes, played: pd.DataFrame, n_sims: int = 10_000, seed: int | None = None):
    """Simulate the remaining fixtures `n_sims` times.

    Returns (summary, positions): summary has current and expected points per team, positions[i, j] is
    the probability that team i finishes in position j + 1. Ties are broken on goal difference, then
    goals scored, then randomly (leagues that use head-to-head first are approximated).
    """
    teams = sorted(set(played["HomeTeam"]) | set(played["AwayTeam"]))
    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    base = np.zeros((3, n_teams))  # points, goals for, goals against
    for match in played.itertuples():
        h, a = index[match.HomeTeam], index[match.AwayTeam]
        base[1, h] += match.FTHG
        base[2, h] += match.FTAG
        base[1, a] += match.FTAG
        base[2, a] += match.FTHG
        if match.FTHG > match.FTAG:
            base[0, h] += 3
        elif match.FTHG < match.FTAG:
            base[0, a] += 3
        else:
            base[0, h] += 1
            base[0, a] += 1

    rng = np.random.default_rng(seed)
    points = np.tile(base[0], (n_sims, 1))
    goals_for = np.tile(base[1], (n_sims, 1))
    goals_against = np.tile(base[2], (n_sims, 1))

    for home, away in remaining_fixtures(played):
        matrix = model.predict(home, away).score_matrix
        cdf = matrix.ravel().cumsum()
        cells = np.minimum(np.searchsorted(cdf, rng.random(n_sims) * cdf[-1]), cdf.size - 1)
        hg, ag = np.divmod(cells, matrix.shape[1])
        h, a = index[home], index[away]
        points[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        points[:, a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        goals_for[:, h] += hg
        goals_against[:, h] += ag
        goals_for[:, a] += ag
        goals_against[:, a] += hg

    goal_diff = goals_for - goals_against
    order = np.lexsort((rng.random(points.shape), -goals_for, -goal_diff, -points), axis=1)
    finish = np.empty_like(order)
    finish[np.arange(n_sims)[:, None], order] = np.arange(n_teams)

    positions = np.zeros((n_teams, n_teams))
    for pos in range(n_teams):
        positions[:, pos] = (finish == pos).mean(axis=0)

    summary = pd.DataFrame({
        "played": [((played["HomeTeam"] == t) | (played["AwayTeam"] == t)).sum() for t in teams],
        "points": base[0].astype(int),
        "expected_points": points.mean(axis=0),
    }, index=teams)
    return summary, positions
