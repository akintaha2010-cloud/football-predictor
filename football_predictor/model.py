"""Dixon-Coles (1997) Poisson model with exponential time decay.

Expected goals:
    home: lambda = exp(home_adv + attack[home] - defence[away])
    away: mu     = exp(attack[away] - defence[home])

Ratings get an L2 penalty `l2` that pulls them toward zero (an average team).
Low scores (0-0, 1-0, 0-1, 1-1) get the Dixon-Coles correction `tau` with
parameter `rho`, and older matches count less: weight = exp(-xi * days_ago).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

# Shrinks team ratings toward average so newly promoted teams with few matches don't get extreme ratings.
DEFAULT_L2 = 1.0

# xi per day; 0.0019 gives a half-life of about a year (Dixon & Coles found 0.0065/week ~ similar).
DEFAULT_XI = 0.0019


def _tau(x, y, lam, mu, rho):
    tau = np.ones_like(lam, dtype=float)
    tau = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return tau


def _tau_gradient(x, y, lam, mu, rho):
    zero = np.zeros_like(lam, dtype=float)
    s00, s01, s10, s11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)
    d_lam = np.where(s00, -mu * rho, np.where(s01, rho, zero))
    d_mu = np.where(s00, -lam * rho, np.where(s10, rho, zero))
    d_rho = np.where(s00, -lam * mu, np.where(s01, lam, np.where(s10, mu, np.where(s11, -1.0, zero))))
    return d_lam, d_mu, d_rho


@dataclass
class MatchPrediction:
    home: str
    away: str
    home_goals: float
    away_goals: float
    score_matrix: np.ndarray

    @property
    def home_win(self) -> float:
        return float(np.tril(self.score_matrix, -1).sum())

    @property
    def draw(self) -> float:
        return float(np.trace(self.score_matrix))

    @property
    def away_win(self) -> float:
        return float(np.triu(self.score_matrix, 1).sum())

    def over(self, line: float = 2.5) -> float:
        goals = np.add.outer(np.arange(self.score_matrix.shape[0]), np.arange(self.score_matrix.shape[1]))
        return float(self.score_matrix[goals > line].sum())

    @property
    def double_chance(self) -> dict[str, float]:
        """1X (home or draw), 12 (either team wins) and X2 (draw or away)."""
        return {
            "1X": self.home_win + self.draw,
            "12": self.home_win + self.away_win,
            "X2": self.draw + self.away_win,
        }

    @property
    def draw_no_bet(self) -> tuple[float, float]:
        """Home and away win probabilities when a draw refunds the stake."""
        decided = self.home_win + self.away_win
        return self.home_win / decided, self.away_win / decided

    def handicap(self, line: float) -> tuple[float, float, float]:
        """Home win / push / home loss probabilities when `line` goals are added to the home score.

        Whole lines (e.g. -1) can push and refund the stake; half lines (e.g. -0.5) cannot.
        """
        margin = np.subtract.outer(np.arange(self.score_matrix.shape[0]), np.arange(self.score_matrix.shape[1])) + line
        return (
            float(self.score_matrix[margin > 0].sum()),
            float(self.score_matrix[margin == 0].sum()),
            float(self.score_matrix[margin < 0].sum()),
        )

    @property
    def both_teams_score(self) -> float:
        return float(self.score_matrix[1:, 1:].sum())

    def to_dict(self, top_n: int = 5) -> dict:
        return {
            "home": self.home,
            "away": self.away,
            "expected_goals": {"home": self.home_goals, "away": self.away_goals},
            "result": {"home_win": self.home_win, "draw": self.draw, "away_win": self.away_win},
            "over": {str(line): self.over(line) for line in (1.5, 2.5, 3.5)},
            "double_chance": self.double_chance,
            "draw_no_bet": dict(zip(("home", "away"), self.draw_no_bet)),
            "both_teams_score": self.both_teams_score,
            "handicap": {
                f"{line:+.1f}": dict(zip(("win", "push", "lose"), self.handicap(line)))
                for line in (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
            },
            "top_scores": [{"score": f"{h}-{a}", "probability": prob} for h, a, prob in self.top_scores(top_n)],
        }

    def top_scores(self, n: int = 5) -> list[tuple[int, int, float]]:
        flat = np.argsort(self.score_matrix, axis=None)[::-1][:n]
        rows, cols = np.unravel_index(flat, self.score_matrix.shape)
        return [(int(h), int(a), float(self.score_matrix[h, a])) for h, a in zip(rows, cols)]


class DixonColes:
    def __init__(self, xi: float = DEFAULT_XI, l2: float = DEFAULT_L2, max_goals: int = 10):
        self.xi = xi
        self.l2 = l2
        self.max_goals = max_goals
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_adv = 0.0
        self.rho = 0.0

    def fit(self, results: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "DixonColes":
        """Fit on columns Date, HomeTeam, AwayTeam, FTHG, FTAG. Matches after `as_of` are ignored."""
        as_of = pd.Timestamp(as_of) if as_of is not None else results["Date"].max()
        df = results[results["Date"] <= as_of]
        if df.empty:
            raise ValueError("No matches to fit on")

        self.teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        index = {team: i for i, team in enumerate(self.teams)}
        n = len(self.teams)
        home = df["HomeTeam"].map(index).to_numpy()
        away = df["AwayTeam"].map(index).to_numpy()
        hg = df["FTHG"].to_numpy()
        ag = df["FTAG"].to_numpy()
        days_ago = (as_of - df["Date"]).dt.days.to_numpy()
        weights = np.exp(-self.xi * days_ago)
        log_fact = -gammaln(hg + 1) - gammaln(ag + 1)

        def unpack(params):
            attack = np.append(params[: n - 1], -params[: n - 1].sum())  # sum(attack) = 0
            defence = params[n - 1 : 2 * n - 1]
            return attack, defence, params[-2], params[-1]

        def neg_log_likelihood(params):
            attack, defence, home_adv, rho = unpack(params)
            lam = np.exp(home_adv + attack[home] - defence[away])
            mu = np.exp(attack[away] - defence[home])
            tau = np.clip(_tau(hg, ag, lam, mu, rho), 1e-10, None)
            ll = np.log(tau) + hg * np.log(lam) - lam + ag * np.log(mu) - mu + log_fact
            penalty = self.l2 * (np.square(attack).sum() + np.square(defence).sum())

            # Analytic gradient: d tau / d lam, d mu, d rho for the four corrected scores.
            dtau_lam, dtau_mu, dtau_rho = _tau_gradient(hg, ag, lam, mu, rho)
            g_lam = weights * (hg - lam + lam * dtau_lam / tau)  # d ll / d log(lam)
            g_mu = weights * (ag - mu + mu * dtau_mu / tau)  # d ll / d log(mu)
            g_attack = np.bincount(home, g_lam, n) + np.bincount(away, g_mu, n) - 2 * self.l2 * attack
            g_defence = -np.bincount(away, g_lam, n) - np.bincount(home, g_mu, n) - 2 * self.l2 * defence
            grad = np.concatenate([
                g_attack[:-1] - g_attack[-1],
                g_defence,
                [g_lam.sum(), (weights * dtau_rho / tau).sum()],
            ])
            return -(weights * ll).sum() + penalty, -grad

        x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25, -0.05]])
        bounds = [(None, None)] * (2 * n - 1) + [(None, None), (-0.3, 0.3)]
        result = minimize(neg_log_likelihood, x0, jac=True, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            raise RuntimeError(f"Model fit failed: {result.message}")

        attack, defence, self.home_adv, self.rho = unpack(result.x)
        self.attack = dict(zip(self.teams, attack))
        self.defence = dict(zip(self.teams, defence))
        return self

    def resolve_team(self, name: str) -> str:
        if name in self.attack:
            return name
        lowered = {team.lower(): team for team in self.teams}
        if name.lower() in lowered:
            return lowered[name.lower()]
        close = difflib.get_close_matches(name, self.teams, n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise KeyError(f"Unknown team {name!r}.{hint}")

    def predict(self, home: str, away: str, neutral: bool = False) -> MatchPrediction:
        """`neutral=True` drops home advantage, e.g. for cup finals at a neutral stadium."""
        home, away = self.resolve_team(home), self.resolve_team(away)
        home_adv = 0.0 if neutral else self.home_adv
        lam = np.exp(home_adv + self.attack[home] - self.defence[away])
        mu = np.exp(self.attack[away] - self.defence[home])
        goals = np.arange(self.max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        x, y = np.meshgrid(goals, goals, indexing="ij")
        matrix *= _tau(x, y, np.full_like(matrix, lam), np.full_like(matrix, mu), self.rho)
        matrix /= matrix.sum()
        return MatchPrediction(home, away, float(lam), float(mu), matrix)

    def ratings(self) -> pd.DataFrame:
        """Teams ranked by expected goal difference against an average side on neutral ground."""
        df = pd.DataFrame({"attack": self.attack, "defence": self.defence})
        df["goals_for"] = np.exp(df["attack"])
        df["goals_against"] = np.exp(-df["defence"])
        df["strength"] = df["goals_for"] - df["goals_against"]
        return df.sort_values("strength", ascending=False)
