"""Recent results of a team and past meetings between two teams."""

from __future__ import annotations

import pandas as pd


def recent_form(results: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    """Last `n` matches of `team`, newest first, with its result (W/D/L) and goals."""
    played = results[(results["HomeTeam"] == team) | (results["AwayTeam"] == team)]
    played = played.sort_values("Date", ascending=False).head(n)
    home = played["HomeTeam"] == team
    goals_for = played["FTHG"].where(home, played["FTAG"])
    goals_against = played["FTAG"].where(home, played["FTHG"])
    return pd.DataFrame({
        "Date": played["Date"],
        "opponent": played["AwayTeam"].where(home, played["HomeTeam"]),
        "venue": home.map({True: "H", False: "A"}),
        "score": goals_for.astype(str) + "-" + goals_against.astype(str),
        "result": (goals_for > goals_against).map({True: "W", False: ""})
        + (goals_for == goals_against).map({True: "D", False: ""})
        + (goals_for < goals_against).map({True: "L", False: ""}),
    }).reset_index(drop=True)


def head_to_head(results: pd.DataFrame, team: str, other: str, n: int = 5) -> pd.DataFrame:
    """Last `n` meetings of the two teams, newest first, at either ground."""
    met = results[((results["HomeTeam"] == team) & (results["AwayTeam"] == other))
                  | ((results["HomeTeam"] == other) & (results["AwayTeam"] == team))]
    met = met.sort_values("Date", ascending=False).head(n)
    return pd.DataFrame({
        "Date": met["Date"],
        "HomeTeam": met["HomeTeam"],
        "AwayTeam": met["AwayTeam"],
        "score": met["FTHG"].astype(str) + "-" + met["FTAG"].astype(str),
    }).reset_index(drop=True)
