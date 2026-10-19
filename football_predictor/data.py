"""Download and load match results from football-data.co.uk."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_AGE_HOURS = 12

LEAGUES = {
    "T1": "Turkey Super Lig",
    "E0": "England Premier League",
    "E1": "England Championship",
    "SP1": "Spain La Liga",
    "I1": "Italy Serie A",
    "D1": "Germany Bundesliga",
    "F1": "France Ligue 1",
    "N1": "Netherlands Eredivisie",
    "P1": "Portugal Primeira Liga",
    "E2": "England League One",
    "SC0": "Scotland Premiership",
    "SC1": "Scotland Championship",
    "B1": "Belgium Pro League",
    "G1": "Greece Super League",
    "D2": "Germany 2. Bundesliga",
    "I2": "Italy Serie B",
    "SP2": "Spain Segunda Division",
    "F2": "France Ligue 2",
}

COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
# Market average closing odds, kept when present so predictions can be compared to bookmakers.
ODDS_COLUMNS = ["AvgH", "AvgD", "AvgA"]


def season_code(start_year: int) -> str:
    """2025 -> "2526" (the 2025/26 season)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def recent_seasons(count: int, today: date | None = None) -> list[str]:
    start = current_season_start(today)
    return [season_code(year) for year in range(start - count + 1, start + 1)]


def _fetch(url: str, path: Path) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": "football-predictor"})
    with urllib.request.urlopen(request, timeout=30) as response:
        path.write_bytes(response.read())
    return path


def download(league: str, season: str, data_dir: Path = DATA_DIR, force: bool = False,
             max_age_hours: float | None = None) -> Path:
    """Cached CSV path. Re-downloads when `force`, or when the file is older than `max_age_hours`.

    If a refresh fails (e.g. offline) but an older copy exists, the old copy is used with a warning.
    """
    if league not in LEAGUES:
        raise ValueError(f"Unknown league {league!r}. Choose from: {', '.join(LEAGUES)}")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{league}_{season}.csv"
    stale = max_age_hours is not None and path.exists() and time.time() - path.stat().st_mtime > max_age_hours * 3600
    if path.exists() and not (force or stale):
        return path
    try:
        return _fetch(BASE_URL.format(season=season, league=league), path)
    except (urllib.error.URLError, TimeoutError) as error:
        if not path.exists():
            raise
        warnings.warn(f"Could not refresh {path.name} ({error}); using the cached copy")
        return path


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in COLUMNS + ODDS_COLUMNS)
    df = df.dropna(subset=COLUMNS)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()
    for col in ODDS_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[COLUMNS + ODDS_COLUMNS]


def load_results(league: str, seasons: list[str], data_dir: Path = DATA_DIR, refresh: bool = False) -> pd.DataFrame:
    """Results for the given seasons, oldest first.

    Finished seasons are downloaded once. The current season is re-downloaded when its copy is more than
    MAX_AGE_HOURS old, or always when refresh=True.
    """
    current = season_code(current_season_start())
    frames = [
        read_results(download(league, s, data_dir, force=refresh and s == current,
                              max_age_hours=MAX_AGE_HOURS if s == current else None))
        for s in seasons
    ]
    return pd.concat(frames, ignore_index=True).sort_values("Date", ignore_index=True)


def load_fixtures(league: str, data_dir: Path = DATA_DIR, path: Path | None = None) -> pd.DataFrame:
    """Upcoming matches with market average odds (falls back to Bet365 when the average is missing)."""
    if path is None:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = _fetch(FIXTURES_URL, data_dir / "fixtures.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["Div"] == league].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()
    for outcome in "HDA":
        avg = df.get(f"Avg{outcome}", pd.Series(float("nan"), index=df.index))
        df[f"Odds{outcome}"] = avg.fillna(df.get(f"B365{outcome}", float("nan")))
    return df[["Date", "HomeTeam", "AwayTeam", "OddsH", "OddsD", "OddsA"]].reset_index(drop=True)
