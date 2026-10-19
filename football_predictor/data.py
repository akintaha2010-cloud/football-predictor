"""Download and load match results from football-data.co.uk."""

from __future__ import annotations

import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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
}

COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def season_code(start_year: int) -> str:
    """2025 -> "2526" (the 2025/26 season)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def recent_seasons(count: int, today: date | None = None) -> list[str]:
    start = current_season_start(today)
    return [season_code(year) for year in range(start - count + 1, start + 1)]


def download(league: str, season: str, data_dir: Path = DATA_DIR, force: bool = False) -> Path:
    if league not in LEAGUES:
        raise ValueError(f"Unknown league {league!r}. Choose from: {', '.join(LEAGUES)}")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{league}_{season}.csv"
    if path.exists() and not force:
        return path
    url = BASE_URL.format(season=season, league=league)
    request = urllib.request.Request(url, headers={"User-Agent": "football-predictor"})
    with urllib.request.urlopen(request, timeout=30) as response:
        path.write_bytes(response.read())
    return path


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in COLUMNS)
    df = df.dropna(subset=COLUMNS)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()
    return df[COLUMNS]


def load_results(league: str, seasons: list[str], data_dir: Path = DATA_DIR, refresh: bool = False) -> pd.DataFrame:
    """Results for the given seasons, oldest first. The current season is re-downloaded when refresh=True."""
    frames = [read_results(download(league, s, data_dir, force=refresh)) for s in seasons]
    return pd.concat(frames, ignore_index=True).sort_values("Date", ignore_index=True)
