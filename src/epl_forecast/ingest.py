"""Download EPL results from football-data.co.uk and normalise them.

Source: https://www.football-data.co.uk/englandm.php (free CSV per season, file E0.csv).
Season code is two 2-digit years, e.g. 2627 = 2026-27.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "B365H", "B365D", "B365A"]


def fetch_season(season: str, raw_dir: Path, timeout: int = 30) -> pd.DataFrame:
    """Download one season, cache the raw CSV, return normalised rows."""
    resp = requests.get(BASE_URL.format(season=season), timeout=timeout)
    resp.raise_for_status()
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"E0_{season}.csv").write_bytes(resp.content)
    return normalise(pd.read_csv(io.BytesIO(resp.content), encoding="latin-1"), season)


def normalise(df: pd.DataFrame, season: str) -> pd.DataFrame:
    cols = [c for c in KEEP if c in df.columns]
    out = df[cols].dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out["season"] = season
    out = out.rename(columns={"HomeTeam": "home", "AwayTeam": "away",
                              "FTHG": "hg", "FTAG": "ag", "Date": "date"})
    # Rows with no score yet (future fixtures occasionally appear) are dropped from results.
    out = out.dropna(subset=["hg", "ag", "date"])
    out[["hg", "ag"]] = out[["hg", "ag"]].astype(int)
    return out.reset_index(drop=True)


def load_history(seasons: list[str], data_dir: Path) -> pd.DataFrame:
    frames = [fetch_season(s, data_dir / "raw") for s in seasons]
    df = pd.concat(frames, ignore_index=True).sort_values("date")
    df.to_csv(data_dir / "matches.csv", index=False)
    return df
