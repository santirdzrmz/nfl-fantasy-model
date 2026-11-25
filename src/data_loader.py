"""
data_loader.py
Pulls and caches raw NFL data needed for the fantasy model:
  - weekly player stats (nfl_data_py, sourced from nflverse-data)
  - game schedules + Vegas lines (nflverse/nfldata games.csv)
  - snap counts (nflverse-data release)

All data is cached to /data as parquet so repeated runs don't re-hit the network.
"""

import os
import warnings
import pandas as pd
import nfl_data_py as nfl

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# nflverse restructured its player-stats release ahead of the 2025 season: the legacy
# `player_stats_{year}.parquet` files that nfl_data_py 0.3.3 reads from are frozen and
# don't extend past 2024. Current seasons live under the new `stats_player_week_{year}`
# release with some columns renamed. We fetch those directly and rename back to the
# legacy schema so the rest of the pipeline doesn't need to care which source a season
# came from.
NEW_SCHEMA_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{0}.parquet"
NEW_SCHEMA_RENAME = {
    "team": "recent_team",
    "passing_interceptions": "interceptions",
    "sacks_suffered": "sacks",
    "sack_yards_lost": "sack_yards",
}
LEGACY_LAST_SEASON = 2024


def _cache_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.parquet")


def _load_new_schema_years(years) -> pd.DataFrame:
    frames = []
    for y in years:
        d = pd.read_parquet(NEW_SCHEMA_URL.format(y))
        d = d.rename(columns=NEW_SCHEMA_RENAME)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # legacy schema has 'dakota' (QB rushing+passing composite); not present in the new
    # release yet, so add as NaN and let the model treat it as missing (LightGBM-native)
    if "dakota" not in df.columns:
        df["dakota"] = float("nan")
    return df


def load_weekly_stats(years, refresh: bool = False) -> pd.DataFrame:
    """Player-week fantasy-relevant stats: targets, snaps-adjacent efficiency, EPA, etc."""
    cache = _cache_path(f"weekly_{min(years)}_{max(years)}")
    if not refresh and os.path.exists(cache):
        return pd.read_parquet(cache)

    legacy_years = [y for y in years if y <= LEGACY_LAST_SEASON]
    new_years = [y for y in years if y > LEGACY_LAST_SEASON]

    frames = []
    if legacy_years:
        frames.append(nfl.import_weekly_data(legacy_years))
    if new_years:
        frames.append(_load_new_schema_years(new_years))

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df.to_parquet(cache)
    return df


def load_schedules(years, refresh: bool = False) -> pd.DataFrame:
    """Game-level schedule with Vegas spread/total, rest days, weather, roof/surface."""
    cache = _cache_path(f"schedules_{min(years)}_{max(years)}")
    if not refresh and os.path.exists(cache):
        return pd.read_parquet(cache)

    sched = pd.read_csv(GAMES_URL)
    sched = sched[sched["season"].isin(years)].copy()
    sched.to_parquet(cache)
    return sched


def load_snap_counts(years, refresh: bool = False) -> pd.DataFrame:
    """Player-week snap counts and snap share (offense_pct)."""
    cache = _cache_path(f"snaps_{min(years)}_{max(years)}")
    if not refresh and os.path.exists(cache):
        return pd.read_parquet(cache)

    years = [y for y in years if y >= 2012]  # snap counts start 2012
    if not years:
        return pd.DataFrame()

    df = nfl.import_snap_counts(years)
    df.to_parquet(cache)
    return df


def load_injuries(years, refresh: bool = False) -> pd.DataFrame:
    """Weekly injury report designations (Questionable/Doubtful/Out) with body-part text."""
    cache = _cache_path(f"injuries_{min(years)}_{max(years)}")
    if not refresh and os.path.exists(cache):
        return pd.read_parquet(cache)

    df = nfl.import_injuries(list(years))
    df.to_parquet(cache)
    return df


def load_all(years, refresh: bool = False):
    weekly = load_weekly_stats(years, refresh)
    sched = load_schedules(years, refresh)
    snaps = load_snap_counts(years, refresh)
    injuries = load_injuries(years, refresh)
    return weekly, sched, snaps, injuries


if __name__ == "__main__":
    YEARS = list(range(2016, 2026))
    weekly, sched, snaps, injuries = load_all(YEARS)
    print("weekly:", weekly.shape)
    print("schedules:", sched.shape)
    print("snaps:", snaps.shape)
    print("injuries:", injuries.shape)
