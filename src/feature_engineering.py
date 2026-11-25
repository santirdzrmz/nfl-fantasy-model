"""
feature_engineering.py

Builds the modeling table. Core design principles:

1. Every rolling/expanding stat is SHIFTED by one game before being attached to a row,
   so a player's week-N features never contain information from week N itself (no leakage).
2. Opponent defense strength is computed as points-allowed-to-position, also shifted,
   so it reflects only games already played.
3. Position-specific feature sets live in POSITION_FEATURES — QB/RB/WR/TE each get a
   different slice of columns, which is what lets us train architecturally different
   models per position later (different feature space, different hyperparameters).
"""

import numpy as np
import pandas as pd

from injury_features import build_injury_risk_table

TARGET = "fantasy_points_ppr"

ROLL_WINDOWS = [3, 5, 8]

# Columns that get rolling-average treatment per player, by position.
QB_ROLL_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "sacks", "passing_epa", "passing_air_yards", "dakota", "carries",
    "rushing_yards", "rushing_tds", "rushing_epa", "fantasy_points_ppr",
]
RB_ROLL_COLS = [
    "carries", "rushing_yards", "rushing_tds", "rushing_epa", "rushing_first_downs",
    "targets", "receptions", "receiving_yards", "receiving_tds", "target_share",
    "wopr", "fantasy_points_ppr",
]
WR_TE_ROLL_COLS = [
    "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_epa",
    "receiving_air_yards", "receiving_yards_after_catch", "target_share",
    "air_yards_share", "wopr", "racr", "fantasy_points_ppr",
]

POSITION_ROLL_COLS = {"QB": QB_ROLL_COLS, "RB": RB_ROLL_COLS, "WR": WR_TE_ROLL_COLS, "TE": WR_TE_ROLL_COLS}


def _add_player_rolling_features(df: pd.DataFrame, cols: list, windows=ROLL_WINDOWS) -> pd.DataFrame:
    """Shifted rolling means per player, computed in season/week order. No leakage."""
    df = df.sort_values(["player_id", "season", "week"]).copy()
    grp = df.groupby("player_id", group_keys=False)
    for col in cols:
        if col not in df.columns:
            continue
        shifted = grp[col].shift(1)  # exclude current game
        for w in windows:
            df[f"{col}_r{w}"] = (
                shifted.groupby(df["player_id"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            )
        # trend: recent window minus longer window, captures role increasing/decreasing
        if len(windows) >= 2:
            df[f"{col}_trend"] = df[f"{col}_r{windows[0]}"] - df[f"{col}_r{windows[-1]}"]
    return df


def _add_snap_share(df: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    if snaps.empty:
        df["snap_pct"] = np.nan
        df["snap_pct_r3"] = np.nan
        return df
    s = snaps[["season", "week", "player", "team", "offense_pct"]].rename(
        columns={"player": "player_display_name", "offense_pct": "snap_pct"}
    )
    df = df.merge(s, on=["season", "week", "player_display_name"], how="left", suffixes=("", "_snap"))
    df = df.drop_duplicates(subset=["player_id", "season", "week"])
    df = df.sort_values(["player_id", "season", "week"])
    grp = df.groupby("player_id", group_keys=False)
    shifted = grp["snap_pct"].shift(1)
    df["snap_pct_r3"] = shifted.groupby(df["player_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    return df


def _add_opponent_defense_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (opponent_team, position, season, week), the average fantasy points that
    team has ALLOWED to that position in prior weeks this season (expanding, shifted).
    This is the single highest-value 'matchup' feature: a defense's true strength
    against a position is not the same as its strength against the run/pass overall.
    """
    df = df.sort_values(["opponent_team", "position", "season", "week"]).copy()
    grp = df.groupby(["opponent_team", "position", "season"], group_keys=False)
    shifted = grp[TARGET].shift(1)
    df["opp_def_vs_pos_avg"] = (
        shifted.groupby([df["opponent_team"], df["position"], df["season"]])
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
    )
    # fallback to prior-season average vs position when early in a season (thin sample)
    season_pos_avg = (
        df.groupby(["opponent_team", "position", "season"])[TARGET].mean().reset_index()
        .rename(columns={TARGET: "opp_def_vs_pos_season_avg"})
    )
    season_pos_avg["season"] = season_pos_avg["season"] + 1
    df = df.merge(season_pos_avg, on=["opponent_team", "position", "season"], how="left")
    df["opp_def_vs_pos_avg"] = df["opp_def_vs_pos_avg"].fillna(df["opp_def_vs_pos_season_avg"])
    return df


def _add_game_context(df: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Vegas-derived game script signals + rest/home/division context."""
    s = sched[[
        "game_id", "season", "week", "home_team", "away_team", "spread_line",
        "total_line", "away_rest", "home_rest", "div_game", "roof", "surface",
        "temp", "wind",
    ]].copy()

    df = df.merge(s, on=["season", "week"], how="left")
    # keep only rows where the player's team matches home or away in this game row
    df = df[(df["recent_team"] == df["home_team"]) | (df["recent_team"] == df["away_team"])].copy()
    df = df.drop_duplicates(subset=["player_id", "season", "week"])

    df["is_home"] = (df["recent_team"] == df["home_team"]).astype(int)
    # implied team total = total_line/2 +/- spread_line/2, spread_line is home-team-perspective (negative = home favored)
    home_implied = df["total_line"] / 2 - df["spread_line"] / 2
    away_implied = df["total_line"] / 2 + df["spread_line"] / 2
    df["implied_team_total"] = np.where(df["is_home"] == 1, home_implied, away_implied)
    df["team_spread"] = np.where(df["is_home"] == 1, df["spread_line"], -df["spread_line"])
    df["rest_days"] = np.where(df["is_home"] == 1, df["home_rest"], df["away_rest"])
    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    return df


def _add_career_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative games played in the dataset as a career-stage proxy (rookie-year ramp, aging curve)."""
    df = df.sort_values(["player_id", "season", "week"]).copy()
    df["career_game_num"] = df.groupby("player_id").cumcount()
    return df


def build_feature_table(weekly: pd.DataFrame, sched: pd.DataFrame, snaps: pd.DataFrame,
                         injuries: pd.DataFrame = None,
                         positions=("QB", "RB", "WR", "TE")) -> dict:
    """Returns {position: dataframe} each with its own engineered feature columns."""
    base = weekly[weekly["position"].isin(positions)].copy()
    base = base[base["season_type"] == "REG"]  # keep regular season for training signal consistency
    base = _add_career_stage(base)
    base = _add_snap_share(base, snaps)
    base = _add_opponent_defense_strength(base)
    base = _add_game_context(base, sched)

    if injuries is not None:
        risk_table = build_injury_risk_table(weekly, sched, injuries)[["player_id", "season", "injury_risk_score"]]
        base = base.merge(risk_table, on=["player_id", "season"], how="left")
        base["injury_risk_score"] = base["injury_risk_score"].fillna(0.0)
    else:
        base["injury_risk_score"] = 0.0

    tables = {}
    for pos in positions:
        pos_df = base[base["position"] == pos].copy()
        pos_df = _add_player_rolling_features(pos_df, POSITION_ROLL_COLS[pos])
        tables[pos] = pos_df
    return tables


FEATURE_COLUMNS = {
    "QB": [f"{c}_r{w}" for c in QB_ROLL_COLS for w in ROLL_WINDOWS] + [f"{c}_trend" for c in QB_ROLL_COLS],
    "RB": [f"{c}_r{w}" for c in RB_ROLL_COLS for w in ROLL_WINDOWS] + [f"{c}_trend" for c in RB_ROLL_COLS],
    "WR": [f"{c}_r{w}" for c in WR_TE_ROLL_COLS for w in ROLL_WINDOWS] + [f"{c}_trend" for c in WR_TE_ROLL_COLS],
    "TE": [f"{c}_r{w}" for c in WR_TE_ROLL_COLS for w in ROLL_WINDOWS] + [f"{c}_trend" for c in WR_TE_ROLL_COLS],
}
SHARED_FEATURES = [
    "snap_pct_r3", "opp_def_vs_pos_avg", "is_home", "implied_team_total",
    "team_spread", "rest_days", "div_game", "is_dome", "career_game_num", "week",
    "injury_risk_score",
]

for pos in FEATURE_COLUMNS:
    FEATURE_COLUMNS[pos] = FEATURE_COLUMNS[pos] + SHARED_FEATURES


if __name__ == "__main__":
    from data_loader import load_all

    weekly, sched, snaps, injuries = load_all(list(range(2016, 2026)))
    tables = build_feature_table(weekly, sched, snaps, injuries)
    for pos, t in tables.items():
        print(pos, t.shape, "features:", len(FEATURE_COLUMNS[pos]))
