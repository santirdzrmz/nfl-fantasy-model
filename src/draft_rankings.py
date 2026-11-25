"""
draft_rankings.py

Turns the weekly models into a single, ranked draft board using Value Over
Replacement (VOR) -- the standard way fantasy rankings compare players across
different positions, since "points scored" alone isn't comparable (a QB scores on
a different scale than a TE, and startable-player scarcity differs by position).

INJURY PENALTY (the piece added on top of the base model)
-----------------------------------------------------------
The weekly model already sees `injury_risk_score` as a feature (see
feature_engineering.py / injury_features.py), which lets it learn things like
"teams often manage snap counts down for players with recent injury history." But
that feature only ever sees rows where the player DID suit up and play -- the
weekly model was never trained on "will this player be active at all this week,"
so it has no way to discount for games missed entirely. The draft-ranking layer
adds that explicitly, as two separate, auditable penalties:

  1. AVAILABILITY penalty (quantity): discounts expected games played, driven
     mostly by how often the player has actually missed games recently.
  2. SEVERITY penalty (quality): a smaller discount to points-per-game-played,
     representing that players returning from more severe injury types (ACL,
     Achilles, etc.) often return somewhat diminished even in games they do
     play -- distinct from just missing time.

Both are driven by injury_risk_score's components (trailing_missed_rate for #1,
trailing_severity for #2) computed in injury_features.py, and are capped so no
player is discounted to zero -- even oft-injured stars usually play meaningful
snaps in most seasons.
"""

import os
import pandas as pd
import numpy as np

from predict import PredictionContext
from injury_features import build_injury_risk_table
from data_loader import load_all

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Standard 12-team league replacement ranks (last "startable" player at each
# position). Tune these for your league size / roster requirements.
REPLACEMENT_RANK = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}

MAX_AVAILABILITY_DISCOUNT = 0.50   # highest-risk player still assumed to play >= 50% of games
MAX_SEVERITY_DISCOUNT = 0.15       # up to 15% lower points-per-game for severe injury history
MIN_GAMES_FLOOR = 2                # never project fewer than this many expected games


def _apply_injury_penalty(season_df: pd.DataFrame, risk_df: pd.DataFrame, season: int) -> pd.DataFrame:
    r = risk_df[risk_df["season"] == season][
        ["player_id", "injury_risk_score", "trailing_missed_rate", "trailing_severity", "chronic_seasons"]
    ]
    df = season_df.merge(r, on="player_id", how="left")
    for c in ["injury_risk_score", "trailing_missed_rate", "trailing_severity", "chronic_seasons"]:
        df[c] = df[c].fillna(0.0)

    df["availability_discount"] = (df["trailing_missed_rate"] * MAX_AVAILABILITY_DISCOUNT / 0.8).clip(0, MAX_AVAILABILITY_DISCOUNT)
    df["severity_discount"] = (df["trailing_severity"] * MAX_SEVERITY_DISCOUNT).clip(0, MAX_SEVERITY_DISCOUNT)

    df["expected_games_played"] = np.maximum(
        df["games_scheduled"] * (1 - df["availability_discount"]), MIN_GAMES_FLOOR
    )
    games_factor = df["expected_games_played"] / df["games_scheduled"]
    performance_factor = 1 - df["severity_discount"]

    df["season_points_adjusted"] = df["season_points_healthy"] * games_factor * performance_factor
    df["season_floor_p10_adjusted"] = df["season_floor_p10"] * games_factor * performance_factor
    df["season_ceiling_p90_adjusted"] = df["season_ceiling_p90"] * games_factor * performance_factor
    df["injury_points_lost"] = df["season_points_healthy"] - df["season_points_adjusted"]
    return df


def _add_vor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["position_rank"] = df.groupby("position")["season_points_adjusted"].rank(ascending=False, method="first").astype(int)

    replacement_points = {}
    for pos, rank in REPLACEMENT_RANK.items():
        pos_df = df[df["position"] == pos].sort_values("season_points_adjusted", ascending=False)
        replacement_points[pos] = (
            pos_df.iloc[rank - 1]["season_points_adjusted"] if len(pos_df) >= rank else pos_df["season_points_adjusted"].min()
        )
    df["replacement_level"] = df["position"].map(replacement_points)
    df["vor"] = df["season_points_adjusted"] - df["replacement_level"]
    df = df.sort_values("vor", ascending=False).reset_index(drop=True)
    df["overall_rank"] = df.index + 1
    return df


def build_draft_board(season: int, years_history=range(2016, 2026)) -> pd.DataFrame:
    ctx = PredictionContext(season, years_history)
    season_proj = ctx.project_season()

    weekly, sched, snaps, injuries = load_all(list(years_history))
    risk_table = build_injury_risk_table(weekly, sched, injuries)

    penalized = _apply_injury_penalty(season_proj, risk_table, season)
    board = _add_vor(penalized)

    cols = [
        "overall_rank", "player_display_name", "position", "position_rank", "recent_team",
        "vor", "season_points_adjusted", "season_points_healthy", "injury_points_lost",
        "games_scheduled", "expected_games_played",
        "sos_points_per_game", "sos_rank",
        "injury_risk_score", "trailing_missed_rate", "trailing_severity", "chronic_seasons",
        "season_floor_p10_adjusted", "season_ceiling_p90_adjusted",
    ]
    return board[cols].rename(columns={
        "player_display_name": "player", "recent_team": "team",
        "season_points_adjusted": "proj_season_points", "season_points_healthy": "proj_points_if_fully_healthy",
        "season_floor_p10_adjusted": "season_floor_p10", "season_ceiling_p90_adjusted": "season_ceiling_p90",
    })


if __name__ == "__main__":
    board = build_draft_board(2026)
    board.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "draft_board_2026.csv"), index=False)

    print("=== Overall top 40 ===")
    print(board.head(40).to_string(index=False))

    print("\n=== Biggest injury-risk discounts among top-100 overall ===")
    top100 = board[board["overall_rank"] <= 100]
    print(top100.sort_values("injury_points_lost", ascending=False).head(15)[
        ["overall_rank", "player", "position", "proj_points_if_fully_healthy",
         "proj_season_points", "injury_points_lost", "injury_risk_score", "chronic_seasons"]
    ].to_string(index=False))

    print("\n=== Easiest schedules among top-100 overall (WR/RB) ===")
    print(top100[top100["position"].isin(["WR", "RB"])].sort_values("sos_points_per_game", ascending=False).head(10)[
        ["overall_rank", "player", "position", "team", "sos_points_per_game", "sos_rank"]
    ].to_string(index=False))

    print("\n=== Toughest schedules among top-100 overall (WR/RB) ===")
    print(top100[top100["position"].isin(["WR", "RB"])].sort_values("sos_points_per_game").head(10)[
        ["overall_rank", "player", "position", "team", "sos_points_per_game", "sos_rank"]
    ].to_string(index=False))
