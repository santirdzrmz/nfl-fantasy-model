"""
predict.py
Generates projections for an upcoming week, or for a full season (summed across
every scheduled week), using each player's most recent rolling features joined
against the actual matchup context for each game on the schedule.

Rolling features for "next week" are the player's existing r3/r5/r8 values carried
forward unshifted (they already only look backward), so no retraining or re-shifting
is needed at inference time -- this is the natural consequence of the leakage-safe
design in feature_engineering.py.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from data_loader import load_all, load_schedules
from feature_engineering import build_feature_table, FEATURE_COLUMNS

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
POSITIONS = ["QB", "RB", "WR", "TE"]


def _load_position_models(pos: str):
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"{pos}_model.txt"))
    q10 = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"{pos}_q10.txt"))
    q90 = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"{pos}_q90.txt"))
    with open(os.path.join(MODEL_DIR, f"{pos}_features.json")) as f:
        features = json.load(f)
    return booster, q10, q90, features


class PredictionContext:
    """Builds feature tables + per-position latest-known snapshot ONCE, so projecting
    every week of a season doesn't redo the expensive feature engineering 18 times."""

    def __init__(self, season: int, years_history=range(2016, 2026)):
        self.season = season
        weekly, sched, snaps, injuries = load_all(list(years_history))
        if season not in sched["season"].values:
            extra_sched = load_schedules([season], refresh=True)
            sched = pd.concat([sched, extra_sched], ignore_index=True)
        self.sched = sched
        self.tables = build_feature_table(weekly, sched, snaps, injuries)

        self.latest = {}
        self.opp_strength = {}
        self.models = {}
        for pos in POSITIONS:
            df = self.tables[pos]
            prior = df[(df["season"] < season)]
            if prior.empty:
                self.latest[pos] = pd.DataFrame()
                continue
            recent_season = prior["season"].max()
            prior = prior[prior["season"] == recent_season]
            self.latest[pos] = prior.sort_values(["season", "week"]).groupby("player_id").tail(1).copy()
            self.opp_strength[pos] = (
                df[df["season"] == recent_season].groupby("opponent_team")["opp_def_vs_pos_avg"].last()
            )
            self.models[pos] = _load_position_models(pos)

    def team_game_for_week(self, week: int) -> dict:
        wk_games = self.sched[(self.sched["season"] == self.season) & (self.sched["week"] == week)]
        team_to_game = {}
        for _, g in wk_games.iterrows():
            team_to_game[g["home_team"]] = g
            team_to_game[g["away_team"]] = g
        return team_to_game

    def project_week(self, week: int) -> pd.DataFrame:
        team_to_game = self.team_game_for_week(week)
        if not team_to_game:
            return pd.DataFrame()

        all_proj = []
        for pos in POSITIONS:
            latest = self.latest.get(pos)
            if latest is None or latest.empty:
                continue

            rows = []
            for _, r in latest.iterrows():
                team = r["recent_team"]
                if team not in team_to_game:
                    continue
                g = team_to_game[team]
                is_home = int(team == g["home_team"])
                opponent = g["away_team"] if is_home else g["home_team"]
                home_implied = g["total_line"] / 2 - g["spread_line"] / 2
                away_implied = g["total_line"] / 2 + g["spread_line"] / 2
                row = r.copy()
                row["is_home"] = is_home
                row["opponent_team"] = opponent
                row["implied_team_total"] = home_implied if is_home else away_implied
                row["team_spread"] = g["spread_line"] if is_home else -g["spread_line"]
                row["rest_days"] = g["home_rest"] if is_home else g["away_rest"]
                row["div_game"] = g["div_game"]
                row["is_dome"] = 1 if g["roof"] in ("dome", "closed") else 0
                row["week"] = week
                rows.append(row)

            if not rows:
                continue
            pred_df = pd.DataFrame(rows)
            pred_df["opp_def_vs_pos_avg"] = pred_df["opponent_team"].map(self.opp_strength[pos]).fillna(
                pred_df["opp_def_vs_pos_avg"]
            )

            booster, q10, q90, feat_cols = self.models[pos]
            X = pred_df[feat_cols]
            pred_df["proj_points"] = np.clip(booster.predict(X), 0, None)
            pred_df["proj_floor_p10"] = np.clip(q10.predict(X), 0, None)
            pred_df["proj_ceiling_p90"] = np.clip(q90.predict(X), 0, None)
            pred_df["position"] = pos

            all_proj.append(pred_df[[
                "player_id", "player_display_name", "position", "recent_team", "opponent_team", "is_home",
                "implied_team_total", "team_spread", "opp_def_vs_pos_avg",
                "proj_floor_p10", "proj_points", "proj_ceiling_p90",
            ]])

        if not all_proj:
            return pd.DataFrame()
        result = pd.concat(all_proj, ignore_index=True)
        return result.sort_values("proj_points", ascending=False).reset_index(drop=True)

    def scheduled_weeks_for_team(self, team: str) -> list:
        s = self.sched[self.sched["season"] == self.season]
        weeks = s[(s["home_team"] == team) | (s["away_team"] == team)]["week"].unique()
        return sorted(int(w) for w in weeks)

    def project_season(self, max_week: int = 18) -> pd.DataFrame:
        """Sums per-week projections across every scheduled game. Season variance is
        approximated from each week's quantile spread (treated as independent), since
        we don't have a joint distribution across weeks -- a reasonable approximation
        for a draft-ranking tool, not a precise interval."""
        weekly_frames = []
        for week in range(1, max_week + 1):
            wk = self.project_week(week)
            if not wk.empty:
                weekly_frames.append(wk)
        if not weekly_frames:
            return pd.DataFrame()

        all_weeks = pd.concat(weekly_frames, ignore_index=True)
        all_weeks["week_std"] = (all_weeks["proj_ceiling_p90"] - all_weeks["proj_floor_p10"]) / 2.563  # 80% CI -> std

        season = all_weeks.groupby(["player_id", "player_display_name", "position", "recent_team"]).agg(
            games_scheduled=("proj_points", "size"),
            season_points_healthy=("proj_points", "sum"),
            season_variance=("week_std", lambda s: (s ** 2).sum()),
            avg_opp_strength_faced=("opp_def_vs_pos_avg", "mean"),
        ).reset_index()
        season["season_std"] = np.sqrt(season["season_variance"])
        season["season_floor_p10"] = np.clip(season["season_points_healthy"] - 1.2816 * season["season_std"], 0, None)
        season["season_ceiling_p90"] = season["season_points_healthy"] + 1.2816 * season["season_std"]
        season = season.drop(columns=["season_variance"])

        # Strength of schedule: how much fantasy production this player's actual
        # opponents allow to this position, on average, vs. the league-average
        # opponent at this position. Positive = easier-than-average schedule
        # (opponents faced allow MORE points to the position); negative = tougher.
        for pos in POSITIONS:
            if pos not in self.opp_strength:
                continue
            league_avg = self.opp_strength[pos].mean()
            mask = season["position"] == pos
            season.loc[mask, "sos_points_per_game"] = season.loc[mask, "avg_opp_strength_faced"] - league_avg
        season["sos_rank"] = season.groupby("position")["sos_points_per_game"].rank(ascending=False, method="min").astype(int)
        return season


def project_week(season: int, week: int, years_history=range(2016, 2026)) -> pd.DataFrame:
    ctx = PredictionContext(season, years_history)
    return ctx.project_week(week)


if __name__ == "__main__":
    proj = project_week(season=2026, week=1)
    print(proj.head(30).to_string(index=False))
