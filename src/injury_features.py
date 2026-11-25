"""
injury_features.py

Builds a per-player, per-season "injury risk score" from two signals:

1. Games actually missed, computed by comparing each player's own team's games
   played that season (from the schedule) against how many of those games the
   player has a stat line for. This is a proxy, not a clinical record -- it also
   catches healthy scratches and suspensions, which is a known limitation (see
   README). It's restricted to players with >=3 games played for that team that
   season, to avoid falsely flagging rookies who simply weren't on the roster yet
   as "missing" the games before they debuted.

2. Injury severity, from the weekly injury report's body-part/injury-type text
   matched to the weeks a player was out. A hand-built severity table scores injury
   types by typical recovery time and recurrence risk (ACL/Achilles far more
   severe than a questionable-designation ankle). Missed games with NO matching
   injury-report entry (common once a player goes on IR -- they stop appearing on
   the weekly "Out" report once officially on IR) default to a higher severity,
   since that pattern usually indicates a longer-term injury rather than a
   game-to-game questionable tag.

The final `injury_risk_score` (0-1) is recency-weighted across the trailing three
seasons, so a torn ACL two years ago that fully healed matters less than a
hamstring that's recurred in each of the last two seasons.
"""

import numpy as np
import pandas as pd

# Severity weights: 0 = minor/short-term, 1 = severe/season-threatening & recurrence-prone.
# Deliberately simple and keyword-matched -- this is a fantasy-relevant proxy, not a
# medical classification.
SEVERITY_WEIGHTS = {
    "acl": 1.0, "achilles": 1.0, "fracture": 0.9, "broken": 0.9, "torn": 0.9,
    "pectoral": 0.75, "acromioclavicular": 0.7, "labrum": 0.7, "meniscus": 0.7,
    "concussion": 0.65, "neck": 0.6, "back": 0.55, "hip": 0.5, "shoulder": 0.5,
    "knee": 0.55, "foot": 0.45, "groin": 0.4, "hamstring": 0.4, "quadricep": 0.4,
    "calf": 0.35, "ankle": 0.3, "rib": 0.3, "ribs": 0.3, "chest": 0.3, "hand": 0.25,
    "wrist": 0.25, "elbow": 0.25, "thumb": 0.2, "finger": 0.2, "toe": 0.2,
    "illness": 0.15, "heel": 0.3, "thigh": 0.35, "abdomen": 0.35, "oblique": 0.35,
}
DEFAULT_UNMATCHED_SEVERITY = 0.65  # missed games w/ no weekly report entry (often IR/long-term)
DEFAULT_MATCHED_SEVERITY = 0.35    # matched but unrecognized injury text

RECENCY_WEIGHTS = {0: 1.0, 1: 0.6, 2: 0.3}  # seasons back -> weight
TRAILING_WINDOW = 3


def _severity_for_text(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return DEFAULT_MATCHED_SEVERITY
    t = text.lower()
    hits = [w for k, w in SEVERITY_WEIGHTS.items() if k in t]
    return max(hits) if hits else DEFAULT_MATCHED_SEVERITY


def _season_games_missed(weekly: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Per (player_id, season, team): games missed vs. that team's REG-season games."""
    team_games = {}
    reg = sched[sched["game_type"] == "REG"]
    for season, grp in reg.groupby("season"):
        counts = pd.concat([grp["home_team"], grp["away_team"]]).value_counts()
        team_games[season] = counts.to_dict()

    appearances = (
        weekly[weekly["season_type"] == "REG"]
        .groupby(["player_id", "season", "recent_team"])["week"]
        .nunique()
        .reset_index()
        .rename(columns={"week": "games_played", "recent_team": "team"})
    )
    appearances["team_games"] = appearances.apply(
        lambda r: team_games.get(r["season"], {}).get(r["team"], np.nan), axis=1
    )
    appearances = appearances.dropna(subset=["team_games"])
    appearances["games_missed"] = (appearances["team_games"] - appearances["games_played"]).clip(lower=0)
    # Only trust "missed games" as an injury signal for seasons where the player was
    # clearly an established starter/full-time role player (>=8 games that season).
    # Below that, low games-played is usually a depth-chart/backup role (e.g. a
    # backup QB with 4 spot starts out of 17 team games), not an injury -- treating
    # it as "missed 13 games" would badly misclassify every long-tenured backup as
    # chronically injured.
    appearances = appearances[appearances["games_played"] >= 8]
    return appearances


def _season_severity(injuries: pd.DataFrame, games_missed: pd.DataFrame) -> pd.DataFrame:
    """For each player-season with missed games, estimate severity from injury-report text."""
    inj = injuries.copy()
    inj["injury_text"] = inj["report_primary_injury"].fillna("") + " " + inj["report_secondary_injury"].fillna("")
    inj["severity"] = inj["injury_text"].apply(_severity_for_text)
    # only weeks flagged Out/Doubtful count as evidence toward a missed game
    inj_out = inj[inj["report_status"].isin(["Out", "Doubtful"])]
    sev_by_player_season = (
        inj_out.groupby(["gsis_id", "season"])["severity"].max().reset_index()
        .rename(columns={"gsis_id": "player_id", "severity": "reported_severity"})
    )

    out = games_missed.merge(sev_by_player_season, on=["player_id", "season"], how="left")
    # players with missed games but no matching injury-report entry get the "unmatched" default
    out["severity_score"] = np.where(
        out["games_missed"] > 0,
        out["reported_severity"].fillna(DEFAULT_UNMATCHED_SEVERITY),
        0.0,
    )
    return out.drop(columns=["reported_severity"])


def build_injury_risk_table(weekly: pd.DataFrame, sched: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (player_id, season) with a recency-weighted, severity-weighted
    `injury_risk_score` in [0, 1], computed ONLY from seasons strictly before `season`
    (so it's safe to use as a pre-season-known feature with no leakage), plus the raw
    components for transparency in the draft-ranking output.
    """
    games_missed = _season_games_missed(weekly, sched)
    season_stats = _season_severity(injuries, games_missed)
    season_stats = season_stats.groupby(["player_id", "season"]).agg(
        games_missed=("games_missed", "max"),
        team_games=("team_games", "max"),
        severity_score=("severity_score", "max"),
    ).reset_index()

    players = season_stats["player_id"].unique()
    all_seasons = sorted(season_stats["season"].unique())
    rows = []
    for season in list(all_seasons) + [max(all_seasons) + 1]:  # +1 lets us score the upcoming season too
        window = [season - k for k in range(1, TRAILING_WINDOW + 1)]
        hist = season_stats[season_stats["season"].isin(window)].copy()
        if hist.empty:
            continue
        hist["weight"] = hist["season"].apply(lambda s: RECENCY_WEIGHTS.get(season - s - 1, 0.0))
        hist["missed_rate"] = (hist["games_missed"] / hist["team_games"]).clip(0, 1)

        for pid, g in hist.groupby("player_id"):
            w = g["weight"]
            if w.sum() == 0:
                continue
            trailing_missed_rate = np.average(g["missed_rate"], weights=w)
            # severity only "counts" in proportion to games actually missed that season
            sev_weights = w * g["games_missed"].clip(lower=0.01)
            trailing_severity = (
                np.average(g["severity_score"], weights=sev_weights) if sev_weights.sum() > 0 else 0.0
            )
            chronic_flag = (g["games_missed"] >= 2).sum()  # seasons in window w/ 2+ games missed

            risk = 0.55 * trailing_missed_rate + 0.30 * trailing_severity + 0.15 * min(chronic_flag / 3, 1.0)
            rows.append({
                "player_id": pid,
                "season": season,
                "injury_risk_score": round(float(np.clip(risk, 0, 1)), 4),
                "trailing_missed_rate": round(float(trailing_missed_rate), 4),
                "trailing_severity": round(float(trailing_severity), 4),
                "chronic_seasons": int(chronic_flag),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import load_all

    weekly, sched, snaps, injuries = load_all(list(range(2016, 2026)))
    risk = build_injury_risk_table(weekly, sched, injuries)
    print(risk.shape)
    print(risk.sort_values("injury_risk_score", ascending=False).head(15).to_string(index=False))
