# NFL Fantasy Football Projection Model

Position-specific machine learning models (separate LightGBM model per position:
QB / RB / WR / TE) that project weekly PPR fantasy points, trained on 2016–2025
data and validated with walk-forward (season-by-season) backtesting.

## Why separate models per position

A QB's scoring is driven by passing efficiency and volume, is comparatively low
variance, and has a small player pool. A WR's scoring depends heavily on target
competition within their own offense, is much higher variance ("boom/bust"), and
has a large, thin-tailed player pool. Pooling all positions into one model washes
out these differences. Each position here gets its own feature set and its own
LightGBM hyperparameters (see `POSITION_PARAMS` in `src/train.py`).

## Project layout

```
src/
  data_loader.py         # pulls + caches weekly stats, schedules/Vegas lines, snap counts, injury reports
  injury_features.py     # per-player injury risk score: games missed + severity + recency
  feature_engineering.py # leakage-safe rolling features, opponent-defense-vs-position, game script
  train.py                # trains 4 position models with walk-forward validation
  evaluate.py             # feature importance, quantile calibration
  predict.py               # generates weekly AND full-season projections + floor/ceiling bands
  draft_rankings.py        # season projections -> injury penalty -> Value Over Replacement -> draft board
models/                   # trained LightGBM boosters (.txt) + feature lists (.json) per position
week1_2026_projections.csv  # example output: full player pool for one scheduled week
draft_board_2026.csv        # example output: full VOR-ranked draft board for the season
```

## Data sources (all free, pulled live)

- **Weekly player stats**: nflverse (`nfl_data_py` for 2016–2024; nflverse's newer
  `stats_player_week` release directly for 2025+, since nflverse restructured this
  release ahead of the 2025 season and `nfl_data_py` 0.3.3 hasn't been updated for
  the new schema — `data_loader.py` harmonizes both into one consistent format)
- **Schedules + Vegas lines**: `nflverse/nfldata` `games.csv` (spread, total, rest
  days, weather, roof/surface)
- **Snap counts**: nflverse `snap_counts` release
- **Injury reports**: nflverse `injuries` release (weekly practice-report
  designations — Questionable/Doubtful/Out — plus the reported body part)

## Draft rankings & injury penalty

`draft_rankings.py` turns the weekly models into a single ranked draft board:

1. **Season projection**: `predict.py`'s `PredictionContext.project_season()`
   projects every scheduled week of the season (using each player's most recent
   rolling feature snapshot against that week's actual opponent/Vegas context) and
   sums them, with an approximate season-level floor/ceiling built from each
   week's quantile spread.

2. **Injury penalty** (`injury_features.py` + `draft_rankings.py`): a per-player
   `injury_risk_score` is built from real injury-report + missed-game history —
   recency-weighted across the trailing 3 seasons, so a two-years-ago ACL tear
   that's fully healed matters less than a hamstring that's recurred each of the
   last two years. It's applied to the draft board as **two separate, auditable
   penalties**, not folded silently into one number:
   - **Availability discount** (quantity) — reduces *expected games played*,
     driven by how often the player has actually missed games recently. Capped at
     50% (even an oft-injured player is assumed to play at least half the season).
   - **Severity discount** (quality) — a smaller discount (capped at 15%) to
     *points-per-game*, reflecting that players returning from more severe injury
     types (ACL, Achilles) often play somewhat diminished even in games they do
     suit up for. Severity is scored from a hand-built body-part/injury-type
     table (`SEVERITY_WEIGHTS` in `injury_features.py`) — ACL/Achilles/fracture
     score near 1.0, ankle/illness/finger score near 0.2–0.3.

   Both discounts are visible separately in the output (`injury_risk_score`,
   `trailing_missed_rate`, `trailing_severity`, `expected_games_played`,
   `injury_points_lost`) so you can see exactly how much of a player's discount
   comes from missing time versus reduced effectiveness.

   The `injury_risk_score` is *also* fed into the weekly model itself as a
   feature (so the model can learn patterns like teams managing snap counts down
   for oft-injured players), but that's separate from this explicit draft-board
   penalty — the weekly model only ever sees rows where the player suited up, so
   it has no way to discount for games missed entirely on its own.

3. **Value Over Replacement (VOR)**: for each position, the "replacement level"
   is the Nth-best player's projected points (default: QB12 / RB24 / WR30 / TE12,
   i.e. the last startable player at each position in a standard 12-team league —
   tune `REPLACEMENT_RANK` in `draft_rankings.py` for your league size/format).
   `vor = player's adjusted season points - replacement level`. Sorting by VOR
   (not raw points) is what makes positions comparable on the same draft board —
   a QB and a TE scoring the same raw points are not equally valuable if
   replacement-level QBs score much higher than replacement-level TEs.

## Strength of schedule

Schedule difficulty is already baked into the season projection itself — each of
a player's up-to-17 weekly projections uses that week's *actual* 2026 opponent,
with `opp_def_vs_pos_avg` (that opponent's real 2025 fantasy-points-allowed to the
position) as a model feature, so a player projects higher in a week against a
weak defense and lower against a strong one. `project_season()` sums those 17
opponent-specific weeks rather than repeating one average matchup 17 times.

That effect is also now surfaced explicitly as its own column rather than staying
invisible inside the season total: `sos_points_per_game` is the player's average
opponent-strength-faced minus the league-average opponent at that position
(positive = easier-than-average schedule, negative = tougher), with `sos_rank`
giving the 1-to-N ranking within the position. In the current 2026 board, e.g.
A.J. Brown / DeVonta Smith (Eagles) have the easiest projected WR schedule and
the Chicago-based skill players have one of the toughest — driven by their
opponents' 2025 defensive performance against the position. The effect size is
modest (well under 1 point/game in most cases), which matches how strength of
schedule actually behaves in fantasy — it's real but rarely swings a season by
more than a player's natural week-to-week variance.

**Injury data limitation worth knowing**: "games missed" is inferred by comparing
a player's own team's games played to their own appearances in the stat log — this
also catches healthy scratches and suspensions, and is restricted to seasons with
≥8 games played to avoid misreading a backup's limited role as an injury (an
earlier version of this without that filter badly misclassified long-tenured
backup QBs as "chronically injured," since a backup with 4 spot starts out of 17
team games looks identical to a starter who missed 13 games hurt — worth knowing
if a specific player's score looks off).

## How the leakage-safe design works

Every rolling average (3/5/8-game windows) and every opponent-defense-strength
number is computed with `.shift(1)` before being attached to a row, so a player's
week-N feature row never contains information from week N itself. Validation is
walk-forward by season — train on seasons before the test season, never randomly
shuffled — which matches how the model would actually be used and avoids the
leakage a random k-fold split would introduce on time-series data.

## Results (mean absolute error, walk-forward across 2019–2025 test seasons)

| Position | Model MAE | Rolling-avg baseline MAE | Improvement |
|---|---|---|---|
| QB | 6.42 | 6.61 | 2.9% |
| RB | 5.12 | 5.20 | 1.5% |
| WR | 5.10 | 5.21 | 2.0% |
| TE | 3.90 | 3.98 | 2.0% |

The baseline is each player's own trailing 5-game average — a strong baseline in
fantasy football, since usage/role is highly persistent week to week. The model's
edge over it is real but modest, which is an honest reflection of how much of
weekly fantasy variance is genuinely unpredictable (game flow, garbage time,
injuries mid-game, td variance). Full per-season numbers are in
`models/walk_forward_results.csv`.

## Top predictive features by position (gain-based importance)

- **QB**: trailing PPR points (r8/r3, ~35% of gain combined), snap share, team
  spread (game script), opponent defense vs. QB, implied team total, dakota
- **RB**: trailing PPR points, snap share (13.5% of gain — the single strongest
  RB signal after scoring history), carries trend, team spread
- **WR**: trailing PPR points, **target share** (9.3% of gain), targets, snap
  share, receiving yards
- **TE**: **WOPR** (weighted opportunity rating — 22% of gain, by far the
  strongest single TE signal), receiving yards, snap share, targets

Run `python src/evaluate.py` for the full top-15 list per position.

## Usage

```bash
pip install -r requirements.txt

# pull data + train all 4 position models (writes to models/)
python src/train.py

# check feature importance
python src/evaluate.py

# generate projections for a given week
python -c "from src.predict import project_week; print(project_week(2026, 1).head(20))"

# build the full VOR-ranked draft board (writes draft_board_2026.csv)
python src/draft_rankings.py
```

## Where to take this further

- Predict scoring **components** (yards, TDs, receptions) separately instead of
  fantasy points directly — more interpretable, easier to debug specific failure
  modes (e.g. the model overrating TD probability specifically)
- Add play-by-play-derived red-zone usage (current features approximate this via
  season-level TD rates, not red-zone-specific target/carry share)
- Add real-time injury-report data at draft time (currently only *historical*
  injury patterns feed the risk score — a player's actual current-week
  questionable/out designation isn't part of this, since the draft board is a
  preseason tool)
- Roster/depth-chart awareness for players who changed teams in free agency —
  the projection snapshot uses each player's most recent team on record, which is
  usually but not always still accurate by the time of a new season
- Refine the injury severity table with real medical recovery-time data instead
  of the current hand-built keyword weights
- Monte Carlo game-script simulation on top of the quantile models, for
  lineup-level win-probability rather than isolated player projections
- Draft-pick-value curves (e.g. positional runs, ADP comparison) layered on top
  of the VOR ranking for live-draft assistance rather than a static pre-draft board
