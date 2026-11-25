"""
train.py

Trains one LightGBM model per position (QB, RB, WR, TE), each with its own feature
space and its own hyperparameters — this is the "different architecture per position"
piece: a QB's scoring is driven by very different mechanics (passing efficiency, low
variance) than a WR's (target competition, high boom/bust), so treating them as one
pooled model washes out position-specific signal.

Validation is walk-forward by season: train on all seasons < test_season, predict
test_season. This mirrors how the model would actually be used (predicting a future
season from past ones) and avoids the leakage you'd get from random k-fold CV on
time-series data.

Also fits a quantile model (10th/90th percentile) per position so predictions come
with an uncertainty band, not just a point estimate — useful for lineup decisions
where a jump-ball WR3 and a low-variance possession TE can have the same mean but
very different risk profiles.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_all
from feature_engineering import build_feature_table, FEATURE_COLUMNS, TARGET

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Position-specific hyperparameters. Positions with more data / more variance
# (WR) get more leaves and lower LR; low-sample positions (QB, TE) are constrained
# harder to avoid overfitting.
POSITION_PARAMS = {
    "QB": dict(num_leaves=15, learning_rate=0.03, min_child_samples=25, n_estimators=600,
               reg_alpha=0.5, reg_lambda=0.5, subsample=0.8, colsample_bytree=0.8),
    "RB": dict(num_leaves=20, learning_rate=0.03, min_child_samples=20, n_estimators=700,
               reg_alpha=0.3, reg_lambda=0.3, subsample=0.8, colsample_bytree=0.8),
    "WR": dict(num_leaves=31, learning_rate=0.025, min_child_samples=20, n_estimators=800,
               reg_alpha=0.3, reg_lambda=0.3, subsample=0.8, colsample_bytree=0.7),
    "TE": dict(num_leaves=15, learning_rate=0.03, min_child_samples=20, n_estimators=600,
               reg_alpha=0.5, reg_lambda=0.5, subsample=0.8, colsample_bytree=0.8),
}

MIN_TRAIN_SEASONS = 3  # need at least this many seasons of history before first test season


def baseline_prediction(df: pd.DataFrame) -> pd.Series:
    """Naive baseline: player's own rolling-5-game average. Every model must beat this."""
    col = f"{TARGET}_r5"
    return df[col].fillna(df[f"{TARGET}_r8"]).fillna(df[f"{TARGET}_r3"])


def walk_forward_splits(seasons: list, min_train_seasons=MIN_TRAIN_SEASONS):
    seasons = sorted(seasons)
    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]
        yield train_seasons, test_season


def train_position_model(pos: str, df: pd.DataFrame, feature_cols: list, seasons: list):
    """Walk-forward evaluation + final model fit on all available data."""
    results = []
    for train_seasons, test_season in walk_forward_splits(seasons):
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]
        if len(test_df) < 20:
            continue

        X_train, y_train = train_df[feature_cols], train_df[TARGET]
        X_test, y_test = test_df[feature_cols], test_df[TARGET]

        model = lgb.LGBMRegressor(objective="regression", random_state=42,
                                   verbosity=-1, **POSITION_PARAMS[pos])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)  # fantasy points floor at 0

        base = baseline_prediction(test_df).fillna(y_train.mean())

        results.append({
            "position": pos,
            "test_season": test_season,
            "n_test": len(test_df),
            "model_mae": mean_absolute_error(y_test, preds),
            "model_rmse": mean_squared_error(y_test, preds) ** 0.5,
            "baseline_mae": mean_absolute_error(y_test, base),
            "baseline_rmse": mean_squared_error(y_test, base) ** 0.5,
        })

    # final model: train on everything, this is what ships for future predictions
    X_all, y_all = df[feature_cols], df[TARGET]
    final_model = lgb.LGBMRegressor(objective="regression", random_state=42,
                                     verbosity=-1, **POSITION_PARAMS[pos])
    final_model.fit(X_all, y_all)

    # quantile models for uncertainty band (floor/ceiling projections)
    q_models = {}
    for q in (0.1, 0.9):
        qm = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=42,
                                verbosity=-1, **POSITION_PARAMS[pos])
        qm.fit(X_all, y_all)
        q_models[q] = qm

    return final_model, q_models, pd.DataFrame(results)


def run_training(years=range(2016, 2026)):
    weekly, sched, snaps, injuries = load_all(list(years))
    tables = build_feature_table(weekly, sched, snaps, injuries)

    all_results = []
    for pos in ["QB", "RB", "WR", "TE"]:
        df = tables[pos].copy()
        feature_cols = FEATURE_COLUMNS[pos]
        seasons = sorted(df["season"].unique())

        model, q_models, results = train_position_model(pos, df, feature_cols, seasons)
        all_results.append(results)

        model.booster_.save_model(os.path.join(MODEL_DIR, f"{pos}_model.txt"))
        for q, qm in q_models.items():
            qm.booster_.save_model(os.path.join(MODEL_DIR, f"{pos}_q{int(q*100)}.txt"))
        with open(os.path.join(MODEL_DIR, f"{pos}_features.json"), "w") as f:
            json.dump(feature_cols, f)

        print(f"\n=== {pos} ===")
        print(results.to_string(index=False))
        if not results.empty:
            print(f"  Mean MAE  -> model: {results.model_mae.mean():.3f}  "
                  f"baseline: {results.baseline_mae.mean():.3f}  "
                  f"(improvement: {(1 - results.model_mae.mean()/results.baseline_mae.mean())*100:.1f}%)")

    full_results = pd.concat(all_results, ignore_index=True)
    full_results.to_csv(os.path.join(MODEL_DIR, "walk_forward_results.csv"), index=False)
    return full_results


if __name__ == "__main__":
    run_training()
