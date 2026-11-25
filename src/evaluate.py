"""
evaluate.py
Feature importance per position model (gain-based from LightGBM; SHAP if installed)
and a boom/bust calibration check on the quantile models.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_model(pos: str):
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"{pos}_model.txt"))
    with open(os.path.join(MODEL_DIR, f"{pos}_features.json")) as f:
        features = json.load(f)
    return booster, features


def feature_importance(pos: str, top_n=15) -> pd.DataFrame:
    booster, features = load_model(pos)
    gains = booster.feature_importance(importance_type="gain")
    imp = pd.DataFrame({"feature": features, "gain": gains})
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
    return imp.sort_values("gain_pct", ascending=False).head(top_n).reset_index(drop=True)


def coverage_check(pos: str, df: pd.DataFrame, target_col: str) -> dict:
    """What fraction of actual outcomes fell within the [10th, 90th] predicted band?
    Should be close to 80% if the quantile models are well calibrated."""
    q10 = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"{pos}_q10.txt"))
    q90 = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"{pos}_q90.txt"))
    _, features = load_model(pos)
    lo = np.clip(q10.predict(df[features]), 0, None)
    hi = np.clip(q90.predict(df[features]), 0, None)
    within = ((df[target_col] >= lo) & (df[target_col] <= hi)).mean()
    return {"position": pos, "target_coverage": 0.80, "actual_coverage": round(float(within), 3)}


if __name__ == "__main__":
    for pos in ["QB", "RB", "WR", "TE"]:
        print(f"\n=== {pos} top features (by gain) ===")
        print(feature_importance(pos).to_string(index=False))
