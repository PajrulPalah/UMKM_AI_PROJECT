#!/usr/bin/env python3
# ============================================================
# retrain.py
# Auto-Retraining Pipeline
# Dipanggil otomatis ketika ada data submission baru
# ============================================================

"""
Script ini akan:
1. Membaca semua data submission dari data/submissions/submissions_log.csv
2. Menggabungkan dengan synthetic dataset asli
3. Melatih ulang semua 4 model dengan data gabungan
4. Menyimpan model baru ke folder model/
5. Menyimpan log retraining ke data/retraining_log.json
"""

import os
import sys
import warnings
import json
import datetime
import time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
MODEL_DIR   = ROOT / "model"
LOG_DIR     = DATA_DIR / "submissions"

RANDOM_STATE = 42
CLASS_ORDER  = ["Low", "Medium", "High"]
FEATURE_COLS = [
    "DigitalCapabilityScore",
    "InnovationCapabilityScore",
    "EntrepreneurialOrientationScore",
    "OrganizationalAgilityScore",
    "ResourceAccessScore",
    "EnvironmentalDynamismScore",
]
CONSTRUCT_ITEMS = {
    "DigitalCapabilityScore":          ["DC1","DC2","DC3","DC4","DC5"],
    "InnovationCapabilityScore":       ["IC1","IC2","IC3","IC4","IC5"],
    "EntrepreneurialOrientationScore": ["EO1","EO2","EO3","EO4","EO5"],
    "OrganizationalAgilityScore":      ["OA1","OA2","OA3","OA4","OA5"],
    "ResourceAccessScore":             ["RA1","RA2","RA3","RA4","RA5"],
    "EnvironmentalDynamismScore":      ["ED1","ED2","ED3","ED4","ED5"],
    "BusinessResilienceScore":         ["BR1","BR2","BR3","BR4","BR5","BR6","BR7"],
}
TARGET_COL = "BusinessResilienceCategory"

NAME_MAP = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree":       "decision_tree",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}

np.random.seed(RANDOM_STATE)


def load_base_dataset() -> pd.DataFrame:
    """Load original synthetic dataset."""
    path = DATA_DIR / "UMKM_Dummy_Data.csv"
    if path.exists():
        df = pd.read_csv(path)
        # Ensure construct scores are present
        for score_col, items in CONSTRUCT_ITEMS.items():
            cols = [c for c in items if c in df.columns]
            if cols:
                df[score_col] = df[cols].mean(axis=1).round(3)
        if TARGET_COL not in df.columns:
            br_s = df.get("BusinessResilienceScore", pd.Series([3]*len(df)))
            q33, q67 = br_s.quantile(1/3), br_s.quantile(2/3)
            df[TARGET_COL] = br_s.apply(
                lambda s: "Low" if s <= q33 else ("Medium" if s <= q67 else "High")
            )
        return df
    return pd.DataFrame()


def load_submissions() -> pd.DataFrame:
    """Load all user submissions."""
    path = LOG_DIR / "submissions_log.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        # Recompute construct scores from raw items if needed
        for score_col, items in CONSTRUCT_ITEMS.items():
            cols = [c for c in items if c in df.columns]
            if cols and score_col not in df.columns:
                df[score_col] = df[cols].mean(axis=1).round(3)
        return df
    except Exception as e:
        print(f"  Warning: Cannot read submissions: {e}")
        return pd.DataFrame()


def build_training_set(base_df: pd.DataFrame, subs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge base synthetic data with real user submissions.
    Submissions without a label get label from self-assessed BR score.
    """
    frames = []

    if len(base_df) > 0:
        base_cols = FEATURE_COLS + [TARGET_COL]
        base_sub = base_df[[c for c in base_cols if c in base_df.columns]].copy()
        base_sub = base_sub.dropna(subset=FEATURE_COLS)
        frames.append(base_sub)

    if len(subs_df) > 0:
        # Build label from self-assessed BR items if available
        br_items = [c for c in ["BR1","BR2","BR3","BR4","BR5","BR6","BR7"] if c in subs_df.columns]
        if br_items:
            subs_df["BusinessResilienceScore"] = subs_df[br_items].mean(axis=1)
        
        # Use predicted category as label if BR score not available
        if TARGET_COL not in subs_df.columns:
            if "Predicted_Category" in subs_df.columns:
                subs_df[TARGET_COL] = subs_df["Predicted_Category"]
            elif "BusinessResilienceScore" in subs_df.columns:
                br_s = subs_df["BusinessResilienceScore"]
                q33, q67 = br_s.quantile(1/3), br_s.quantile(2/3)
                subs_df[TARGET_COL] = br_s.apply(
                    lambda s: "Low" if s <= q33 else ("Medium" if s <= q67 else "High")
                )

        sub_cols = FEATURE_COLS + [TARGET_COL]
        subs_sub = subs_df[[c for c in sub_cols if c in subs_df.columns]].copy()
        subs_sub = subs_sub.dropna(subset=FEATURE_COLS)
        frames.append(subs_sub)

    if not frames:
        raise ValueError("No training data available!")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return combined


def train_and_save(X_train: np.ndarray, y_train, X_val: np.ndarray, y_val,
                   scaler: StandardScaler, le: LabelEncoder) -> dict:
    """Train all 4 models, save to disk, return metrics."""

    MODEL_CONFIGS = {
        "Logistic Regression": {
            "model": LogisticRegression(solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE),
            "param_grid": {"C": [0.1, 1.0, 10.0]},
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {"max_depth": [3, 5, 7], "min_samples_split": [2, 5]},
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {"n_estimators": [100, 200], "max_depth": [5, 8, None]},
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE, verbosity=0),
            "param_grid": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.1, 0.2]},
        },
    }

    metrics = {}

    for model_name, config in MODEL_CONFIGS.items():
        t0 = time.time()
        is_xgb = model_name == "XGBoost"
        y_tr   = le.transform(y_train) if is_xgb else y_train
        y_vl   = le.transform(y_val)   if is_xgb else y_val

        gs = GridSearchCV(
            config["model"], config["param_grid"],
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1_weighted", n_jobs=-1, refit=True, verbose=0
        )
        gs.fit(X_train, y_tr)
        best_model = gs.best_estimator_

        # Evaluate on validation set
        y_pred = best_model.predict(X_val)
        f1  = f1_score(y_vl, y_pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_vl, y_pred)

        # Save model
        fname = NAME_MAP.get(model_name, model_name.lower().replace(" ", "_"))
        fpath = MODEL_DIR / f"{fname}.pkl"
        payload = {
            "model": best_model,
            "scaler": scaler,
            "label_encoder": le,
            "feature_cols": FEATURE_COLS,
            "class_order": CLASS_ORDER,
            "model_name": model_name,
            "best_params": gs.best_params_,
        }
        joblib.dump(payload, fpath, compress=3)
        elapsed = time.time() - t0

        metrics[model_name] = {"f1": round(f1, 4), "acc": round(acc, 4)}
        print(f"  {model_name:25s} | F1: {f1:.4f} | Acc: {acc:.4f} | {elapsed:.1f}s → {fname}.pkl")

    return metrics


def main():
    start_time = datetime.datetime.now()
    print("=" * 60)
    print("  UMKM AI — Auto-Retraining Pipeline")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 1. Load data ────────────────────────────────────────
    print("\n[1/5] Loading datasets...")
    base_df = load_base_dataset()
    subs_df = load_submissions()
    print(f"  Base synthetic data: {len(base_df)} records")
    print(f"  User submissions:    {len(subs_df)} records")

    # ── 2. Build combined training set ──────────────────────
    print("\n[2/5] Building combined training set...")
    try:
        combined = build_training_set(base_df, subs_df)
    except ValueError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    print(f"  Total training samples: {len(combined)}")
    dist = combined[TARGET_COL].value_counts()
    print(f"  Class distribution: {dict(dist)}")

    X = combined[FEATURE_COLS].values
    y = combined[TARGET_COL].values

    # ── 3. Scale & encode ───────────────────────────────────
    print("\n[3/5] Preprocessing...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    le.fit(CLASS_ORDER)

    # SMOTE if enough samples
    if SMOTE_AVAILABLE and len(combined) >= 30:
        min_count = dist.min()
        k = min(5, min_count - 1) if min_count > 1 else 1
        try:
            smote = SMOTE(sampling_strategy="auto", random_state=RANDOM_STATE, k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_scaled, y)
            print(f"  SMOTE: {len(y)} → {len(y_train)} samples")
        except Exception as e:
            print(f"  SMOTE skipped: {e}")
            X_train, y_train = X_scaled, y
    else:
        X_train, y_train = X_scaled, y
        print("  SMOTE skipped (insufficient data)")

    # Use last 20% as validation (if enough data)
    val_size = max(int(len(X_scaled) * 0.2), min(10, len(X_scaled)))
    X_val = X_scaled[-val_size:]
    y_val = y[-val_size:]

    # ── 4. Train models ─────────────────────────────────────
    print("\n[4/5] Training all 4 models...")
    metrics = train_and_save(X_train, y_train, X_val, y_val, scaler, le)

    # ── 5. Save retraining log ──────────────────────────────
    print("\n[5/5] Saving retraining log...")
    log = {
        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_base": int(len(base_df)),
        "n_submissions": int(len(subs_df)),
        "n_total": int(len(combined)),
        "duration_seconds": round((datetime.datetime.now() - start_time).total_seconds(), 1),
        "metrics": metrics,
    }
    log_path = DATA_DIR / "retraining_log.json"
    history  = []
    if log_path.exists():
        try:
            history = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history.append(log)
    log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  RETRAINING COMPLETE!")
    print(f"  Duration: {log['duration_seconds']}s")
    print(f"  Best F1: {max(m['f1'] for m in metrics.values()):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
