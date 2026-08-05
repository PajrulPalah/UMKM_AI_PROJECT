# ============================================================
# training.py
# UMKM AI Business Resilience Prediction System
# Module: Model Training & Hyperparameter Tuning
# Author: AI Engineer
# Version: 1.0.0
# ============================================================

"""
Training module for UMKM Business Resilience Prediction System.

Handles:
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    - XGBoost Classifier
    - GridSearchCV Hyperparameter Tuning
    - SMOTE Oversampling for class imbalance
    - Model persistence with Joblib
"""

import os
import warnings
import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MODEL_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
RANDOM_STATE: int = 42
CV_FOLDS: int = 10


# ─────────────────────────────────────────────
# 1. MODEL DEFINITIONS
# ─────────────────────────────────────────────

def get_base_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Return a dictionary of base (un-tuned) ML models.

    Models included:
        - Logistic Regression  : Linear probabilistic classifier
        - Decision Tree        : Rule-based tree classifier
        - Random Forest        : Ensemble of decision trees
        - XGBoost              : Gradient-boosted ensemble

    Args:
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary mapping model name → sklearn-compatible estimator.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
            C=1.0,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_state,
            verbosity=0,
        ),
    }
    logger.info(f"✅ Loaded {len(models)} base models.")
    return models


# ─────────────────────────────────────────────
# 2. HYPERPARAMETER GRIDS
# ─────────────────────────────────────────────

def get_hyperparameter_grids() -> Dict[str, Dict[str, List[Any]]]:
    """
    Return hyperparameter search grids for GridSearchCV.

    Returns:
        Dictionary mapping model name → parameter grid.
    """
    grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "max_iter": [500, 1000],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 8, 10, None],
            "min_samples_split": [2, 5],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
        },
    }
    return grids


# ─────────────────────────────────────────────
# 3. SMOTE OVERSAMPLING
# ─────────────────────────────────────────────

def apply_smote(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
    strategy: str = "auto",
) -> Tuple[np.ndarray, pd.Series]:
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to training data.

    Used when class distribution is imbalanced. SMOTE synthetically
    generates minority-class samples to balance the training set.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        random_state: Reproducibility seed.
        strategy: SMOTE sampling strategy ('auto', 'minority', etc.).

    Returns:
        Tuple of (X_resampled, y_resampled).
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            f"✅ SMOTE applied | Before: {len(y_train)} | After: {len(y_res)}"
        )
        return X_res, y_res
    except ImportError:
        logger.warning("⚠️ imbalanced-learn not found. SMOTE skipped.")
        return X_train, y_train
    except Exception as e:
        logger.warning(f"⚠️ SMOTE failed: {e}. Using original data.")
        return X_train, y_train


# ─────────────────────────────────────────────
# 4. CROSS VALIDATION
# ─────────────────────────────────────────────

def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: pd.Series,
    cv: int = CV_FOLDS,
    scoring: str = "f1_weighted",
    random_state: int = RANDOM_STATE,
) -> Dict[str, float]:
    """
    Perform stratified k-fold cross-validation.

    Args:
        model: Fitted or un-fitted sklearn estimator.
        X: Feature matrix.
        y: Target labels.
        cv: Number of cross-validation folds.
        scoring: Scoring metric (default: 'f1_weighted').
        random_state: Random seed.

    Returns:
        Dictionary with mean and std of CV scores.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    result = {
        "cv_mean": float(np.mean(scores).round(4)),
        "cv_std": float(np.std(scores).round(4)),
        "cv_scores": scores.tolist(),
    }
    logger.info(
        f"✅ CV ({cv}-fold) | Mean {scoring}: {result['cv_mean']:.4f} "
        f"± {result['cv_std']:.4f}"
    )
    return result


# ─────────────────────────────────────────────
# 5. TRAINING SINGLE MODEL
# ─────────────────────────────────────────────

def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: pd.Series,
    model_name: str = "Model",
) -> Any:
    """
    Fit a single sklearn-compatible model on training data.

    Args:
        model: Un-fitted sklearn estimator.
        X_train: Scaled training feature matrix.
        y_train: Training target labels.
        model_name: Display name for logging.

    Returns:
        Fitted model.
    """
    logger.info(f"🔄 Training {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"✅ {model_name} training complete.")
    return model


# ─────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

def tune_model(
    model: Any,
    param_grid: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: pd.Series,
    model_name: str = "Model",
    cv: int = 5,
    scoring: str = "f1_weighted",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune model hyperparameters using GridSearchCV with stratified k-fold.

    Args:
        model: Base sklearn estimator.
        param_grid: Hyperparameter search space.
        X_train: Training feature matrix.
        y_train: Training labels.
        model_name: Name for logging.
        cv: Number of cross-validation folds.
        scoring: Optimisation metric.
        random_state: Random seed.
        n_jobs: Parallel jobs (-1 = all cores).

    Returns:
        Tuple of (best_fitted_model, best_params_dict).
    """
    logger.info(f"🔍 Tuning {model_name} with GridSearchCV ({cv}-fold)...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    logger.info(
        f"✅ {model_name} best params: {best_params} | "
        f"Best CV {scoring}: {best_score:.4f}"
    )
    return best_model, {"best_params": best_params, "best_cv_score": best_score}


# ─────────────────────────────────────────────
# 7. TRAIN ALL MODELS
# ─────────────────────────────────────────────

def train_all_models(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_all: np.ndarray,
    y_all: pd.Series,
    use_smote: bool = True,
    use_tuning: bool = True,
    random_state: int = RANDOM_STATE,
    cv_folds: int = CV_FOLDS,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all models with optional SMOTE and hyperparameter tuning.

    Args:
        X_train: Scaled training features.
        y_train: Training labels (string).
        X_all: All features (for cross-validation).
        y_all: All labels (for cross-validation).
        use_smote: Whether to apply SMOTE oversampling.
        use_tuning: Whether to run GridSearchCV tuning.
        random_state: Random seed.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dictionary of model name → dict with keys:
            'model', 'cv_results', 'tuning_info'
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting Model Training Pipeline")
    logger.info("=" * 60)

    # Optionally apply SMOTE to training data
    if use_smote:
        X_tr, y_tr = apply_smote(X_train, y_train, random_state)
    else:
        X_tr, y_tr = X_train, y_train

    base_models = get_base_models(random_state)
    param_grids = get_hyperparameter_grids() if use_tuning else {}

    results: Dict[str, Dict[str, Any]] = {}

    for name, base_model in base_models.items():
        logger.info(f"\n{'─'*50}")
        logger.info(f"📌 Processing: {name}")

        tuning_info: Dict[str, Any] = {}

        if use_tuning and name in param_grids:
            model, tuning_info = tune_model(
                base_model, param_grids[name], X_tr, y_tr,
                model_name=name, cv=5, random_state=random_state,
            )
        else:
            model = train_model(base_model, X_tr, y_tr, name)

        # Cross-validation on full (unsmoted) training data
        cv_results = cross_validate_model(
            model, X_all, y_all, cv=cv_folds,
            scoring="f1_weighted", random_state=random_state,
        )

        results[name] = {
            "model": model,
            "cv_results": cv_results,
            "tuning_info": tuning_info,
        }

    logger.info("\n✅ All models trained successfully.")
    return results


# ─────────────────────────────────────────────
# 8. BEST MODEL SELECTION
# ─────────────────────────────────────────────

def select_best_model(
    model_results: Dict[str, Dict[str, Any]],
    eval_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    metric: str = "f1_weighted",
) -> Tuple[str, Any]:
    """
    Automatically select the best performing model.

    Selection is based on the evaluation metric from the test set
    (if eval_metrics provided) or CV mean score (fallback).

    Args:
        model_results: Dictionary from train_all_models().
        eval_metrics: Optional dict of model_name → metric_dict from evaluation.
        metric: Metric name to compare.

    Returns:
        Tuple of (best_model_name, best_model_estimator).
    """
    scores: Dict[str, float] = {}

    for name, result in model_results.items():
        if eval_metrics and name in eval_metrics and metric in eval_metrics[name]:
            scores[name] = eval_metrics[name][metric]
        else:
            scores[name] = result["cv_results"]["cv_mean"]

    best_name = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_model = model_results[best_name]["model"]

    logger.info(f"\n🏆 Best Model: {best_name} | Score ({metric}): {scores[best_name]:.4f}")
    for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        marker = " ← BEST" if n == best_name else ""
        logger.info(f"   {n:30s}: {s:.4f}{marker}")

    return best_name, best_model


# ─────────────────────────────────────────────
# 9. MODEL PERSISTENCE
# ─────────────────────────────────────────────

def save_model(
    model: Any,
    model_name: str,
    save_dir: str = MODEL_DIR,
    scaler: Optional[Any] = None,
    label_encoder: Optional[Any] = None,
) -> str:
    """
    Save a trained model and optional preprocessors to disk using Joblib.

    Args:
        model: Fitted sklearn/xgboost model.
        model_name: File name key (e.g. 'random_forest').
        save_dir: Directory to save .pkl files.
        scaler: Optional fitted scaler to save alongside the model.
        label_encoder: Optional fitted LabelEncoder.

    Returns:
        Full file path of saved model.
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    filepath = os.path.join(save_dir, f"{safe_name}.pkl")

    payload = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "model_name": model_name,
    }
    joblib.dump(payload, filepath, compress=3)
    logger.info(f"✅ Model saved → {filepath}")
    return filepath


def load_model(filepath: str) -> Dict[str, Any]:
    """
    Load a previously saved model payload from disk.

    Args:
        filepath: Path to the .pkl model file.

    Returns:
        Dictionary with 'model', 'scaler', 'label_encoder', 'model_name'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    payload = joblib.load(filepath)
    logger.info(f"✅ Model loaded from {filepath}")
    return payload


def save_all_models(
    model_results: Dict[str, Dict[str, Any]],
    scaler: Optional[Any] = None,
    label_encoder: Optional[Any] = None,
    save_dir: str = MODEL_DIR,
) -> Dict[str, str]:
    """
    Save all trained models to the model directory.

    Args:
        model_results: Dictionary from train_all_models().
        scaler: Fitted feature scaler.
        label_encoder: Fitted LabelEncoder.
        save_dir: Output directory.

    Returns:
        Dictionary mapping model name → saved file path.
    """
    saved_paths: Dict[str, str] = {}
    for name, result in model_results.items():
        path = save_model(
            result["model"], name, save_dir, scaler, label_encoder
        )
        saved_paths[name] = path
    logger.info(f"✅ All {len(saved_paths)} models saved to '{save_dir}'.")
    return saved_paths
