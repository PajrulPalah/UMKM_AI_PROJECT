# ============================================================
# evaluation.py
# UMKM AI Business Resilience Prediction System
# Module: Model Evaluation & Performance Metrics
# Author: AI Engineer
# Version: 1.0.0
# ============================================================

"""
Evaluation module for UMKM Business Resilience Prediction System.

Handles:
    - Accuracy, Precision, Recall, F1-Score
    - ROC AUC (multi-class OvR)
    - Confusion Matrix
    - Classification Report
    - Cross-Validation Summary
    - Model Comparison Table
    - Best Model Identification
    - Export evaluation results
"""

import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelBinarizer, label_binarize

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CLASS ORDER FOR CONSISTENT METRICS
# ─────────────────────────────────────────────

CLASS_ORDER: List[str] = ["Low", "Medium", "High"]


# ─────────────────────────────────────────────
# 1. SINGLE MODEL METRICS
# ─────────────────────────────────────────────

def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute classification performance metrics for a single model.

    Metrics computed:
        - Accuracy
        - Precision (weighted)
        - Recall (weighted)
        - F1-Score (weighted)
        - ROC AUC (One-vs-Rest, if probabilities provided)

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted class probabilities (shape: n_samples × n_classes).
        classes: Ordered list of class names for AUC binarization.
        average: Averaging strategy for precision/recall/F1.

    Returns:
        Dictionary of metric name → metric value.
    """
    metrics: Dict[str, float] = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=average, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average=average, zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average=average, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }

    # ROC AUC (requires probability estimates)
    if y_prob is not None and classes is not None:
        try:
            # Binarize for multi-class OvR
            y_bin = label_binarize(y_true, classes=classes)
            # Reorder probability columns to match class order
            auc = roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")
            metrics["roc_auc"] = round(auc, 4)
        except Exception as e:
            logger.warning(f"⚠️ ROC AUC computation failed: {e}")
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def compute_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute and return a labelled confusion matrix as a DataFrame.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered class labels for matrix rows/cols.

    Returns:
        Confusion matrix as a DataFrame.
    """
    if labels is None:
        labels = sorted(y_true.unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return cm_df


def get_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> str:
    """
    Generate a full sklearn classification report as a formatted string.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Class labels.

    Returns:
        Formatted classification report string.
    """
    return classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    )


# ─────────────────────────────────────────────
# 2. EVALUATE ALL MODELS
# ─────────────────────────────────────────────

def evaluate_all_models(
    model_results: Dict[str, Dict[str, Any]],
    X_test: np.ndarray,
    y_test: pd.Series,
    classes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all trained models on the held-out test set.

    For each model computes:
        - Standard metrics (accuracy, precision, recall, F1, ROC AUC)
        - Confusion matrix
        - Full classification report
        - Predictions and probabilities

    Args:
        model_results: Dictionary from training.train_all_models().
        X_test: Scaled test feature matrix.
        y_test: Test ground-truth labels.
        classes: Ordered class names for AUC computation.

    Returns:
        Dictionary of model_name → evaluation results dict.
    """
    if classes is None:
        classes = CLASS_ORDER

    eval_results: Dict[str, Dict[str, Any]] = {}

    for name, result in model_results.items():
        model = result["model"]
        y_pred = model.predict(X_test)

        # Probabilities (not all models guarantee this)
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                pass

        metrics = compute_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob, classes=classes)
        cm_df = compute_confusion_matrix(y_true=y_test, y_pred=y_pred, labels=classes)
        report = get_classification_report(y_true=y_test, y_pred=y_pred, labels=classes)
        cv_results = result.get("cv_results", {})

        eval_results[name] = {
            "metrics": metrics,
            "confusion_matrix": cm_df,
            "classification_report": report,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "cv_mean": cv_results.get("cv_mean", float("nan")),
            "cv_std": cv_results.get("cv_std", float("nan")),
        }

        logger.info(
            f"✅ {name:30s} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1(w): {metrics['f1_weighted']:.4f} | "
            f"AUC: {metrics.get('roc_auc', float('nan')):.4f}"
        )

    return eval_results


# ─────────────────────────────────────────────
# 3. COMPARISON TABLE
# ─────────────────────────────────────────────

def build_comparison_table(
    eval_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a side-by-side model comparison DataFrame.

    Columns: Model, Accuracy, Precision, Recall, F1 (Weighted),
             F1 (Macro), ROC AUC, CV Mean (F1), CV Std.

    The best model per column is automatically identified.

    Args:
        eval_results: Output from evaluate_all_models().

    Returns:
        Comparison DataFrame sorted by F1 (Weighted) descending.
    """
    rows = []
    for name, result in eval_results.items():
        m = result["metrics"]
        row = {
            "Model": name,
            "Accuracy": m.get("accuracy", np.nan),
            "Precision": m.get("precision", np.nan),
            "Recall": m.get("recall", np.nan),
            "F1 (Weighted)": m.get("f1_weighted", np.nan),
            "F1 (Macro)": m.get("f1_macro", np.nan),
            "ROC AUC": m.get("roc_auc", np.nan),
            "CV Mean (F1)": result.get("cv_mean", np.nan),
            "CV Std": result.get("cv_std", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("F1 (Weighted)", ascending=False)
    df = df.reset_index(drop=True)

    # Add rank column
    df.insert(0, "Rank", range(1, len(df) + 1))

    best_model_name = df.iloc[0]["Model"]
    logger.info(f"\n🏆 Best Model by F1 (Weighted): {best_model_name}")
    logger.info(f"\n{df.to_string(index=False)}")

    return df


# ─────────────────────────────────────────────
# 4. EXPORT EVALUATION RESULTS
# ─────────────────────────────────────────────

def export_evaluation_report(
    comparison_df: pd.DataFrame,
    output_path: str = "evaluation_report.csv",
) -> None:
    """
    Export the model comparison table to a CSV file.

    Args:
        comparison_df: DataFrame from build_comparison_table().
        output_path: Output file path.
    """
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"✅ Evaluation report exported → {output_path}")


def get_best_model_name(comparison_df: pd.DataFrame) -> str:
    """
    Return the name of the best-performing model from the comparison table.

    Args:
        comparison_df: Output from build_comparison_table().

    Returns:
        Model name string.
    """
    return str(comparison_df.iloc[0]["Model"])


# ─────────────────────────────────────────────
# 5. LEARNING CURVE ANALYSIS
# ─────────────────────────────────────────────

def compute_learning_curve(
    model: Any,
    X: np.ndarray,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "f1_weighted",
    train_sizes: Optional[List[float]] = None,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Compute learning curve data points for the given model.

    Learning curves show how model performance evolves as training
    set size increases — useful for diagnosing bias vs. variance.

    Args:
        model: Sklearn-compatible estimator.
        X: Feature matrix (full dataset).
        y: Target labels (full dataset).
        cv: Cross-validation folds.
        scoring: Performance metric.
        train_sizes: Fractions of training data to evaluate.
        random_state: Random seed.

    Returns:
        Dictionary with 'train_sizes', 'train_scores', 'val_scores'.
    """
    from sklearn.model_selection import learning_curve, StratifiedKFold

    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=random_state,
    )
    logger.info(f"✅ Learning curve computed over {len(sizes)} training sizes.")
    return {
        "train_sizes": sizes,
        "train_scores_mean": train_scores.mean(axis=1),
        "train_scores_std": train_scores.std(axis=1),
        "val_scores_mean": val_scores.mean(axis=1),
        "val_scores_std": val_scores.std(axis=1),
    }


# ─────────────────────────────────────────────
# 6. PERMUTATION IMPORTANCE
# ─────────────────────────────────────────────

def compute_permutation_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: List[str],
    n_repeats: int = 20,
    scoring: str = "f1_weighted",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation feature importance on the test set.

    Unlike SHAP, permutation importance directly measures how much
    model performance drops when a feature's values are randomly shuffled.

    Args:
        model: Fitted sklearn-compatible estimator.
        X_test: Test feature matrix.
        y_test: Test labels.
        feature_names: Names corresponding to feature columns.
        n_repeats: Number of permutation repeats.
        scoring: Metric to evaluate importance.
        random_state: Random seed.

    Returns:
        DataFrame with columns ['Feature', 'Importance_Mean', 'Importance_Std']
        sorted by importance descending.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False).reset_index(drop=True)

    logger.info(f"✅ Permutation importance computed. Top feature: {perm_df.iloc[0]['Feature']}")
    return perm_df
