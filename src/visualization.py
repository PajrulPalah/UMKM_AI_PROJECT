# ============================================================
# visualization.py
# UMKM AI Business Resilience Prediction System
# Module: Visualization & Plotting
# Author: AI Engineer
# Version: 1.0.0
# ============================================================

"""
Visualization module for UMKM Business Resilience Prediction System.

Generates:
    - Correlation Heatmap
    - Distribution Plots (Histograms, KDE)
    - Boxplots (per construct)
    - Class Distribution Bar Chart
    - Accuracy / F1 Comparison Chart
    - ROC Curve (multi-class OvR)
    - Confusion Matrix Heatmap
    - Feature Importance Bar Chart
    - Learning Curve Plot
    - Calibration Curve Plot
    - Probability Distribution Plot
    - SHAP Summary Plot (wrapper)

All figures are saved to the figures/ directory automatically.
"""

import os
import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

matplotlib.use("Agg")  # Non-interactive backend (safe for scripts & notebooks)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# GLOBAL STYLE CONFIGURATION
# ─────────────────────────────────────────────

FIGURES_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "figures"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Premium color palette
PALETTE_PRIMARY: List[str] = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
PALETTE_CLASSES: Dict[str, str] = {
    "Low": "#C73E1D",
    "Medium": "#F18F01",
    "High": "#2E86AB",
}
CMAP_HEATMAP = LinearSegmentedColormap.from_list(
    "umkm_heatmap", ["#1a1a2e", "#16213e", "#2E86AB", "#A23B72", "#F18F01"]
)

# Global plot styling
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "axes.titlecolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.alpha": 0.5,
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#e6edf3",
})


def _save_fig(fig: plt.Figure, filename: str, dpi: int = 150) -> str:
    """Save a figure to the figures directory and close it."""
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"✅ Figure saved → {filepath}")
    return filepath


# ─────────────────────────────────────────────
# 1. EDA PLOTS
# ─────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Feature Correlation Heatmap",
    filename: str = "01_correlation_heatmap.png",
) -> str:
    """
    Generate an annotated correlation heatmap for selected numeric columns.

    Args:
        df: DataFrame with numeric features.
        columns: Subset of columns to correlate; defaults to all numeric.
        title: Plot title.
        filename: Output file name.

    Returns:
        Absolute file path of saved figure.
    """
    if columns:
        data = df[columns].select_dtypes(include=[np.number])
    else:
        data = df.select_dtypes(include=[np.number])

    corr = data.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return _save_fig(fig, filename)


def plot_distributions(
    df: pd.DataFrame,
    columns: List[str],
    title: str = "Feature Distributions",
    filename: str = "02_distributions.png",
    ncols: int = 3,
) -> str:
    """
    Plot histogram + KDE for a list of numeric columns.

    Args:
        df: Source DataFrame.
        columns: Columns to plot.
        title: Main figure title.
        filename: Output file name.
        ncols: Number of subplot columns.

    Returns:
        File path of saved figure.
    """
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).flatten()

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(columns)))

    for idx, (col, color) in enumerate(zip(columns, colors)):
        ax = axes[idx]
        if col in df.columns:
            data = df[col].dropna()
            ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
            ax2 = ax.twinx()
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                ax2.plot(x_range, kde(x_range), color="white", linewidth=1.5, alpha=0.85)
                ax2.set_yticks([])
            except Exception:
                pass
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            mean_val = data.mean()
            ax.axvline(mean_val, color="#F18F01", linestyle="--", linewidth=1.2, label=f"μ={mean_val:.2f}")
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save_fig(fig, filename)


def plot_boxplots(
    df: pd.DataFrame,
    columns: List[str],
    hue_col: Optional[str] = "BusinessResilienceCategory",
    title: str = "Construct Score Distributions by Resilience Category",
    filename: str = "03_boxplots.png",
) -> str:
    """
    Generate side-by-side boxplots, optionally grouped by category.

    Args:
        df: Source DataFrame.
        columns: Construct score columns to plot.
        hue_col: Optional grouping column for color differentiation.
        title: Figure title.
        filename: Output file name.

    Returns:
        File path of saved figure.
    """
    ncols = 3
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).flatten()

    palette = {"Low": "#C73E1D", "Medium": "#F18F01", "High": "#2E86AB"}

    for idx, col in enumerate(columns):
        ax = axes[idx]
        if col in df.columns:
            if hue_col and hue_col in df.columns:
                cats = [c for c in CLASS_ORDER if c in df[hue_col].unique()]
                groups = [df[df[hue_col] == c][col].dropna().values for c in cats]
                bp = ax.boxplot(
                    groups, patch_artist=True, notch=False, widths=0.5,
                    medianprops=dict(color="white", linewidth=2),
                )
                for patch, cat in zip(bp["boxes"], cats):
                    patch.set_facecolor(palette.get(cat, "#8b949e"))
                    patch.set_alpha(0.85)
                ax.set_xticklabels(cats)
            else:
                ax.boxplot(df[col].dropna().values, patch_artist=True)

            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.set_ylabel("Score (1–5)")
            ax.yaxis.grid(True, alpha=0.3)

    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, filename)


# Refer to CLASS_ORDER from evaluation
CLASS_ORDER: List[str] = ["Low", "Medium", "High"]


def plot_class_distribution(
    y: pd.Series,
    title: str = "Business Resilience Category Distribution",
    filename: str = "04_class_distribution.png",
) -> str:
    """
    Plot bar chart of target class frequencies.

    Args:
        y: Target label series.
        title: Figure title.
        filename: Output file name.

    Returns:
        File path of saved figure.
    """
    counts = y.value_counts()
    ordered = [c for c in CLASS_ORDER if c in counts.index]
    values = [counts[c] for c in ordered]
    colors = [PALETTE_CLASSES.get(c, "#8b949e") for c in ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(ordered, values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val}\n({val/sum(values)*100:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Resilience Category", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_ylim(0, max(values) * 1.25)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, filename)


# ─────────────────────────────────────────────
# 2. MODEL EVALUATION PLOTS
# ─────────────────────────────────────────────

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    filename: str = "05_model_comparison.png",
) -> str:
    """
    Generate a grouped bar chart comparing model metrics.

    Args:
        comparison_df: Output from evaluation.build_comparison_table().
        filename: Output file name.

    Returns:
        File path of saved figure.
    """
    metrics = ["Accuracy", "F1 (Weighted)", "ROC AUC", "CV Mean (F1)"]
    x = np.arange(len(comparison_df))
    n_metrics = len(metrics)
    width = 0.18
    colors = ["#2E86AB", "#F18F01", "#A23B72", "#3BA55D"]

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = comparison_df[metric].fillna(0).values
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=metric, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90,
            )

    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Model"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, filename)


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    model_name: str = "Model",
    filename: Optional[str] = None,
) -> str:
    """
    Generate a styled confusion matrix heatmap.

    Args:
        cm_df: Confusion matrix DataFrame (rows=actual, cols=predicted).
        model_name: Model name for title.
        filename: Output file name.

    Returns:
        File path of saved figure.
    """
    if filename is None:
        safe = model_name.lower().replace(" ", "_")
        filename = f"06_confusion_matrix_{safe}.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_df, annot=True, fmt="d", cmap="Blues",
        linewidths=0.8, linecolor="#30363d",
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return _save_fig(fig, filename)


def plot_roc_curves(
    eval_results: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    classes: Optional[List[str]] = None,
    filename: str = "07_roc_curves.png",
) -> str:
    """
    Plot multi-class ROC curves (One-vs-Rest) for all models.

    Args:
        eval_results: Output from evaluation.evaluate_all_models().
        y_test: True test labels.
        classes: Ordered class names.
        filename: Output file name.

    Returns:
        File path of saved figure.
    """
    if classes is None:
        classes = CLASS_ORDER

    y_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    model_colors = plt.cm.tab10(np.linspace(0, 1, len(eval_results)))

    for cls_idx, (ax, cls_name) in enumerate(zip(axes, classes)):
        for (model_name, result), color in zip(eval_results.items(), model_colors):
            y_prob = result.get("y_prob")
            if y_prob is None or y_prob.ndim < 2:
                continue
            if y_prob.shape[1] != n_classes:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_prob[:, cls_idx])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{model_name} (AUC={roc_auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.5, label="Random")
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color="white")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_title(f"ROC – Class: {cls_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
        ax.yaxis.grid(True, alpha=0.3)

    fig.suptitle("Multi-Class ROC Curves (One-vs-Rest)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, filename)


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "Model",
    filename: Optional[str] = None,
    top_n: int = 20,
) -> str:
    """
    Plot feature importances from tree-based models.

    Works with Random Forest, XGBoost (feature_importances_ attribute).
    For Logistic Regression, uses absolute coefficient values.

    Args:
        model: Fitted sklearn estimator.
        feature_names: List of feature names.
        model_name: Model name for title.
        filename: Output file name.
        top_n: Maximum features to display.

    Returns:
        File path of saved figure.
    """
    if filename is None:
        safe = model_name.lower().replace(" ", "_")
        filename = f"08_feature_importance_{safe}.png"

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)

    if importances is None:
        logger.warning(f"⚠️ {model_name} does not expose feature importances.")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Feature importances not available.",
                ha="center", va="center", transform=ax.transAxes)
        return _save_fig(fig, filename)

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n)

    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(feat_df)))

    fig, ax = plt.subplots(figsize=(10, max(5, len(feat_df) * 0.45)))
    bars = ax.barh(
        feat_df["Feature"][::-1], feat_df["Importance"][::-1],
        color=colors[::-1], edgecolor="white", linewidth=0.4,
    )
    for bar, val in zip(bars, feat_df["Importance"][::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_title(f"Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.xaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, filename)


# ─────────────────────────────────────────────
# 3. ADVANCED PLOTS
# ─────────────────────────────────────────────

def plot_learning_curve(
    lc_data: Dict[str, np.ndarray],
    model_name: str = "Best Model",
    filename: str = "09_learning_curve.png",
) -> str:
    """
    Plot training vs. validation learning curves.

    Args:
        lc_data: Output from evaluation.compute_learning_curve().
        model_name: Model name for title.
        filename: Output file name.

    Returns:
        File path of saved figure.
    """
    sizes = lc_data["train_sizes"]
    train_mean = lc_data["train_scores_mean"]
    train_std = lc_data["train_scores_std"]
    val_mean = lc_data["val_scores_mean"]
    val_std = lc_data["val_scores_std"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, train_mean, "o-", color="#2E86AB", linewidth=2.5, label="Training Score", markersize=6)
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2E86AB")

    ax.plot(sizes, val_mean, "s-", color="#F18F01", linewidth=2.5, label="Validation Score", markersize=6)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#F18F01")

    ax.set_title(f"Learning Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("F1 Score (Weighted)")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _save_fig(fig, filename)


def plot_calibration_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
    model_name: str = "Best Model",
    classes: Optional[List[str]] = None,
    filename: str = "10_calibration_curve.png",
) -> str:
    """
    Plot probability calibration curves.

    Calibration curves show how well predicted probabilities match
    actual frequencies (diagonal line = perfect calibration).

    Args:
        model: Fitted estimator with predict_proba().
        X_test: Test feature matrix.
        y_test: True labels.
        model_name: Model name for title.
        classes: Class names.
        filename: Output file name.

    Returns:
        File path of saved figure.
    """
    if classes is None:
        classes = CLASS_ORDER

    if not hasattr(model, "predict_proba"):
        logger.warning(f"⚠️ {model_name} has no predict_proba. Skipping calibration.")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "Calibration not available.", ha="center", va="center", transform=ax.transAxes)
        return _save_fig(fig, filename)

    y_prob = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=classes)

    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 5), sharey=True)
    if len(classes) == 1:
        axes = [axes]

    colors = ["#2E86AB", "#F18F01", "#A23B72"]

    for cls_idx, (ax, cls_name, color) in enumerate(zip(axes, classes, colors)):
        if cls_idx >= y_prob.shape[1]:
            continue
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_bin[:, cls_idx], y_prob[:, cls_idx], n_bins=8
            )
            ax.plot([0, 1], [0, 1], "w--", linewidth=1, label="Perfect Calibration")
            ax.plot(mean_predicted_value, fraction_of_positives, "o-",
                    color=color, linewidth=2, markersize=7, label=f"{cls_name}")
            ax.set_title(f"Class: {cls_name}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.legend(fontsize=9)
            ax.yaxis.grid(True, alpha=0.3)
        except Exception as e:
            logger.warning(f"⚠️ Calibration for {cls_name}: {e}")

    fig.suptitle(f"Calibration Curves — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, filename)


def plot_probability_distribution(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
    model_name: str = "Best Model",
    classes: Optional[List[str]] = None,
    filename: str = "11_probability_distribution.png",
) -> str:
    """
    Plot predicted probability distributions per class.

    Args:
        model: Fitted model with predict_proba().
        X_test: Test features.
        y_test: True labels.
        model_name: Model name.
        classes: Class labels.
        filename: Output file name.

    Returns:
        File path of saved figure.
    """
    if classes is None:
        classes = CLASS_ORDER

    if not hasattr(model, "predict_proba"):
        logger.warning("⚠️ Model has no predict_proba. Skipping.")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Probability distribution not available.",
                ha="center", va="center", transform=ax.transAxes)
        return _save_fig(fig, filename)

    y_prob = model.predict_proba(X_test)
    colors = ["#C73E1D", "#F18F01", "#2E86AB"]

    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 5))
    if len(classes) == 1:
        axes = [axes]

    for cls_idx, (ax, cls_name, color) in enumerate(zip(axes, classes, colors)):
        if cls_idx >= y_prob.shape[1]:
            continue
        probs = y_prob[:, cls_idx]
        ax.hist(probs, bins=20, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(probs.mean(), color="white", linestyle="--", linewidth=1.5, label=f"μ={probs.mean():.3f}")
        ax.set_title(f"P({cls_name})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)

    fig.suptitle(f"Predicted Probability Distributions — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, filename)


# ─────────────────────────────────────────────
# 4. MASTER DASHBOARD
# ─────────────────────────────────────────────

def generate_all_visualizations(
    df: pd.DataFrame,
    eval_results: Dict[str, Dict[str, Any]],
    comparison_df: pd.DataFrame,
    best_model: Any,
    best_model_name: str,
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: List[str],
    lc_data: Optional[Dict[str, np.ndarray]] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Generate and save the complete visualization suite.

    Args:
        df: Processed DataFrame.
        eval_results: Output from evaluate_all_models().
        comparison_df: Output from build_comparison_table().
        best_model: Best fitted estimator.
        best_model_name: Name of best model.
        X_test: Test features.
        y_test: Test labels.
        feature_names: Feature column names.
        lc_data: Optional learning curve data.
        classes: Class labels.

    Returns:
        Dictionary mapping plot name → file path.
    """
    if classes is None:
        classes = CLASS_ORDER

    saved = {}

    # Construct score columns
    score_cols = [
        "DigitalCapabilityScore", "InnovationCapabilityScore",
        "EntrepreneurialOrientationScore", "OrganizationalAgilityScore",
        "ResourceAccessScore", "EnvironmentalDynamismScore", "BusinessResilienceScore"
    ]
    available_score_cols = [c for c in score_cols if c in df.columns]

    logger.info("\n📊 Generating Visualization Suite...")

    saved["correlation_heatmap"] = plot_correlation_heatmap(
        df, columns=available_score_cols
    )
    saved["distributions"] = plot_distributions(
        df, columns=available_score_cols
    )
    saved["boxplots"] = plot_boxplots(
        df, columns=available_score_cols
    )
    saved["class_distribution"] = plot_class_distribution(
        df["BusinessResilienceCategory"] if "BusinessResilienceCategory" in df.columns else y_test
    )
    saved["model_comparison"] = plot_model_comparison(comparison_df)

    for model_name, result in eval_results.items():
        cm_df = result["confusion_matrix"]
        saved[f"confusion_matrix_{model_name}"] = plot_confusion_matrix(cm_df, model_name)

    saved["roc_curves"] = plot_roc_curves(eval_results, y_test, classes)
    saved["feature_importance"] = plot_feature_importance(
        best_model, feature_names, best_model_name
    )

    if lc_data is not None:
        saved["learning_curve"] = plot_learning_curve(lc_data, best_model_name)

    saved["calibration_curve"] = plot_calibration_curve(
        best_model, X_test, y_test, best_model_name, classes
    )
    saved["probability_distribution"] = plot_probability_distribution(
        best_model, X_test, y_test, best_model_name, classes
    )

    logger.info(f"✅ {len(saved)} figures generated.")
    return saved
