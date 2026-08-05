# ============================================================
# explainability.py
# UMKM AI Business Resilience Prediction System
# Module: Explainable AI with SHAP
# Author: AI Engineer
# Version: 1.0.0
# ============================================================

"""
Explainability module for UMKM Business Resilience Prediction System.

Uses SHAP (SHapley Additive exPlanations) to explain model predictions.

Generates:
    - SHAP Summary Plot (beeswarm)
    - SHAP Bar Plot (mean absolute)
    - SHAP Waterfall Plot (single prediction)
    - SHAP Dependence Plots
    - Natural language interpretation
    - Feature importance ranking

SHAP values quantify each feature's marginal contribution to a
specific prediction relative to the expected model output.
"""

import os
import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "figures"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PLOT STYLING
# ─────────────────────────────────────────────

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
})


def _save_fig(fig: plt.Figure, filename: str, dpi: int = 150) -> str:
    """Save a figure to the figures directory."""
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"✅ SHAP figure saved → {filepath}")
    return filepath


# ─────────────────────────────────────────────
# 1. SHAP EXPLAINER FACTORY
# ─────────────────────────────────────────────

def create_explainer(
    model: Any,
    X_background: np.ndarray,
    model_name: str = "Model",
    feature_names: Optional[List[str]] = None,
) -> Any:
    """
    Create the appropriate SHAP Explainer for the given model type.

    Explainer selection logic:
        - Tree-based (RF, XGBoost, DT) → TreeExplainer (exact, fast)
        - Linear (Logistic Regression)  → LinearExplainer (exact)
        - Others                         → KernelExplainer (approximation)

    Args:
        model: Fitted sklearn/xgboost estimator.
        X_background: Background dataset for KernelExplainer
                      (or training data for others).
        model_name: Model name for logging.
        feature_names: Optional feature names.

    Returns:
        Fitted SHAP Explainer object.
    """
    try:
        import shap

        model_type = type(model).__name__
        logger.info(f"🔄 Creating SHAP explainer for {model_name} ({model_type})...")

        tree_types = [
            "RandomForestClassifier", "DecisionTreeClassifier",
            "XGBClassifier", "GradientBoostingClassifier",
            "ExtraTreesClassifier", "LGBMClassifier",
        ]

        if model_type in tree_types:
            explainer = shap.TreeExplainer(model, data=X_background)
            logger.info(f"✅ TreeExplainer created for {model_name}.")
        elif "LogisticRegression" in model_type or "Linear" in model_type:
            explainer = shap.LinearExplainer(
                model, X_background,
                feature_perturbation="interventional",
            )
            logger.info(f"✅ LinearExplainer created for {model_name}.")
        else:
            # KernelExplainer — model-agnostic but slower
            background = shap.kmeans(X_background, k=min(20, X_background.shape[0]))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            logger.info(f"✅ KernelExplainer created for {model_name} (may be slow).")

        return explainer

    except ImportError:
        raise ImportError("SHAP library is required. Install with: pip install shap")


# ─────────────────────────────────────────────
# 2. COMPUTE SHAP VALUES
# ─────────────────────────────────────────────

def compute_shap_values(
    explainer: Any,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 100,
) -> Tuple[Any, pd.DataFrame]:
    """
    Compute SHAP values for the provided dataset.

    For multi-class models, SHAP values have shape:
        (n_samples, n_features, n_classes)

    This function computes SHAP values and wraps the mean-absolute
    importance into a ranked DataFrame.

    Args:
        explainer: Fitted SHAP Explainer.
        X: Feature matrix to explain (scaled).
        feature_names: Column names for features.
        max_samples: Maximum samples to explain (for performance).

    Returns:
        Tuple of (shap_values_object, importance_dataframe).
    """
    import shap

    # Limit to max_samples for performance
    n = min(max_samples, X.shape[0])
    X_subset = X[:n]

    logger.info(f"🔄 Computing SHAP values for {n} samples...")
    shap_values = explainer(X_subset)
    logger.info(f"✅ SHAP values computed. Shape: {shap_values.values.shape}")

    # Compute mean absolute importance
    if shap_values.values.ndim == 3:
        # Multi-class: average over all classes
        mean_abs = np.abs(shap_values.values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values.values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_SHAP": mean_abs.round(5),
    }).sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)

    importance_df["Rank"] = range(1, len(importance_df) + 1)
    importance_df["Pct_Contribution"] = (
        importance_df["Mean_SHAP"] / importance_df["Mean_SHAP"].sum() * 100
    ).round(2)

    return shap_values, importance_df


# ─────────────────────────────────────────────
# 3. SHAP PLOTS
# ─────────────────────────────────────────────

def plot_shap_summary(
    shap_values: Any,
    X: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    model_name: str = "Model",
    filename: str = "12_shap_summary.png",
    plot_type: str = "beeswarm",
) -> str:
    """
    Generate SHAP summary (beeswarm) plot showing feature impact across samples.

    The beeswarm plot shows:
        - X-axis: SHAP value (impact on model output)
        - Y-axis: Feature (ranked by importance)
        - Color: Feature value (blue=low, red=high)

    Args:
        shap_values: SHAP Explanation object.
        X: Feature matrix (must match n_samples of shap_values).
        feature_names: Feature column names.
        class_names: Class labels for multi-class output.
        model_name: Model name for title.
        filename: Output file name.
        plot_type: 'beeswarm' or 'bar'.

    Returns:
        File path of saved figure.
    """
    import shap

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 7))

    n = X.shape[0] if hasattr(shap_values, "values") else len(X)
    X_subset = X[:n]

    try:
        if shap_values.values.ndim == 3:
            # Multi-class: plot for each class or use mean
            # Use class index 2 (High Resilience) for primary summary
            sv_class = shap_values[:, :, -1]  # Last class (High)
            if class_names:
                title = f"SHAP Summary — {model_name} (Class: {class_names[-1]})"
            else:
                title = f"SHAP Summary — {model_name}"
        else:
            sv_class = shap_values
            title = f"SHAP Summary — {model_name}"

        if plot_type == "bar":
            shap.plots.bar(sv_class, max_display=15, show=False, ax=ax)
        else:
            shap.plots.beeswarm(sv_class, max_display=15, show=False)
            ax = plt.gca()

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    except Exception as e:
        logger.warning(f"⚠️ SHAP beeswarm fallback: {e}")
        shap.summary_plot(
            shap_values.values if hasattr(shap_values, "values") else shap_values,
            X_subset,
            feature_names=feature_names,
            show=False,
            max_display=15,
            plot_type="bar",
        )
        ax = plt.gca()
        ax.set_title(f"SHAP Feature Importance — {model_name}", fontsize=13, fontweight="bold")

    fig = plt.gcf()
    fig.set_facecolor("#0d1117")
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close("all")
    logger.info(f"✅ SHAP summary plot saved → {filepath}")
    return filepath


def plot_shap_bar(
    importance_df: pd.DataFrame,
    model_name: str = "Model",
    filename: str = "13_shap_bar_importance.png",
    top_n: int = 15,
) -> str:
    """
    Generate a styled horizontal bar chart of SHAP feature importances.

    Args:
        importance_df: DataFrame from compute_shap_values().
        model_name: Model name for title.
        filename: Output file name.
        top_n: Number of top features to display.

    Returns:
        File path of saved figure.
    """
    top_df = importance_df.head(top_n)
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_df)))

    fig, ax = plt.subplots(figsize=(10, max(5, len(top_df) * 0.5 + 1)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bars = ax.barh(
        top_df["Feature"][::-1].values,
        top_df["Mean_SHAP"][::-1].values,
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
        height=0.65,
    )

    for bar, pct in zip(bars, top_df["Pct_Contribution"][::-1].values):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", fontsize=9, color="#e6edf3",
        )

    ax.set_title(
        f"SHAP Feature Importance — {model_name}",
        fontsize=14, fontweight="bold", color="#e6edf3", pad=12,
    )
    ax.set_xlabel("Mean |SHAP Value|", color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.grid(True, alpha=0.3, color="#21262d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save_fig(fig, filename)


def plot_shap_waterfall(
    shap_values: Any,
    sample_idx: int = 0,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    model_name: str = "Model",
    filename: str = "14_shap_waterfall.png",
) -> str:
    """
    Generate a SHAP Waterfall plot for a single prediction.

    The waterfall plot breaks down the prediction for one specific
    instance, showing each feature's additive contribution from the
    baseline (expected output) to the final prediction.

    Args:
        shap_values: SHAP Explanation object.
        sample_idx: Index of the sample to explain.
        feature_names: Feature names.
        class_names: Class labels.
        model_name: Model name.
        filename: Output file name.

    Returns:
        File path of saved figure.
    """
    import shap

    plt.style.use("dark_background")

    try:
        if shap_values.values.ndim == 3:
            # Explain the "High" class (last class)
            sv_single = shap_values[sample_idx, :, -1]
            cls_label = class_names[-1] if class_names else "Class"
            title = f"SHAP Waterfall — {model_name} | Sample #{sample_idx} | Class: {cls_label}"
        else:
            sv_single = shap_values[sample_idx]
            title = f"SHAP Waterfall — {model_name} | Sample #{sample_idx}"

        shap.plots.waterfall(sv_single, max_display=15, show=False)
        ax = plt.gca()
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    except Exception as e:
        logger.warning(f"⚠️ SHAP waterfall fallback: {e}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"SHAP Waterfall unavailable: {e}",
                 ha="center", va="center", transform=plt.gca().transAxes)
        ax = plt.gca()

    fig = plt.gcf()
    fig.set_facecolor("#0d1117")
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close("all")
    logger.info(f"✅ SHAP waterfall plot saved → {filepath}")
    return filepath


def plot_shap_dependence(
    shap_values: Any,
    X: np.ndarray,
    feature_names: List[str],
    target_feature: str,
    interaction_feature: Optional[str] = None,
    model_name: str = "Model",
    filename: Optional[str] = None,
) -> str:
    """
    Generate a SHAP dependence plot for one feature.

    Dependence plots reveal how a single feature affects the
    model output, with optional interaction coloring.

    Args:
        shap_values: SHAP Explanation object.
        X: Feature matrix.
        feature_names: Feature names.
        target_feature: Feature to plot on X-axis.
        interaction_feature: Optional feature for color-coding interaction.
        model_name: Model name.
        filename: Output file name.

    Returns:
        File path of saved figure.
    """
    import shap

    if filename is None:
        safe = target_feature.lower().replace(" ", "_")
        filename = f"15_shap_dependence_{safe}.png"

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    n = X.shape[0]
    feature_idx = feature_names.index(target_feature) if target_feature in feature_names else 0

    try:
        if shap_values.values.ndim == 3:
            sv = shap_values.values[:n, :, -1]
        else:
            sv = shap_values.values[:n]

        scatter = ax.scatter(
            X[:n, feature_idx],
            sv[:, feature_idx],
            c=sv[:, feature_idx],
            cmap="coolwarm",
            alpha=0.8,
            s=60,
            edgecolors="white",
            linewidths=0.2,
        )
        plt.colorbar(scatter, ax=ax, label="SHAP Value")
        ax.axhline(0, color="#8b949e", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(target_feature, color="#e6edf3", fontsize=12)
        ax.set_ylabel("SHAP Value", color="#e6edf3", fontsize=12)
        ax.set_title(
            f"SHAP Dependence — {target_feature} | {model_name}",
            fontsize=13, fontweight="bold", color="#e6edf3",
        )
        ax.yaxis.grid(True, alpha=0.3, color="#21262d")

    except Exception as e:
        logger.warning(f"⚠️ SHAP dependence plot error: {e}")
        ax.text(0.5, 0.5, f"Dependence plot error: {e}",
                ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    return _save_fig(fig, filename)


# ─────────────────────────────────────────────
# 4. NATURAL LANGUAGE INTERPRETATION
# ─────────────────────────────────────────────

def generate_natural_language_interpretation(
    importance_df: pd.DataFrame,
    prediction: str,
    probabilities: Dict[str, float],
    feature_values: Dict[str, float],
    model_name: str = "Model",
) -> str:
    """
    Generate a natural language explanation of a prediction.

    This provides a human-readable summary suitable for academic
    documentation, business reports, or dashboard tooltips.

    Args:
        importance_df: SHAP importance DataFrame with Pct_Contribution column.
        prediction: Predicted class label.
        probabilities: Dict mapping class → probability.
        feature_values: Dict mapping feature → input value.
        model_name: Model name.

    Returns:
        Multi-paragraph natural language interpretation string.
    """
    top_features = importance_df.head(3)
    total_pct = importance_df["Pct_Contribution"].sum()

    interpretation = (
        f"📋 SHAP Prediction Explanation\n"
        f"{'='*60}\n\n"
        f"🤖 Model: {model_name}\n"
        f"🎯 Prediction: {prediction.upper()}\n\n"
        f"📊 Predicted Probabilities:\n"
    )

    for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        interpretation += f"   {cls:8s}: [{bar}] {prob*100:.1f}%\n"

    interpretation += f"\n🔑 Top Contributing Factors:\n"
    for _, row in top_features.iterrows():
        feat = row["Feature"]
        pct = row["Pct_Contribution"]
        val = feature_values.get(feat, "N/A")
        direction = "↑ positively" if val and isinstance(val, (int, float)) and val >= 3.0 else "↓ negatively"
        interpretation += (
            f"   • {feat} contributed {pct:.1f}% to this prediction.\n"
            f"     Input value: {val:.2f}/5.00 — influenced the model {direction}.\n"
        )

    interpretation += (
        f"\n💡 Interpretation:\n"
        f"   The model predicts a {prediction} level of Business Resilience with "
        f"{probabilities.get(prediction, 0)*100:.1f}% confidence.\n"
        f"   The top three features account for "
        f"{top_features['Pct_Contribution'].sum():.1f}% of the total predictive influence.\n"
    )

    if prediction == "High":
        interpretation += (
            f"\n✅ Business Insight: This UMKM demonstrates strong organizational capabilities\n"
            f"   and is well-positioned to withstand market disruptions.\n"
            f"   Key strengths: high construct scores in the top contributing dimensions.\n"
        )
    elif prediction == "Medium":
        interpretation += (
            f"\n⚠️ Business Insight: This UMKM shows moderate resilience.\n"
            f"   Strategic investment in the highest-impact capabilities could elevate\n"
            f"   resilience to the 'High' category.\n"
        )
    else:
        interpretation += (
            f"\n🔴 Business Insight: This UMKM faces significant resilience challenges.\n"
            f"   Targeted interventions in digital capability, agility, and resource access\n"
            f"   are recommended as priority development areas.\n"
        )

    return interpretation


# ─────────────────────────────────────────────
# 5. EXPORT SHAP IMPORTANCE
# ─────────────────────────────────────────────

def export_shap_importance(
    importance_df: pd.DataFrame,
    output_path: str = "shap_feature_importance.xlsx",
) -> None:
    """
    Export SHAP feature importance DataFrame to Excel.

    Args:
        importance_df: Output from compute_shap_values().
        output_path: Output file path (.xlsx).
    """
    importance_df.to_excel(output_path, index=False, sheet_name="SHAP_Importance")
    logger.info(f"✅ SHAP importance exported → {output_path}")


# ─────────────────────────────────────────────
# 6. FULL SHAP PIPELINE
# ─────────────────────────────────────────────

def run_explainability_pipeline(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    model_name: str = "Best Model",
    class_names: Optional[List[str]] = None,
    sample_idx: int = 0,
    export_dir: str = ".",
) -> Dict[str, Any]:
    """
    Execute the complete SHAP explainability pipeline.

    Steps:
        1. Create appropriate explainer
        2. Compute SHAP values
        3. Generate all SHAP plots
        4. Export importance table

    Args:
        model: Fitted estimator.
        X_train: Training feature matrix (background for explainer).
        X_test: Test features to explain.
        feature_names: Feature column names.
        model_name: Model display name.
        class_names: Class labels.
        sample_idx: Which test sample to use for waterfall plot.
        export_dir: Directory for .xlsx exports.

    Returns:
        Dictionary with all explainability artefacts.
    """
    if class_names is None:
        class_names = ["Low", "Medium", "High"]

    logger.info("=" * 60)
    logger.info("🧠 Starting SHAP Explainability Pipeline")
    logger.info("=" * 60)

    # 1. Create explainer
    explainer = create_explainer(model, X_train, model_name, feature_names)

    # 2. Compute SHAP values
    shap_values, importance_df = compute_shap_values(
        explainer, X_test, feature_names, max_samples=min(80, X_test.shape[0])
    )

    # 3. Generate plots
    n_test = min(80, X_test.shape[0])
    X_explain = X_test[:n_test]

    saved_plots = {
        "shap_summary": plot_shap_summary(
            shap_values, X_explain, feature_names,
            class_names, model_name, "12_shap_summary.png"
        ),
        "shap_bar": plot_shap_bar(importance_df, model_name, "13_shap_bar_importance.png"),
        "shap_waterfall": plot_shap_waterfall(
            shap_values, sample_idx, feature_names,
            class_names, model_name, "14_shap_waterfall.png"
        ),
        "shap_dependence": plot_shap_dependence(
            shap_values, X_explain, feature_names,
            feature_names[0], None, model_name, "15_shap_dependence.png"
        ),
    }

    # 4. Export importance
    export_path = os.path.join(export_dir, "shap_feature_importance.xlsx")
    export_shap_importance(importance_df, export_path)

    logger.info("✅ SHAP Explainability Pipeline Complete.")

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "importance_df": importance_df,
        "saved_plots": saved_plots,
        "export_path": export_path,
    }
