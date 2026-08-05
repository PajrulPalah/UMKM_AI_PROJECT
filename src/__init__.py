# ============================================================
# src/__init__.py
# UMKM AI Business Resilience Prediction System
# Package Initialization
# ============================================================

"""
UMKM AI — Business Resilience Prediction System
================================================

A complete Machine Learning pipeline for predicting Business Resilience
of Indonesian SMEs (UMKM) using organizational capability variables.

Modules
-------
preprocessing  : Data cleaning, encoding, scaling, train-test split
training       : Model training, SMOTE, GridSearchCV, persistence
evaluation     : Metrics, confusion matrix, model comparison, learning curves
visualization  : All plot generation functions
explainability : SHAP-based model explainability
prediction     : Inference for new SME instances

Usage
-----
    from src.preprocessing import run_preprocessing_pipeline
    from src.training import train_all_models, select_best_model
    from src.evaluation import evaluate_all_models, build_comparison_table
    from src.visualization import generate_all_visualizations
    from src.explainability import run_explainability_pipeline
    from src.prediction import predict_single, predict_batch

Version
-------
    1.0.0 — Initial release
"""

__version__ = "1.0.0"
__author__ = "AI Engineer"
__project__ = "UMKM AI Business Resilience Prediction System"
