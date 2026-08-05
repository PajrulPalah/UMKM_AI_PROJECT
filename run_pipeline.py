#!/usr/bin/env python3
# ============================================================
# run_pipeline.py
# UMKM AI Business Resilience Prediction System
# Master Execution Script — Runs the complete pipeline
# outside of Jupyter for validation and automated output generation
# ============================================================

"""
This script executes the complete UMKM AI pipeline programmatically,
generating all outputs: dataset, models, figures, and reports.

Run from the UMKM_AI_PROJECT root directory:
    python run_pipeline.py
"""

import os
import sys
import warnings
import time

warnings.filterwarnings("ignore")

# ── Add src to Python path ─────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, GridSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import joblib
from xgboost import XGBClassifier

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT, "data")
MODEL_DIR   = os.path.join(ROOT, "model")
FIGURES_DIR = os.path.join(ROOT, "figures")

for d in [DATA_DIR, MODEL_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
CLASS_ORDER  = ["Low", "Medium", "High"]
np.random.seed(RANDOM_STATE)

print("=" * 65)
print("  UMKM AI — Business Resilience Prediction System")
print("  Full Pipeline Execution")
print("=" * 65)


# ══════════════════════════════════════════════════════════════
# STEP 1: DATA GENERATION
# ══════════════════════════════════════════════════════════════

def generate_likert(base, n_items=5, noise_std=0.40, seed_offset=0):
    np.random.seed(RANDOM_STATE + seed_offset)
    items = {}
    for i in range(n_items):
        raw = base + np.random.normal(0, noise_std, len(base))
        items[i] = np.clip(np.round(raw), 1, 5).astype(int)
    return pd.DataFrame(items)


def generate_dataset(n=100):
    np.random.seed(RANDOM_STATE)
    provinces = ["Jawa Barat", "Jawa Tengah", "Jawa Timur", "DKI Jakarta",
                 "Sumatera Utara", "Bali", "Sulawesi Selatan", "Kalimantan Timur"]
    cities = {
        "Jawa Barat": ["Bandung", "Bekasi", "Bogor"],
        "Jawa Tengah": ["Semarang", "Solo", "Yogyakarta"],
        "Jawa Timur": ["Surabaya", "Malang", "Sidoarjo"],
        "DKI Jakarta": ["Jakarta Pusat", "Jakarta Selatan", "Jakarta Barat"],
        "Sumatera Utara": ["Medan", "Binjai", "Pematangsiantar"],
        "Bali": ["Denpasar", "Badung", "Gianyar"],
        "Sulawesi Selatan": ["Makassar", "Gowa", "Maros"],
        "Kalimantan Timur": ["Samarinda", "Balikpapan", "Bontang"],
    }
    sectors = ["Kuliner", "Fashion & Tekstil", "Kerajinan Tangan",
               "Teknologi & Digital", "Pertanian & Pangan",
               "Perdagangan Eceran", "Jasa & Konsultasi", "Manufaktur"]
    genders = ["Laki-laki", "Perempuan"]
    educations = ["SD/SMP", "SMA/SMK", "Diploma (D1-D3)",
                  "Sarjana (S1)", "Magister (S2)", "Doktor (S3)"]
    legal_statuses = ["Tidak Berbadan Hukum", "CV", "PT", "Koperasi", "Usaha Dagang (UD)"]

    prov_list = np.random.choice(provinces, n)
    city_list = [np.random.choice(cities[p]) for p in prov_list]

    profile = pd.DataFrame({
        "ID_UMKM": [f"UMKM-{i+1:03d}" for i in range(n)],
        "Business_Name": [f"Usaha {chr(65 + i % 26)}{i+1}" for i in range(n)],
        "Province": prov_list,
        "City": city_list,
        "Business_Sector": np.random.choice(sectors, n),
        "Business_Age": np.clip(np.random.exponential(6, n), 0.5, 30).round(1),
        "Number_of_Employees": np.clip(np.random.negative_binomial(3, 0.4, n), 1, 50),
        "Annual_Revenue": np.clip(np.random.lognormal(17.5, 1.2, n), 5e6, 5e9).astype(int),
        "Digital_Sales_Percentage": np.clip(np.random.beta(2, 3, n) * 100, 0, 100).round(1),
        "Owner_Age": np.clip(np.random.normal(42, 10, n), 20, 70).astype(int),
        "Owner_Gender": np.random.choice(genders, n, p=[0.55, 0.45]),
        "Education": np.random.choice(educations, n, p=[0.05, 0.25, 0.20, 0.35, 0.12, 0.03]),
        "Legal_Status": np.random.choice(legal_statuses, n, p=[0.30, 0.25, 0.20, 0.10, 0.15]),
    })

    # ── Stronger causal structure for clear class separation ──────
    # Use uniform base scores spread across the full 1–5 range
    # to ensure distinct Low / Medium / High clusters
    DC_base = np.clip(np.random.uniform(1.2, 5.0, n), 1.0, 5.0)
    EO_base = np.clip(np.random.uniform(1.2, 5.0, n) * 0.5
                      + np.random.normal(3.0, 0.6, n) * 0.5, 1.0, 5.0)

    IC_base = np.clip(
        0.62 * DC_base + 0.28 * EO_base + np.random.normal(0, 0.32, n),
        1.0, 5.0
    )
    OA_base = np.clip(
        0.58 * IC_base + 0.25 * DC_base + np.random.normal(0, 0.30, n),
        1.0, 5.0
    )
    RA_base = np.clip(
        0.48 * OA_base + np.random.normal(1.0, 0.50, n),
        1.0, 5.0
    )
    ED_base = np.clip(np.random.normal(3.0, 0.90, n), 1.0, 5.0)

    # Business Resilience — strong signal, low noise
    ED_penalty = ED_base * np.clip(1.0 - 0.42 * (OA_base / 5.0), 0.2, 1.0)
    BR_base = np.clip(
        0.40 * OA_base
        + 0.27 * RA_base
        + 0.20 * IC_base
        + 0.08 * DC_base
        - 0.12 * ED_penalty
        + np.random.normal(0, 0.22, n),
        1.0, 5.0
    )

    # ── Generate Likert Items (tighter noise for higher α) ────────
    dc = generate_likert(DC_base, 5, 0.35, 1);  dc.columns = [f"DC{i+1}"  for i in range(5)]
    ic = generate_likert(IC_base, 5, 0.33, 2);  ic.columns = [f"IC{i+1}"  for i in range(5)]
    eo = generate_likert(EO_base, 5, 0.38, 3);  eo.columns = [f"EO{i+1}"  for i in range(5)]
    oa = generate_likert(OA_base, 5, 0.35, 4);  oa.columns = [f"OA{i+1}"  for i in range(5)]
    ra = generate_likert(RA_base, 5, 0.36, 5);  ra.columns = [f"RA{i+1}"  for i in range(5)]
    ed = generate_likert(ED_base, 5, 0.42, 6);  ed.columns = [f"ED{i+1}"  for i in range(5)]
    br = generate_likert(BR_base, 7, 0.30, 7);  br.columns = [f"BR{i+1}"  for i in range(7)]

    return pd.concat([profile.reset_index(drop=True), dc, ic, eo, oa, ra, ed, br], axis=1)


print("\n[1/8] Generating synthetic dataset...")
df = generate_dataset(100)
print(f"      Shape: {df.shape}")

# ── Save dataset (handle Excel file-lock gracefully) ───────
try:
    df.to_excel(os.path.join(DATA_DIR, "UMKM_Dummy_Data.xlsx"), index=False)
except PermissionError:
    from datetime import datetime
    df.to_excel(os.path.join(DATA_DIR, f"UMKM_Dummy_Data_{datetime.now().strftime('%H%M%S')}.xlsx"), index=False)
    print("      (UMKM_Dummy_Data.xlsx locked — saved with timestamp suffix)")
df.to_csv(os.path.join(DATA_DIR, "UMKM_Dummy_Data.csv"), index=False, encoding="utf-8-sig")
print(f"      Saved to data/")



# ══════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

CONSTRUCTS = {
    "DigitalCapabilityScore":          ["DC1","DC2","DC3","DC4","DC5"],
    "InnovationCapabilityScore":       ["IC1","IC2","IC3","IC4","IC5"],
    "EntrepreneurialOrientationScore": ["EO1","EO2","EO3","EO4","EO5"],
    "OrganizationalAgilityScore":      ["OA1","OA2","OA3","OA4","OA5"],
    "ResourceAccessScore":             ["RA1","RA2","RA3","RA4","RA5"],
    "EnvironmentalDynamismScore":      ["ED1","ED2","ED3","ED4","ED5"],
    "BusinessResilienceScore":         ["BR1","BR2","BR3","BR4","BR5","BR6","BR7"],
}

print("\n[2/8] Computing construct scores...")
for score_col, items in CONSTRUCTS.items():
    df[score_col] = df[[c for c in items if c in df.columns]].mean(axis=1).round(3)

# Use TERTILE-based thresholds → perfectly balanced classes (~33/33/34)
br_scores = df["BusinessResilienceScore"]
low_thresh  = br_scores.quantile(1/3)
high_thresh = br_scores.quantile(2/3)

df["BusinessResilienceCategory"] = br_scores.apply(
    lambda s: "Low" if s <= low_thresh else ("Medium" if s <= high_thresh else "High")
)

FEATURE_COLS = [
    "DigitalCapabilityScore", "InnovationCapabilityScore",
    "EntrepreneurialOrientationScore", "OrganizationalAgilityScore",
    "ResourceAccessScore", "EnvironmentalDynamismScore",
]
TARGET_COL = "BusinessResilienceCategory"

dist = df[TARGET_COL].value_counts()
print(f"      Distribution: {dict(dist)}")


# ══════════════════════════════════════════════════════════════
# STEP 3: EDA VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

print("\n[3/8] Generating EDA visualizations...")

SCORE_COLS = list(CONSTRUCTS.keys())

# Correlation heatmap
fig, ax = plt.subplots(figsize=(11, 9))
corr = df[[c for c in SCORE_COLS if c in df.columns]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 9}, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Construct Score Correlation Matrix", fontsize=15, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "01_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# Distributions
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(SCORE_COLS)))
for idx, (col, color) in enumerate(zip(SCORE_COLS, colors)):
    if col not in df.columns:
        continue
    data = df[col].dropna()
    axes[idx].hist(data, bins=15, color=color, alpha=0.75, edgecolor="white")
    axes[idx].axvline(data.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mu={data.mean():.2f}")
    axes[idx].set_title(col.replace("Score", ""), fontsize=9, fontweight="bold")
    axes[idx].set_xlabel("Score (1-5)")
    axes[idx].legend(fontsize=7)
for idx in range(len(SCORE_COLS), len(axes)):
    axes[idx].set_visible(False)
fig.suptitle("Construct Score Distributions", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "02_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()

# Class distribution
fig, ax = plt.subplots(figsize=(7, 5))
palette = {"Low": "#C73E1D", "Medium": "#F18F01", "High": "#2E86AB"}
cats = [c for c in CLASS_ORDER if c in dist.index]
vals = [dist[c] for c in cats]
ax.bar(cats, vals, color=[palette[c] for c in cats], edgecolor="white", linewidth=0.8, width=0.5)
for i, (c, v) in enumerate(zip(cats, vals)):
    ax.text(i, v + 0.3, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
ax.set_title("Business Resilience Category Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
ax.set_ylim(0, max(vals) * 1.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "04_class_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

print("      Saved: 01_correlation_heatmap, 02_distributions, 04_class_distribution")


# ══════════════════════════════════════════════════════════════
# STEP 4: PREPROCESSING
# ══════════════════════════════════════════════════════════════

print("\n[4/8] Preprocessing data...")
X = df[FEATURE_COLS]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

if SMOTE_AVAILABLE:
    try:
        smote = SMOTE(sampling_strategy="auto", random_state=RANDOM_STATE, k_neighbors=3)
        X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)
        print(f"      SMOTE: {len(y_train)} -> {len(y_train_res)} samples")
    except Exception as e:
        print(f"      SMOTE skipped: {e}")
        X_train_res, y_train_res = X_train_sc, y_train
else:
    X_train_res, y_train_res = X_train_sc, y_train

le = LabelEncoder()
le.fit(CLASS_ORDER)
y_train_enc = le.transform(y_train_res)
y_test_enc  = le.transform(y_test)
X_all_sc = scaler.transform(X)


# ══════════════════════════════════════════════════════════════
# STEP 5: MODEL TRAINING
# ══════════════════════════════════════════════════════════════

print("\n[5/8] Training models with GridSearchCV...")

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
        "model": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                               random_state=RANDOM_STATE, verbosity=0),
        "param_grid": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.1, 0.2]},
    },
}

trained_models = {}
cv_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

for model_name, config in MODEL_CONFIGS.items():
    t0 = time.time()
    is_xgb = model_name == "XGBoost"
    X_tr = X_train_res
    y_tr = y_train_enc if is_xgb else y_train_res

    gs = GridSearchCV(
        config["model"], config["param_grid"],
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_weighted", n_jobs=-1, refit=True, verbose=0
    )
    gs.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    trained_models[model_name] = {
        "model": gs.best_estimator_,
        "best_params": gs.best_params_,
        "best_cv_score": gs.best_score_,
    }
    print(f"      {model_name:25s} | Best F1: {gs.best_score_:.4f} | {elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════
# STEP 6: EVALUATION
# ══════════════════════════════════════════════════════════════

print("\n[6/8] Evaluating models...")

eval_results = {}
y_bin = label_binarize(y_test, classes=CLASS_ORDER)

for model_name, result in trained_models.items():
    model = result["model"]
    is_xgb = model_name == "XGBoost"

    if is_xgb:
        y_pred_enc = model.predict(X_test_sc)
        y_pred = le.inverse_transform(y_pred_enc)
        y_prob_raw = model.predict_proba(X_test_sc)
        xgb_cls = list(le.inverse_transform(model.classes_))
        y_prob = np.column_stack([y_prob_raw[:, xgb_cls.index(c)] for c in CLASS_ORDER])
    else:
        y_pred = model.predict(X_test_sc)
        y_prob_raw = model.predict_proba(X_test_sc)
        model_cls = list(model.classes_)
        y_prob = np.column_stack([y_prob_raw[:, model_cls.index(c)] for c in CLASS_ORDER])

    acc  = accuracy_score(y_test, y_pred)
    f1w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        roc = roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")
    except Exception:
        roc = float("nan")

    y_cv = le.transform(y) if is_xgb else y
    cv_sc = cross_val_score(model, X_all_sc, y_cv, cv=cv_skf, scoring="f1_weighted", n_jobs=-1)

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)

    eval_results[model_name] = {
        "model": model, "y_pred": y_pred, "y_prob": y_prob,
        "accuracy": round(acc, 4), "f1_weighted": round(f1w, 4),
        "roc_auc": round(roc, 4) if not np.isnan(roc) else float("nan"),
        "cv_mean": round(cv_sc.mean(), 4), "cv_std": round(cv_sc.std(), 4),
        "confusion_matrix": cm_df,
    }
    print(f"      {model_name:25s} | Acc: {acc:.4f} | F1: {f1w:.4f} | AUC: {roc:.4f} | CV: {cv_sc.mean():.4f}")


# ══════════════════════════════════════════════════════════════
# STEP 7: COMPARISON & VISUALIZATION
# ══════════════════════════════════════════════════════════════

print("\n[7/8] Generating evaluation plots & selecting best model...")

rows = []
for mn, r in eval_results.items():
    rows.append({
        "Model": mn, "Accuracy": r["accuracy"], "F1 Weighted": r["f1_weighted"],
        "ROC AUC": r["roc_auc"], "CV Mean": r["cv_mean"], "CV Std": r["cv_std"],
    })
comparison_df = pd.DataFrame(rows).sort_values("F1 Weighted", ascending=False).reset_index(drop=True)
comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))
comparison_df.to_csv(os.path.join(ROOT, "evaluation_report.csv"), index=False)

best_model_name = comparison_df.iloc[0]["Model"]
best_model = eval_results[best_model_name]["model"]
print(f"      Best model: {best_model_name} (F1={comparison_df.iloc[0]['F1 Weighted']:.4f})")

# Confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, (mn, r) in zip(axes, eval_results.items()):
    sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                ax=ax, cbar=False, linewidths=0.8,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"{mn}\nF1={r['f1_weighted']:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "06_confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()

# Model comparison bar chart
metrics_plot = ["Accuracy", "F1 Weighted", "ROC AUC", "CV Mean"]
x = np.arange(len(comparison_df))
width = 0.18
colors_bar = ["#2E86AB", "#F18F01", "#A23B72", "#3BA55D"]
fig, ax = plt.subplots(figsize=(13, 6))
for i, (metric, color) in enumerate(zip(metrics_plot, colors_bar)):
    vals = comparison_df[metric].fillna(0).values
    offset = (i - 2) * width + width / 2
    ax.bar(x + offset, vals, width, label=metric, color=color, alpha=0.85, edgecolor="white")
ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(comparison_df["Model"].tolist(), rotation=10, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.12)
ax.legend(loc="upper right", fontsize=9); ax.yaxis.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "05_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ROC curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
model_colors = ["#2E86AB", "#F18F01", "#A23B72", "#3BA55D"]
for cls_idx, (ax, cls_name) in enumerate(zip(axes, CLASS_ORDER)):
    for (mn, r), color in zip(eval_results.items(), model_colors):
        yp = r.get("y_prob")
        if yp is None or yp.shape[1] < 3:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], yp[:, cls_idx])
        ax.plot(fpr, tpr, linewidth=2, color=color, label=f"{mn} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1],"k--",linewidth=1,alpha=0.5)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_title(f"ROC — Class: {cls_name}", fontweight="bold")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.3)
fig.suptitle("Multi-Class ROC Curves (OvR)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "07_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"      Figures saved to: {FIGURES_DIR}")


# ══════════════════════════════════════════════════════════════
# STEP 8: SAVE MODELS & EXPORT RESULTS
# ══════════════════════════════════════════════════════════════

print("\n[8/8] Saving models and exporting results...")

name_map = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}
for mn, result in trained_models.items():
    payload = {
        "model": result["model"], "scaler": scaler,
        "label_encoder": le, "feature_cols": FEATURE_COLS,
        "class_order": CLASS_ORDER, "model_name": mn,
        "best_params": result["best_params"],
    }
    fpath = os.path.join(MODEL_DIR, f"{name_map.get(mn, mn.lower().replace(' ','_'))}.pkl")
    joblib.dump(payload, fpath, compress=3)
    sz = os.path.getsize(fpath) / 1024
    print(f"      Saved: {os.path.basename(fpath)} ({sz:.1f} KB)")

# Export batch predictions
X_all_sc2 = scaler.transform(X)
if best_model_name == "XGBoost":
    y_pred_all = le.inverse_transform(best_model.predict(X_all_sc2))
else:
    y_pred_all = best_model.predict(X_all_sc2)

y_prob_all = best_model.predict_proba(X_all_sc2)
if best_model_name == "XGBoost":
    mc = list(le.inverse_transform(best_model.classes_))
else:
    mc = list(best_model.classes_)

pred_df = df[["ID_UMKM", "Business_Name", "Province", "Business_Sector",
              "BusinessResilienceScore", "BusinessResilienceCategory"]].copy()
pred_df["Predicted_Category"] = y_pred_all
for cls in CLASS_ORDER:
    if cls in mc:
        pred_df[f"Prob_{cls}"] = y_prob_all[:, mc.index(cls)].round(4)
    else:
        pred_df[f"Prob_{cls}"] = 0.0
pred_df["Confidence"] = y_prob_all.max(axis=1).round(4)
pred_df["Correct"] = pred_df["Predicted_Category"] == pred_df["BusinessResilienceCategory"]
try:
    pred_df.to_excel(os.path.join(ROOT, "prediction_results.xlsx"), index=False)
    print(f"      prediction_results.xlsx saved ({pred_df['Correct'].mean():.1%} accuracy on full dataset)")
except PermissionError:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback = os.path.join(ROOT, f"prediction_results_{ts}.xlsx")
    pred_df.to_excel(fallback, index=False)
    print(f"      Saved as {os.path.basename(fallback)} ({pred_df['Correct'].mean():.1%} accuracy — original file was open in Excel)")

print("\n" + "=" * 65)
print("  PIPELINE COMPLETE!")
print("=" * 65)
print(f"\n  Best Model   : {best_model_name}")
print(f"  Test F1      : {comparison_df.iloc[0]['F1 Weighted']:.4f}")
print(f"  Test Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
print(f"  Models saved : {MODEL_DIR}")
print(f"  Figures saved: {FIGURES_DIR}")
print(f"\n  Run the Jupyter notebook for full analysis:")
print(f"  jupyter notebook notebook/UMKM_AI.ipynb")
print("=" * 65)
