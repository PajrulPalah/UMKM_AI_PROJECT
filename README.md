# 🏢 AI-Based Business Resilience Prediction System for Indonesian SMEs (UMKM)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/XGBoost-2.0+-red" alt="XGBoost">
  <img src="https://img.shields.io/badge/SHAP-0.43+-green" alt="SHAP">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## 📋 Project Overview

This is a complete, end-to-end **Machine Learning research system** designed to predict the **Business Resilience Level** of Indonesian Small and Medium Enterprises (SMEs / UMKM).

The system is built for:
- 🎓 **Academic Research** — Master Thesis / Doctoral Dissertation
- 📊 **Business Analytics** — UMKM performance benchmarking
- 🚀 **Future Deployment** — Streamlit web application ready

---

## 🔬 Research Background

Indonesian SMEs (UMKM) contribute approximately **60.5% of national GDP** and employ **97% of the workforce** (BPS, 2023). Despite their economic significance, UMKM face persistent resilience challenges during economic shocks, pandemics, and market disruptions.

This research operationalizes **Business Resilience** as a multi-dimensional construct influenced by:

| Construct | Abbreviation | Items | Scale |
|-----------|-------------|-------|-------|
| Digital Capability | DC | 5 | Likert 1-5 |
| Innovation Capability | IC | 5 | Likert 1-5 |
| Entrepreneurial Orientation | EO | 5 | Likert 1-5 |
| Organizational Agility | OA | 5 | Likert 1-5 |
| Resource Access | RA | 5 | Likert 1-5 |
| Environmental Dynamism | ED | 5 | Likert 1-5 |
| **Business Resilience** | **BR** | **7** | **Likert 1-5** |

### Theoretical Framework (Causal Structure)

```
Digital Capability (DC)
    └──► Innovation Capability (IC)         [β ≈ 0.65]
              └──► Organizational Agility (OA)  [β ≈ 0.60]
                        └──► Business Resilience (BR) [β ≈ 0.70]

Entrepreneurial Orientation (EO) ──► IC     [β ≈ 0.55]
Resource Access (RA)             ──► BR     [β ≈ 0.60]
Environmental Dynamism (ED)      ──► BR     [β ≈ -0.40, moderated by OA]
```

---

## 📁 Folder Structure

```
UMKM_AI_PROJECT/
│
├── notebook/
│   └── UMKM_AI.ipynb          ← Main Jupyter Notebook (15 sections, 57 cells)
│
├── data/
│   ├── UMKM_Dummy_Data.xlsx   ← Synthetic dataset (100 SMEs, 50 variables)
│   └── UMKM_Dummy_Data.csv    ← Same dataset in CSV format
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── figures/                   ← Auto-generated visualizations (15+ plots)
│   ├── 01_correlation_heatmap.png
│   ├── 02_distributions.png
│   ├── 03_boxplots.png
│   ├── 04_class_distribution.png
│   ├── 05_model_comparison.png
│   ├── 06_confusion_matrices.png
│   ├── 07_roc_curves.png
│   ├── 09_learning_curve.png
│   ├── 12_shap_summary.png
│   ├── 13_shap_bar_importance.png
│   ├── 14_shap_waterfall.png
│   └── 15_shap_dependence.png
│
├── src/
│   ├── preprocessing.py       ← Data cleaning, encoding, scaling
│   ├── training.py            ← Model training, SMOTE, GridSearchCV
│   ├── evaluation.py          ← Metrics, comparison, learning curves
│   ├── visualization.py       ← All plot generation functions
│   ├── explainability.py      ← SHAP explainability pipeline
│   ├── prediction.py          ← Inference for new SME instances
│   └── generate_notebook.py   ← Notebook builder script
│
├── evaluation_report.csv      ← Model comparison table
├── prediction_results.xlsx    ← Full dataset predictions
├── shap_feature_importance.xlsx ← SHAP importance ranking
│
├── requirements.txt           ← Python package dependencies
├── README.md                  ← This file
└── report.md                  ← Markdown research report
```

---

## ⚙️ Installation

### Step 1: Clone / Download Project

```bash
# Navigate to project root
cd "d:/PROJECT UMKM FORECAST/UMKM_AI_PROJECT"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Jupyter (if not already installed)

```bash
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=umkm_ai
```

---

## 📦 Requirements

```
pandas>=2.1.0
numpy>=1.26.0
scipy>=1.11.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
shap>=0.43.0
matplotlib>=3.8.0
seaborn>=0.13.0
openpyxl>=3.1.2
joblib>=1.3.0
ipykernel>=6.25.0
```

---

## 🚀 How to Run

### Option A: Jupyter Notebook (Recommended)

```bash
# Start Jupyter from project root
jupyter notebook notebook/UMKM_AI.ipynb
```

Then run all cells: **Kernel → Restart & Run All**

### Option B: Google Colab

1. Upload the `UMKM_AI_PROJECT/` folder to Google Drive
2. Open `notebook/UMKM_AI.ipynb` in Colab
3. Mount Drive and update `PROJECT_ROOT` path
4. Install requirements: `!pip install -r requirements.txt`

### Option C: VSCode

1. Open project in VSCode
2. Install Python + Jupyter extensions
3. Open `notebook/UMKM_AI.ipynb`
4. Select `umkm_ai` kernel
5. Run all cells

### Option D: Regenerate Notebook

```bash
python src/generate_notebook.py
```

---

## 📊 Expected Outputs

After successful execution, the following files will be generated:

| File | Description |
|------|-------------|
| `data/UMKM_Dummy_Data.xlsx` | Synthetic UMKM dataset |
| `data/UMKM_Dummy_Data.csv` | CSV version of dataset |
| `model/*.pkl` | Trained ML models (4 files) |
| `figures/*.png` | Visualization plots (15+ files) |
| `evaluation_report.csv` | Model comparison table |
| `prediction_results.xlsx` | Batch predictions with probabilities |
| `shap_feature_importance.xlsx` | SHAP importance ranking |

---

## 🤖 Machine Learning Models

| Model | Algorithm Type | Key Strength |
|-------|---------------|-------------|
| Logistic Regression | Linear, Probabilistic | Fast, interpretable baseline |
| Decision Tree | Rule-based | Highly interpretable |
| Random Forest | Ensemble (Bagging) | Robust, handles nonlinearity |
| **XGBoost** | **Ensemble (Boosting)** | **Typically best performance** |

### Model Selection Criteria
- **Primary**: F1-Score (Weighted) — accounts for class imbalance
- **Secondary**: ROC AUC — discrimination ability
- **Stability**: 10-fold Cross-Validation

---

## 🧠 Explainable AI (SHAP)

SHAP values are computed using the appropriate explainer:
- **TreeExplainer** — for Random Forest, Decision Tree, XGBoost (exact)
- **LinearExplainer** — for Logistic Regression (exact)
- **KernelExplainer** — fallback model-agnostic (approximate)

### SHAP Plots Generated:
1. **Summary Plot** — Feature impact distribution across test samples
2. **Bar Plot** — Ranked mean absolute importance
3. **Waterfall Plot** — Single prediction decomposition
4. **Dependence Plot** — Feature value vs SHAP value relationship

---

## 🔮 Prediction Usage

To predict for a new SME, provide 6 construct mean scores (1.0–5.0):

```python
import joblib

payload = joblib.load('model/random_forest.pkl')
model   = payload['model']
scaler  = payload['scaler']

# Example: New UMKM survey data
new_sme = {
    'DigitalCapabilityScore':          4.2,
    'InnovationCapabilityScore':       3.8,
    'EntrepreneurialOrientationScore': 4.0,
    'OrganizationalAgilityScore':      4.5,
    'ResourceAccessScore':             3.6,
    'EnvironmentalDynamismScore':      3.2,
}

feature_cols = list(new_sme.keys())
X_new = scaler.transform([[new_sme[f] for f in feature_cols]])
prediction = model.predict(X_new)
probabilities = model.predict_proba(X_new)

print(f"Predicted Category: {prediction[0]}")
print(f"Confidence: {probabilities.max():.2%}")
```

---

## 📸 Screenshots Placeholder

> *Screenshots will be added after the first full notebook execution.*

| Plot | Description |
|------|-------------|
| `figures/01_correlation_heatmap.png` | Construct correlation matrix |
| `figures/05_model_comparison.png` | Algorithm performance comparison |
| `figures/07_roc_curves.png` | Multi-class ROC curves |
| `figures/12_shap_summary.png` | SHAP beeswarm feature importance |
| `figures/14_shap_waterfall.png` | Single prediction explanation |

---

## 🚀 Future Development

- [ ] **Streamlit App** — Interactive web dashboard for UMKM counselors
- [ ] **Real Data Collection** — Google Forms survey pipeline → auto-prediction
- [ ] **SEM Analysis** — Structural Equation Modelling (SmartPLS / lavaan)
- [ ] **Longitudinal Tracking** — Monitor resilience changes over quarters
- [ ] **API Service** — FastAPI endpoint for integration with gov systems
- [ ] **Mobile App** — Android/iOS for field data collection
- [ ] **ASEAN Benchmarking** — Cross-country SME resilience comparison

---

## 📚 References

1. Teece, D. J. (2007). Explicating dynamic capabilities. *Strategic Management Journal*, 28(13), 1319-1350.
2. Burnard, K., & Bhamra, R. (2011). Organisational resilience. *International Journal of Production Research*, 49(18), 5581-5599.
3. BPS-Statistics Indonesia. (2023). *Laporan Perkembangan UMKM Indonesia*.
4. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.

---

## 📄 License

MIT License — Free for academic and commercial use.

---

*Built with ❤️ for Indonesian UMKM Research | Python 3.12 | 2025*
