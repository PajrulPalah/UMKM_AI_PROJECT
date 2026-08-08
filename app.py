# ============================================================
# app.py
# UMKM Business Resilience Prediction System
# Streamlit Web Application
# Version: 2.0.0 — VSCode + Auto-Retraining Edition
# ============================================================

import os
import sys
import warnings
import json
import datetime
import time
import threading

warnings.filterwarnings("ignore")

# ── Import Gemini helper (opsional, graceful fallback) ─────
try:
    from gemini_helper import render_gemini_section, is_gemini_ready
    GEMINI_MODULE_OK = True
except ImportError:
    GEMINI_MODULE_OK = False


import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
MODEL_DIR   = ROOT / "model"
FIGURES_DIR = ROOT / "figures"
LOG_DIR     = ROOT / "data" / "submissions"

for d in [DATA_DIR, MODEL_DIR, FIGURES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Add src to path ────────────────────────────────────────
sys.path.insert(0, str(ROOT / "src"))

# ── Constants ──────────────────────────────────────────────
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

LIKERT_LABELS = {
    1: "1 – Sangat Tidak Setuju",
    2: "2 – Tidak Setuju",
    3: "3 – Netral",
    4: "4 – Setuju",
    5: "5 – Sangat Setuju",
}

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="UMKM Resilience AI | Sistem Prediksi Ketahanan Bisnis",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Premium Dark Design) ──────────────────────
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a2332 100%);
        color: #e6edf3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #1f2937 100%);
        border-right: 1px solid #30363d;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d1b69 50%, #1a3a2a 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .hero-banner h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #a5f3fc, #6ee7b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-banner p {
        color: #8b949e;
        font-size: 1.05rem;
    }

    /* Section cards */
    .section-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #58a6ff;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #21262d;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result cards */
    .result-high {
        background: linear-gradient(135deg, #0d2818 0%, #1a4731 100%);
        border: 2px solid #2ea043;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-medium {
        background: linear-gradient(135deg, #2d1e00 0%, #4d3300 100%);
        border: 2px solid #d29922;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #1f0a0a 0%, #3d1515 100%);
        border: 2px solid #da3633;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-label {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .result-sublabel {
        font-size: 1rem;
        color: #8b949e;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #8b949e;
        margin-top: 0.2rem;
    }

    /* Stacked progress bar */
    .prob-bar-container {
        background: #21262d;
        border-radius: 8px;
        overflow: hidden;
        height: 28px;
        margin: 0.3rem 0;
        display: flex;
    }

    /* Likert radio styling */
    .stRadio > label {
        color: #e6edf3 !important;
        font-size: 0.9rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043, #3fb950);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(46,160,67,0.4);
    }

    /* Info box */
    .info-box {
        background: rgba(56, 139, 253, 0.1);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #79c0ff;
    }

    /* Warning box */
    .warn-box {
        background: rgba(210, 153, 34, 0.1);
        border: 1px solid rgba(210, 153, 34, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #e3b341;
    }

    /* Recommendation list */
    .rec-item {
        background: #1c2128;
        border-left: 3px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #c9d1d9;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Score gauge */
    .score-gauge {
        text-align: center;
        padding: 1rem;
    }
    .score-number {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
    }
    .score-max {
        font-size: 1.2rem;
        color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "random_forest"):
    """Load trained model from disk."""
    path = MODEL_DIR / f"{model_name}.pkl"
    if not path.exists():
        return None, None, None, None
    payload = joblib.load(path)
    return (
        payload["model"],
        payload["scaler"],
        payload.get("label_encoder"),
        payload.get("feature_cols", FEATURE_COLS),
    )


def compute_construct_scores(responses: dict) -> dict:
    """Compute mean score for each construct from Likert items."""
    scores = {}
    for score_col, items in CONSTRUCT_ITEMS.items():
        vals = [responses.get(item, 3) for item in items]
        scores[score_col] = round(np.mean(vals), 3)
    return scores


def predict_resilience(model, scaler, label_encoder, scores: dict) -> dict:
    """Run prediction from construct scores."""
    feat_vals = [scores.get(f, 3.0) for f in FEATURE_COLS]
    X_input   = np.array([feat_vals])
    X_scaled  = scaler.transform(X_input)

    pred_raw  = model.predict(X_scaled)[0]

    # Map encoded prediction back to label
    if label_encoder is not None and isinstance(pred_raw, (int, np.integer)):
        pred_class = label_encoder.inverse_transform([pred_raw])[0]
    else:
        pred_class = str(pred_raw)

    # Probabilities
    probs = {}
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[0]
        model_classes = list(model.classes_)
        for cls_enc, prob in zip(model_classes, y_prob):
            if label_encoder is not None and isinstance(cls_enc, (int, np.integer)):
                cls_name = label_encoder.inverse_transform([cls_enc])[0]
            else:
                cls_name = str(cls_enc)
            probs[cls_name] = round(float(prob), 4)

    # Ensure all classes present
    for c in CLASS_ORDER:
        probs.setdefault(c, 0.0)

    confidence = probs.get(pred_class, 0.0)

    # Weighted resilience score estimate (1–5)
    weights = {
        "DigitalCapabilityScore": 0.12,
        "InnovationCapabilityScore": 0.18,
        "EntrepreneurialOrientationScore": 0.10,
        "OrganizationalAgilityScore": 0.30,
        "ResourceAccessScore": 0.22,
        "EnvironmentalDynamismScore": 0.08,
    }
    br_score = sum(scores.get(f, 3.0) * w for f, w in weights.items())

    return {
        "predicted_class": pred_class,
        "probabilities": probs,
        "confidence": confidence,
        "br_score": round(float(np.clip(br_score, 1.0, 5.0)), 2),
    }


def save_submission(profile: dict, responses: dict, scores: dict, result: dict):
    """
    Save each form submission to CSV for auto-retraining.
    New data is appended to data/submissions/submissions_log.csv
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Timestamp": now,
        **profile,
        **responses,  # all 37 Likert items
        **{k: v for k, v in scores.items()},
        "Predicted_Category": result["predicted_class"],
        "Confidence": result["confidence"],
        "BR_Score_Estimated": result["br_score"],
    }
    log_path = LOG_DIR / "submissions_log.csv"
    df_row = pd.DataFrame([row])

    if log_path.exists():
        df_row.to_csv(log_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df_row.to_csv(log_path, mode="w", header=True, index=False, encoding="utf-8-sig")

    return log_path


def get_submission_count() -> int:
    """Return total number of logged submissions."""
    log_path = LOG_DIR / "submissions_log.csv"
    if not log_path.exists():
        return 0
    try:
        return len(pd.read_csv(log_path)) 
    except Exception:
        return 0


def retrain_model_background(n_new: int = 10):
    """
    Auto-retrain all models when new submissions reach threshold.
    Runs in background thread — non-blocking.
    """
    def _retrain():
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(ROOT / "retrain.py")],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                # Clear model cache so next prediction uses new model
                load_model.clear()
                st.session_state["retrain_done"] = True
                st.session_state["retrain_msg"] = f"✅ Model berhasil dilatih ulang dengan {n_new} data baru!"
            else:
                st.session_state["retrain_msg"] = f"⚠️ Retraining gagal: {result.stderr[:200]}"
        except Exception as e:
            st.session_state["retrain_msg"] = f"⚠️ Retraining error: {str(e)}"

    thread = threading.Thread(target=_retrain, daemon=True)
    thread.start()


def get_recommendations(result: dict, scores: dict) -> list:
    """Generate strategic recommendations based on prediction and scores."""
    pred   = result["predicted_class"]
    recs   = []

    # Find weakest constructs
    construct_labels = {
        "DigitalCapabilityScore":          "Kapabilitas Digital",
        "InnovationCapabilityScore":       "Kemampuan Inovasi",
        "EntrepreneurialOrientationScore": "Orientasi Kewirausahaan",
        "OrganizationalAgilityScore":      "Agilitas Organisasi",
        "ResourceAccessScore":             "Akses Sumber Daya",
        "EnvironmentalDynamismScore":      "Dinamika Lingkungan",
    }
    # Sort features by score (weakest first, exclude ED which is external)
    sorted_feats = sorted(
        [(k, v) for k, v in scores.items() if k in FEATURE_COLS and k != "EnvironmentalDynamismScore"],
        key=lambda x: x[1]
    )

    if pred == "High":
        recs.append("🏆 **Ketahanan Bisnis TINGGI** — UMKM Anda berada pada level terbaik!")
        recs.append("📈 Pertahankan keunggulan digital dan agilitias organisasi yang sudah baik.")
        recs.append("🌏 Pertimbangkan ekspansi pasar atau kemitraan strategis untuk pertumbuhan.")
        recs.append("💡 Jadilah mentor bagi UMKM lain di ekosistem bisnis Anda.")
        if sorted_feats and sorted_feats[0][1] < 3.5:
            recs.append(f"⚡ Tingkatkan {construct_labels.get(sorted_feats[0][0], '')} (skor: {sorted_feats[0][1]:.2f}) untuk mempertahankan keunggulan.")

    elif pred == "Medium":
        recs.append("⚠️ **Ketahanan Bisnis SEDANG** — Ada ruang signifikan untuk perbaikan.")
        if sorted_feats:
            weak1 = construct_labels.get(sorted_feats[0][0], "")
            recs.append(f"🎯 **Prioritas Utama:** Tingkatkan {weak1} (skor: {sorted_feats[0][1]:.2f}/5.00).")
        if len(sorted_feats) > 1:
            weak2 = construct_labels.get(sorted_feats[1][0], "")
            recs.append(f"🎯 **Prioritas Kedua:** Kembangkan {weak2} (skor: {sorted_feats[1][1]:.2f}/5.00).")
        if scores.get("ResourceAccessScore", 3) < 3.0:
            recs.append("💰 Cari akses KUR (Kredit Usaha Rakyat) atau program subsidi UMKM dari pemerintah.")
        if scores.get("DigitalCapabilityScore", 3) < 3.0:
            recs.append("📱 Ikuti pelatihan literasi digital dari Kemenkominfo atau platform online gratis.")
        recs.append("🤝 Bergabunglah dengan komunitas UMKM untuk berbagi sumber daya dan jaringan.")

    else:  # Low
        recs.append("🔴 **Ketahanan Bisnis RENDAH** — Diperlukan intervensi segera dan terstruktur.")
        recs.append("🆘 **Langkah Darurat:** Hubungi Dinas Koperasi & UMKM setempat untuk pendampingan.")
        recs.append("💳 Daftarkan usaha ke program KUR Mikro (plafon hingga Rp 100 juta, bunga 6%).")
        recs.append("📚 Ikuti program UMKM Go Digital dari Kementerian Perdagangan secara gratis.")
        if sorted_feats:
            weak1 = construct_labels.get(sorted_feats[0][0], "")
            recs.append(f"⚡ Fokuskan energi pada peningkatan {weak1} sebagai prioritas pertama.")
        recs.append("🏗️ Formalkan legalitas usaha (NIB via OSS.go.id) untuk akses pembiayaan yang lebih baik.")
        recs.append("🌐 Mulai dari satu kanal digital (WhatsApp Business) sebagai langkah pertama digitalisasi.")

    # Environmental dynamism advice
    ed_score = scores.get("EnvironmentalDynamismScore", 3.0)
    oa_score = scores.get("OrganizationalAgilityScore", 3.0)
    if ed_score > 3.5 and oa_score < 3.0:
        recs.append("⚡ Lingkungan bisnis Anda sangat dinamis namun agilitas organisasi rendah — ini risiko kritis yang harus segera ditangani.")

    return recs


def render_probability_chart(probs: dict):
    """Render a horizontal stacked probability bar."""
    low  = probs.get("Low",    0.0) * 100
    mid  = probs.get("Medium", 0.0) * 100
    high = probs.get("High",   0.0) * 100

    fig, ax = plt.subplots(figsize=(8, 1.2))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    ax.barh([0], [low],  color="#da3633", height=0.5, label=f"Low {low:.1f}%")
    ax.barh([0], [mid],  color="#d29922", height=0.5, left=low, label=f"Medium {mid:.1f}%")
    ax.barh([0], [high], color="#2ea043", height=0.5, left=low+mid, label=f"High {high:.1f}%")

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Probabilitas (%)", color="#8b949e", fontsize=9)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5),
                       ncol=3, frameon=False, fontsize=9,
                       labelcolor="#c9d1d9")
    plt.tight_layout()
    return fig


def render_radar_chart(scores: dict):
    """Render a radar chart of construct scores."""
    labels = ["Digital\nCapability", "Innovation\nCapability",
              "Entrepreneurial\nOrientation", "Organizational\nAgility",
              "Resource\nAccess", "Environmental\nDynamism"]
    values = [scores.get(f, 3.0) for f in FEATURE_COLS]
    values_plot = values + [values[0]]  # Close the polygon

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#1c2128")

    ax.plot(angles, values_plot, "o-", linewidth=2.5, color="#58a6ff")
    ax.fill(angles, values_plot, alpha=0.25, color="#58a6ff")

    # Reference circle at 3.0
    ref = [3.0] * len(angles)
    ax.plot(angles, ref, "--", linewidth=0.8, color="#8b949e", alpha=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color="#c9d1d9")
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1","2","3","4","5"], size=7, color="#8b949e")
    ax.grid(color="#30363d", alpha=0.5)
    ax.spines["polar"].set_visible(False)
    ax.set_title("Profil Kapabilitas UMKM", color="#e6edf3",
                 fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🏢 UMKM Resilience AI")
    st.markdown("---")

    # Model selector
    st.markdown("#### 🤖 Pilih Model AI")
    model_choice = st.selectbox(
        "Model Prediksi",
        options=["random_forest", "xgboost", "decision_tree", "logistic_regression"],
        format_func=lambda x: {
            "random_forest": "🌲 Random Forest",
            "xgboost": "⚡ XGBoost",
            "decision_tree": "🌿 Decision Tree",
            "logistic_regression": "📈 Logistic Regression",
        }.get(x, x),
        index=0,
        help="Pilih algoritma ML untuk prediksi ketahanan bisnis"
    )

    st.markdown("---")

    # Submission stats
    n_subs = get_submission_count()
    st.markdown("#### 📊 Statistik Data")
    st.metric("Total Responden", n_subs, help="Jumlah UMKM yang telah mengisi formulir")

    # Auto-retrain trigger
    RETRAIN_THRESHOLD = 10
    if n_subs > 0 and n_subs % RETRAIN_THRESHOLD == 0:
        if "last_retrain" not in st.session_state or st.session_state.get("last_retrain") != n_subs:
            st.session_state["last_retrain"] = n_subs
            st.markdown(f"""<div class="warn-box">
                🔄 <strong>{RETRAIN_THRESHOLD} data baru terkumpul!</strong><br>
                Model akan dilatih ulang otomatis.
            </div>""", unsafe_allow_html=True)
            retrain_model_background(n_new=RETRAIN_THRESHOLD)

    if st.session_state.get("retrain_msg"):
        st.success(st.session_state["retrain_msg"])
        del st.session_state["retrain_msg"]

    # Manual retrain
    st.markdown("#### 🔄 Latih Ulang Model")
    if st.button("🔄 Retrain Sekarang", help="Latih ulang model dengan semua data submission"):
        retrain_model_background(n_new=n_subs)
        st.info("Retraining dimulai di background...")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#8b949e; line-height:1.6;">
    📌 <strong>Panduan Skala Likert:</strong><br>
    1 = Sangat Tidak Setuju<br>
    2 = Tidak Setuju<br>
    3 = Netral<br>
    4 = Setuju<br>
    5 = Sangat Setuju
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("v2.0.0 | UMKM AI Research | 2026")


# ══════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <h1>🏢 Sistem Prediksi Ketahanan Bisnis UMKM</h1>
    <p>
        Berbasis Kecerdasan Buatan (AI) &nbsp;|&nbsp;
        Isi formulir di bawah &rarr; Klik <strong>Generate Prediksi</strong> &rarr;
        Dapatkan hasil analisis & rekomendasi instan
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    ℹ️ <strong>Petunjuk Pengisian:</strong> Isi semua bagian formulir di bawah ini dengan jujur sesuai kondisi UMKM Anda.
    Untuk pertanyaan kuesioner, pilih angka 1–5 yang paling mencerminkan kondisi usaha Anda saat ini.
    Data Anda akan digunakan untuk meningkatkan akurasi model AI secara berkelanjutan.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# FORM
# ══════════════════════════════════════════════════════════

with st.form("umkm_form", clear_on_submit=False):

    # ── BAGIAN 1: PROFIL UMKM ─────────────────────────────
    st.markdown('<div class="section-header">📋 Bagian 1: Profil UMKM</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        business_name = st.text_input(
            "Nama UMKM *",
            placeholder="Contoh: Batik Nusantara Indah",
            help="Apa nama UMKM Anda?"
        )
        province = st.selectbox(
            "Provinsi *",
            options=["-- Pilih Provinsi --",
                     "Aceh","Sumatera Utara","Sumatera Barat","Riau","Kepulauan Riau",
                     "Jambi","Bengkulu","Sumatera Selatan","Kepulauan Bangka Belitung","Lampung",
                     "DKI Jakarta","Jawa Barat","Banten","Jawa Tengah","DI Yogyakarta","Jawa Timur",
                     "Bali","Nusa Tenggara Barat","Nusa Tenggara Timur",
                     "Kalimantan Barat","Kalimantan Tengah","Kalimantan Selatan",
                     "Kalimantan Timur","Kalimantan Utara",
                     "Sulawesi Utara","Gorontalo","Sulawesi Tengah","Sulawesi Barat",
                     "Sulawesi Selatan","Sulawesi Tenggara",
                     "Maluku","Maluku Utara","Papua Barat","Papua",
                     "Papua Selatan","Papua Tengah","Papua Pegunungan","Papua Barat Daya"]
        )
        city = st.text_input(
            "Kota/Kabupaten *",
            placeholder="Contoh: Bandung",
            help="Di kota/kabupaten manakah UMKM Anda beroperasi?"
        )

    with col2:
        business_sector = st.selectbox(
            "Sektor Usaha *",
            options=["-- Pilih Sektor --",
                     "Kuliner & Makanan-Minuman",
                     "Fashion & Tekstil",
                     "Kerajinan Tangan & Seni",
                     "Teknologi & Digital",
                     "Pertanian & Pangan",
                     "Perdagangan Eceran",
                     "Jasa & Konsultasi",
                     "Manufaktur & Produksi",
                     "Kesehatan & Kecantikan",
                     "Pendidikan & Pelatihan",
                     "Pariwisata & Perhotelan",
                     "Konstruksi & Properti",
                     "Transportasi & Logistik",
                     "Lainnya"]
        )
        business_age = st.number_input(
            "Usia Usaha (Tahun) *",
            min_value=0.0, max_value=50.0, value=3.0, step=0.5,
            help="Berapa usia usaha Anda (dalam tahun)?"
        )
        num_employees = st.number_input(
            "Jumlah Karyawan *",
            min_value=0, max_value=300, value=5, step=1,
            help="Berapa jumlah karyawan yang saat ini bekerja di UMKM Anda?"
        )

    with col3:
        annual_revenue = st.selectbox(
            "Omzet Tahunan (Rp) *",
            options=["-- Pilih Kisaran --",
                     "< Rp 50 Juta",
                     "Rp 50–100 Juta",
                     "Rp 100–300 Juta",
                     "Rp 300–500 Juta",
                     "Rp 500 Juta – 1 Miliar",
                     "Rp 1–2,5 Miliar",
                     "Rp 2,5–5 Miliar",
                     "> Rp 5 Miliar"]
        )
        digital_pct = st.slider(
            "Persentase Penjualan Digital (%)",
            min_value=0, max_value=100, value=20, step=5,
            help="Berapa % penjualan dari kanal digital (marketplace, medsos, website, dll.)?"
        )
        owner_age = st.number_input(
            "Usia Pemilik *",
            min_value=17, max_value=80, value=35, step=1
        )

    col4, col5 = st.columns(2)
    with col4:
        owner_gender = st.radio(
            "Jenis Kelamin Pemilik *",
            options=["Laki-laki", "Perempuan"],
            horizontal=True
        )
        education = st.selectbox(
            "Pendidikan Terakhir Pemilik *",
            options=["-- Pilih Pendidikan --",
                     "SD / SMP",
                     "SMA / SMK",
                     "Diploma (D1–D3)",
                     "Sarjana (S1)",
                     "Magister (S2)",
                     "Doktor (S3)"]
        )
    with col5:
        legal_status = st.selectbox(
            "Status Legalitas Usaha *",
            options=["-- Pilih Status --",
                     "Belum berbadan hukum (informal)",
                     "Usaha Dagang (UD)",
                     "CV (Commanditaire Vennootschap)",
                     "PT (Perseroan Terbatas)",
                     "Koperasi",
                     "Yayasan / Perkumpulan"]
        )

    st.markdown("---")

    # Helper: render Likert question
    def likert_q(label: str, key: str, question: str, default: int = 3) -> int:
        st.markdown(f"**{label}** — *{question}*")
        val = st.radio(
            label, options=[1,2,3,4,5],
            format_func=lambda x: LIKERT_LABELS[x],
            index=default - 1,
            horizontal=True,
            key=key,
            label_visibility="collapsed"
        )
        return val

    # ── BAGIAN 2: KAPABILITAS DIGITAL ─────────────────────
    st.markdown('<div class="section-header">💻 Bagian 2: Kapabilitas Digital</div>', unsafe_allow_html=True)

    dc1 = likert_q("DC1","dc1_q","UMKM kami menggunakan teknologi digital dalam kegiatan operasional sehari-hari.")
    dc2 = likert_q("DC2","dc2_q","UMKM kami memanfaatkan media digital (media sosial, marketplace, atau website) untuk menjalankan bisnis.")
    dc3 = likert_q("DC3","dc3_q","UMKM kami menggunakan data digital sebagai dasar dalam pengambilan keputusan bisnis.")
    dc4 = likert_q("DC4","dc4_q","Teknologi digital telah terintegrasi dengan baik dalam proses bisnis UMKM kami.")
    dc5 = likert_q("DC5","dc5_q","UMKM kami mampu mempelajari dan mengadopsi teknologi digital baru dengan cepat.")
    st.markdown("---")

    # ── BAGIAN 3: KEMAMPUAN INOVASI ───────────────────────
    st.markdown('<div class="section-header">💡 Bagian 3: Kemampuan Inovasi</div>', unsafe_allow_html=True)

    ic1 = likert_q("IC1","ic1_q","UMKM kami secara rutin mengembangkan produk atau layanan baru.")
    ic2 = likert_q("IC2","ic2_q","UMKM kami terus melakukan perbaikan terhadap proses operasional.")
    ic3 = likert_q("IC3","ic3_q","UMKM kami menerapkan cara pemasaran yang baru dan kreatif.")
    ic4 = likert_q("IC4","ic4_q","UMKM kami mampu menyesuaikan produk sesuai kebutuhan pelanggan.")
    ic5 = likert_q("IC5","ic5_q","Ide-ide baru dapat diterapkan dengan cepat di UMKM kami.")
    st.markdown("---")

    # ── BAGIAN 4: ORIENTASI KEWIRAUSAHAAN ─────────────────
    st.markdown('<div class="section-header">🚀 Bagian 4: Orientasi Kewirausahaan</div>', unsafe_allow_html=True)

    eo1 = likert_q("EO1","eo1_q","UMKM kami aktif mencari peluang usaha baru sebelum pesaing.")
    eo2 = likert_q("EO2","eo2_q","UMKM kami berani mengambil risiko yang telah diperhitungkan untuk mengembangkan usaha.")
    eo3 = likert_q("EO3","eo3_q","UMKM kami selalu berupaya menciptakan pembaruan dalam bisnis.")
    eo4 = likert_q("EO4","eo4_q","UMKM kami secara aktif bersaing untuk meningkatkan posisi di pasar.")
    eo5 = likert_q("EO5","eo5_q","UMKM kami memiliki kebebasan dalam mengambil keputusan strategis.")
    st.markdown("---")

    # ── BAGIAN 5: AGILITAS ORGANISASI ─────────────────────
    st.markdown('<div class="section-header">⚡ Bagian 5: Agilitas Organisasi</div>', unsafe_allow_html=True)

    oa1 = likert_q("OA1","oa1_q","UMKM kami mampu mendeteksi perubahan kebutuhan pasar dengan cepat.")
    oa2 = likert_q("OA2","oa2_q","UMKM kami mampu mengambil keputusan bisnis dengan cepat ketika terjadi perubahan.")
    oa3 = likert_q("OA3","oa3_q","UMKM kami dapat menyesuaikan proses bisnis dengan cepat sesuai kondisi yang berubah.")
    oa4 = likert_q("OA4","oa4_q","UMKM kami mampu mengalokasikan kembali sumber daya secara cepat ketika diperlukan.")
    oa5 = likert_q("OA5","oa5_q","UMKM kami mampu merespons kebutuhan pelanggan dengan cepat.")
    st.markdown("---")

    # ── BAGIAN 6: AKSES SUMBER DAYA ──────────────────────
    st.markdown('<div class="section-header">💰 Bagian 6: Akses Sumber Daya</div>', unsafe_allow_html=True)

    ra1 = likert_q("RA1","ra1_q","UMKM kami memiliki akses yang memadai terhadap sumber pembiayaan usaha.")
    ra2 = likert_q("RA2","ra2_q","UMKM kami memiliki sumber daya manusia yang kompeten.")
    ra3 = likert_q("RA3","ra3_q","UMKM kami memiliki akses yang stabil terhadap bahan baku atau sumber daya utama.")
    ra4 = likert_q("RA4","ra4_q","UMKM kami memiliki jaringan kerja sama bisnis yang mendukung perkembangan usaha.")
    ra5 = likert_q("RA5","ra5_q","UMKM kami mudah memperoleh informasi mengenai kondisi pasar.")
    st.markdown("---")

    # ── BAGIAN 7: DINAMIKA LINGKUNGAN ─────────────────────
    st.markdown('<div class="section-header">🌍 Bagian 7: Dinamika Lingkungan</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
        ℹ️ Bagian ini mengukur seberapa <strong>dinamis lingkungan bisnis</strong> Anda.
        Nilai tinggi berarti lingkungan bisnis Anda <em>sangat berubah-ubah</em>.
    </div>""", unsafe_allow_html=True)

    ed1 = likert_q("ED1","ed1_q","Permintaan pelanggan terhadap produk/jasa UMKM kami sering mengalami perubahan.")
    ed2 = likert_q("ED2","ed2_q","Tingkat persaingan di lingkungan usaha kami berubah dengan cepat.")
    ed3 = likert_q("ED3","ed3_q","Perkembangan teknologi di sektor usaha kami berlangsung dengan cepat.")
    ed4 = likert_q("ED4","ed4_q","Kondisi ekonomi yang memengaruhi usaha kami sulit diprediksi.")
    ed5 = likert_q("ED5","ed5_q","Ketersediaan bahan baku atau pasokan usaha kami sering mengalami perubahan.")
    st.markdown("---")

    # ── BAGIAN 8: KETAHANAN BISNIS (BR) ──────────────────
    st.markdown('<div class="section-header">🛡️ Bagian 8: Ketahanan Bisnis (Self-Assessment)</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
        ℹ️ Bagian ini digunakan untuk <strong>validasi model AI</strong>.
        Jawablah sesuai persepsi Anda tentang ketahanan bisnis UMKM Anda sendiri.
    </div>""", unsafe_allow_html=True)

    br1 = likert_q("BR1","br1_q","UMKM kami mampu mempertahankan operasional meskipun menghadapi gangguan.")
    br2 = likert_q("BR2","br2_q","UMKM kami mampu beradaptasi terhadap perubahan lingkungan bisnis.")
    br3 = likert_q("BR3","br3_q","UMKM kami mampu pulih dengan cepat setelah mengalami gangguan usaha.")
    br4 = likert_q("BR4","br4_q","UMKM kami mampu mempertahankan pelanggan dalam berbagai kondisi.")
    br5 = likert_q("BR5","br5_q","UMKM kami mampu menjaga kestabilan pendapatan meskipun menghadapi tantangan.")
    br6 = likert_q("BR6","br6_q","UMKM kami mampu memanfaatkan peluang bisnis baru setelah terjadi perubahan lingkungan.")
    br7 = likert_q("BR7","br7_q","UMKM kami memiliki kesiapan dalam menghadapi berbagai risiko bisnis.")
    st.markdown("---")

    # ── SUBMIT BUTTON ──────────────────────────────────────
    st.markdown("### 🔮 Generate Prediksi Ketahanan Bisnis")
    st.markdown("""<div class="info-box">
        Pastikan semua field telah diisi dengan lengkap sebelum menekan tombol di bawah.
    </div>""", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🔮 GENERATE PREDIKSI KETAHANAN BISNIS SAYA",
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════
# PREDICTION RESULTS
# ══════════════════════════════════════════════════════════

if submitted:
    # Validation
    errors = []
    if not business_name or business_name.strip() == "":
        errors.append("Nama UMKM harus diisi.")
    if province == "-- Pilih Provinsi --":
        errors.append("Provinsi harus dipilih.")
    if not city or city.strip() == "":
        errors.append("Kota/Kabupaten harus diisi.")
    if business_sector == "-- Pilih Sektor --":
        errors.append("Sektor Usaha harus dipilih.")
    if annual_revenue == "-- Pilih Kisaran --":
        errors.append("Omzet Tahunan harus dipilih.")
    if education == "-- Pilih Pendidikan --":
        errors.append("Pendidikan Terakhir harus dipilih.")
    if legal_status == "-- Pilih Status --":
        errors.append("Status Legalitas Usaha harus dipilih.")

    if errors:
        for err in errors:
            st.error(f"❌ {err}")
        st.stop()

    # ── Load model ─────────────────────────────────────────
    with st.spinner("🤖 Memuat model AI dan memproses prediksi..."):
        model, scaler, label_encoder, feat_cols = load_model(model_choice)

    if model is None:
        st.error(f"❌ Model '{model_choice}' tidak ditemukan di folder model/. Jalankan `python run_pipeline.py` terlebih dahulu.")
        st.stop()

    # ── Collect all responses ──────────────────────────────
    responses = {
        "DC1":dc1,"DC2":dc2,"DC3":dc3,"DC4":dc4,"DC5":dc5,
        "IC1":ic1,"IC2":ic2,"IC3":ic3,"IC4":ic4,"IC5":ic5,
        "EO1":eo1,"EO2":eo2,"EO3":eo3,"EO4":eo4,"EO5":eo5,
        "OA1":oa1,"OA2":oa2,"OA3":oa3,"OA4":oa4,"OA5":oa5,
        "RA1":ra1,"RA2":ra2,"RA3":ra3,"RA4":ra4,"RA5":ra5,
        "ED1":ed1,"ED2":ed2,"ED3":ed3,"ED4":ed4,"ED5":ed5,
        "BR1":br1,"BR2":br2,"BR3":br3,"BR4":br4,"BR5":br5,"BR6":br6,"BR7":br7,
    }

    # Compute construct scores
    scores = compute_construct_scores(responses)

    # Predict
    result = predict_resilience(model, scaler, label_encoder, scores)

    # Save to CSV log
    profile = {
        "Business_Name": business_name.strip(),
        "Province": province,
        "City": city.strip(),
        "Business_Sector": business_sector,
        "Business_Age": business_age,
        "Number_of_Employees": num_employees,
        "Annual_Revenue": annual_revenue,
        "Digital_Sales_Percentage": digital_pct,
        "Owner_Age": owner_age,
        "Owner_Gender": owner_gender,
        "Education": education,
        "Legal_Status": legal_status,
    }
    log_path = save_submission(profile, responses, scores, result)
    n_total  = get_submission_count()

    # ── SIMPAN SEMUA HASIL KE SESSION STATE ────────────────
    # Agar data tetap tersedia saat tombol Gemini diklik (rerun)
    st.session_state["prediction_done"] = True
    st.session_state["pred_result"] = result
    st.session_state["pred_scores"] = scores
    st.session_state["pred_business_name"] = business_name.strip()
    st.session_state["pred_business_sector"] = business_sector
    st.session_state["pred_province"] = province
    st.session_state["pred_city"] = city.strip()
    st.session_state["pred_model_choice"] = model_choice
    st.session_state["pred_n_total"] = n_total

    # Auto-retrain check
    if n_total > 0 and n_total % 10 == 0:
        st.info(f"🔄 Mencapai **{n_total}** responden — model sedang dilatih ulang secara otomatis di background...")
        retrain_model_background(n_new=10)


# ══════════════════════════════════════════════════════════
# DISPLAY RESULTS (dari session_state agar tetap ada saat rerun)
# ══════════════════════════════════════════════════════════

if st.session_state.get("prediction_done"):
    result         = st.session_state["pred_result"]
    scores         = st.session_state["pred_scores"]
    business_name  = st.session_state["pred_business_name"]
    business_sector = st.session_state["pred_business_sector"]
    province       = st.session_state["pred_province"]
    model_choice_display = st.session_state["pred_model_choice"]
    n_total        = st.session_state["pred_n_total"]

    pred       = result["predicted_class"]
    confidence = result["confidence"]
    probs      = result["probabilities"]
    br_score   = result["br_score"]

    # ── RESULTS DISPLAY ────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Hasil Prediksi Ketahanan Bisnis UMKM Anda")

    # Result banner
    color_class = {"High": "result-high", "Medium": "result-medium", "Low": "result-low"}.get(pred, "result-medium")
    emoji_map   = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    label_id    = {"High": "TINGGI", "Medium": "SEDANG", "Low": "RENDAH"}
    color_hex   = {"High": "#2ea043", "Medium": "#d29922", "Low": "#da3633"}

    st.markdown(f"""
    <div class="{color_class}">
        <div class="result-sublabel">Hasil Prediksi untuk <strong>{business_name}</strong></div>
        <div class="result-label" style="color:{color_hex[pred]}">
            {emoji_map[pred]} {label_id[pred]}
        </div>
        <div class="result-sublabel">Level Ketahanan Bisnis</div>
        <div style="font-size:0.95rem; margin-top:0.8rem; color:#c9d1d9;">
            Model: <strong>{model_choice_display.replace('_',' ').title()}</strong> &nbsp;|&nbsp;
            Kepercayaan: <strong>{confidence:.1%}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics row ────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{color_hex[pred]}">{label_id[pred]}</div>
            <div class="metric-label">Kategori Ketahanan</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{confidence:.1%}</div>
            <div class="metric-label">Kepercayaan Model</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{br_score:.2f}</div>
            <div class="metric-label">Estimasi Skor BR (/ 5.0)</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">#{n_total}</div>
            <div class="metric-label">Anda Responden ke-</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.markdown("#### 📈 Distribusi Probabilitas Kelas")
        fig_prob = render_probability_chart(probs)
        st.pyplot(fig_prob, use_container_width=True)
        plt.close()

        # Probability table
        prob_data = {
            "Kategori": ["🔴 Rendah (Low)", "🟡 Sedang (Medium)", "🟢 Tinggi (High)"],
            "Probabilitas": [
                f"{probs.get('Low',0):.1%}",
                f"{probs.get('Medium',0):.1%}",
                f"{probs.get('High',0):.1%}",
            ]
        }
        st.dataframe(pd.DataFrame(prob_data), hide_index=True, use_container_width=True)

    with col_chart2:
        st.markdown("#### 🕸️ Radar Kapabilitas UMKM")
        fig_radar = render_radar_chart(scores)
        st.pyplot(fig_radar, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Construct scores breakdown ─────────────────────────
    st.markdown("#### 📊 Detail Skor Konstruk")
    score_labels = {
        "DigitalCapabilityScore":          "💻 Kapabilitas Digital",
        "InnovationCapabilityScore":       "💡 Kemampuan Inovasi",
        "EntrepreneurialOrientationScore": "🚀 Orientasi Kewirausahaan",
        "OrganizationalAgilityScore":      "⚡ Agilitas Organisasi",
        "ResourceAccessScore":             "💰 Akses Sumber Daya",
        "EnvironmentalDynamismScore":      "🌍 Dinamika Lingkungan",
        "BusinessResilienceScore":         "🛡️ Ketahanan Bisnis (Self)",
    }
    score_cols = st.columns(4)
    for idx, (key, label) in enumerate(score_labels.items()):
        val = scores.get(key, 0.0)
        bar_w = int(val / 5.0 * 100)
        bar_color = "#2ea043" if val >= 3.5 else ("#d29922" if val >= 2.5 else "#da3633")
        with score_cols[idx % 4]:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.78rem;color:#8b949e;margin-bottom:0.4rem">{label}</div>
                <div class="metric-value" style="color:{bar_color};font-size:1.5rem">{val:.2f}</div>
                <div style="background:#21262d;border-radius:4px;height:6px;margin-top:6px;">
                    <div style="background:{bar_color};width:{bar_w}%;height:6px;border-radius:4px;"></div>
                </div>
                <div style="font-size:0.7rem;color:#8b949e;margin-top:2px;">/ 5.00</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommendations ────────────────────────────────────
    st.markdown("#### 💡 Rekomendasi Strategis")
    recommendations = get_recommendations(result, scores)
    for rec in recommendations:
        st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download result ────────────────────────────────────
    st.markdown("#### 📥 Unduh Laporan")
    result_df = pd.DataFrame([{
        "UMKM": business_name,
        "Provinsi": province,
        "Kota": st.session_state.get("pred_city", ""),
        "Sektor": business_sector,
        "Prediksi": pred,
        "Kategori": label_id[pred],
        "Kepercayaan": f"{confidence:.1%}",
        "Skor BR Estimasi": br_score,
        **{score_labels.get(k, k): f"{v:.2f}" for k, v in scores.items()},
        "Tanggal": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Model": model_choice_display,
    }])
    csv_bytes = result_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 Download Hasil Prediksi (CSV)",
        data=csv_bytes,
        file_name=f"prediksi_{business_name.replace(' ','_')}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.success(f"✅ Data UMKM **{business_name}** berhasil disimpan sebagai responden ke-**{n_total}**. Terima kasih telah berkontribusi dalam penelitian ini!")

    # ── GEMINI AI RECOMMENDATION ───────────────────────────
    if GEMINI_MODULE_OK:
        render_gemini_section(
            business_name   = business_name,
            business_sector = business_sector,
            province        = province,
            result          = result,
            scores          = scores,
        )
    else:
        st.markdown("---")
        st.info("💡 **Tip:** Tambahkan API Key Gemini di file `.env` untuk mendapatkan rekomendasi AI yang lebih mendalam.")

