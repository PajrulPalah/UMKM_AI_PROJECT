# ============================================================
# gemini_helper.py
# Google Gemini AI Integration untuk UMKM Resilience System
# ============================================================
#
# PROMPT VERSION CONFIG
# Ganti nilai di bawah untuk memilih versi prompt:
#   "v1" → Prompt sederhana (versi awal, ringkas)
#   "v2" → Prompt SWOT lengkap + evidence-based (versi baru)
# ============================================================

PROMPT_VERSION = "v2"   # <- Ganti ke "v1" untuk kembali ke versi lama

import os
import streamlit as st
from pathlib import Path

# ── Coba import google-generativeai ────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ============================================================
# INDICATOR LABEL MAPS
# ============================================================

DC_LABELS = {
    "DC1": "Penggunaan Teknologi Digital",
    "DC2": "Pemanfaatan Media Digital",
    "DC3": "Pemanfaatan Data Digital",
    "DC4": "Integrasi Teknologi Digital",
    "DC5": "Kemampuan Mengadopsi Teknologi Baru",
}
IC_LABELS = {
    "IC1": "Pengembangan Produk/Layanan Baru",
    "IC2": "Eksperimen dan Uji Coba",
    "IC3": "Adopsi Inovasi Terbaik",
    "IC4": "Kolaborasi untuk Inovasi",
    "IC5": "Orientasi pada Tren Baru",
}
EO_LABELS = {
    "EO1": "Keberanian Mengambil Risiko",
    "EO2": "Proaktif Mencari Peluang",
    "EO3": "Inovasi dalam Operasional",
    "EO4": "Kemandirian dan Otonomi",
    "EO5": "Kompetitivitas",
}
OA_LABELS = {
    "OA1": "Kecepatan Respons terhadap Perubahan",
    "OA2": "Fleksibilitas Organisasi",
    "OA3": "Kemampuan Adaptasi Cepat",
    "OA4": "Koordinasi Lintas Fungsi",
    "OA5": "Pembelajaran Organisasional",
}
RA_LABELS = {
    "RA1": "Akses Modal/Pendanaan",
    "RA2": "Akses SDM/Tenaga Kerja",
    "RA3": "Akses Jaringan Bisnis",
    "RA4": "Akses Informasi dan Pengetahuan",
    "RA5": "Akses Teknologi dan Infrastruktur",
}
ED_LABELS = {
    "ED1": "Dinamika Pasar",
    "ED2": "Intensitas Persaingan",
    "ED3": "Perubahan Teknologi",
    "ED4": "Perubahan Regulasi",
    "ED5": "Perubahan Perilaku Konsumen",
}
BR_LABELS = {
    "BR1": "Kemampuan Bertahan",
    "BR2": "Kemampuan Beradaptasi",
    "BR3": "Kemampuan Pemulihan",
    "BR4": "Kemampuan Mempertahankan Pelanggan",
    "BR5": "Stabilitas Pendapatan",
    "BR6": "Kemampuan Menangkap Peluang Baru",
    "BR7": "Kesiapan Menghadapi Risiko",
}


def _fmt_indicators(label_map: dict, indicators: dict) -> str:
    """Format satu grup indikator menjadi baris teks."""
    lines = []
    for code, label in label_map.items():
        val = indicators.get(code, "N/A")
        val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
        lines.append(f"  {code} ({label}): {val_str}")
    return "\n".join(lines)


def get_api_key() -> str | None:
    """
    Ambil API key dari beberapa sumber (prioritas dari atas ke bawah):
    1. st.secrets["GEMINI_API_KEY"]     ← .streamlit/secrets.toml
    2. Environment variable GEMINI_API_KEY
    3. File .env di root proyek
    """
    # Sumber 1: Streamlit secrets (untuk production/cloud)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # Sumber 2: Environment variable
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key

    # Sumber 3: File .env (untuk development lokal)
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    return None


def is_gemini_ready() -> bool:
    """Cek apakah Gemini siap digunakan."""
    return GEMINI_AVAILABLE and bool(get_api_key())


# ============================================================
# PROMPT V1  (versi lama — sederhana & ringkas)
# ============================================================

def build_umkm_prompt_v1(
    business_name: str,
    business_sector: str,
    province: str,
    predicted_class: str,
    confidence: float,
    br_score: float,
    scores: dict,
    probabilities: dict,
    indicators: dict = None,   # tidak digunakan di v1, hanya untuk kompatibilitas
    profile: dict = None,      # tidak digunakan di v1, hanya untuk kompatibilitas
) -> str:
    """
    PROMPT V1 — Versi awal yang sederhana.
    Memberikan ringkasan kondisi + rekomendasi 6 bagian dalam Bahasa Indonesia.
    Disimpan agar bisa dibandingkan dengan v2.
    """

    score_lines = "\n".join([
        f"  - Kapabilitas Digital (DC):          {scores.get('DigitalCapabilityScore', 0):.2f} / 5.00",
        f"  - Kemampuan Inovasi (IC):             {scores.get('InnovationCapabilityScore', 0):.2f} / 5.00",
        f"  - Orientasi Kewirausahaan (EO):       {scores.get('EntrepreneurialOrientationScore', 0):.2f} / 5.00",
        f"  - Agilitas Organisasi (OA):           {scores.get('OrganizationalAgilityScore', 0):.2f} / 5.00",
        f"  - Akses Sumber Daya (RA):             {scores.get('ResourceAccessScore', 0):.2f} / 5.00",
        f"  - Dinamika Lingkungan (ED):           {scores.get('EnvironmentalDynamismScore', 0):.2f} / 5.00",
        f"  - Ketahanan Bisnis Self-Assessment:   {scores.get('BusinessResilienceScore', 0):.2f} / 5.00",
    ])

    # Identifikasi 2 konstruk terlemah
    feature_map = {
        "DigitalCapabilityScore":          "Kapabilitas Digital",
        "InnovationCapabilityScore":       "Kemampuan Inovasi",
        "EntrepreneurialOrientationScore": "Orientasi Kewirausahaan",
        "OrganizationalAgilityScore":      "Agilitas Organisasi",
        "ResourceAccessScore":             "Akses Sumber Daya",
    }
    sorted_scores = sorted(
        [(label, scores.get(k, 3.0)) for k, label in feature_map.items()],
        key=lambda x: x[1]
    )
    weak1 = f"{sorted_scores[0][0]} (skor: {sorted_scores[0][1]:.2f})"
    weak2 = f"{sorted_scores[1][0]} (skor: {sorted_scores[1][1]:.2f})"

    label_map  = {"Low": "RENDAH", "Medium": "SEDANG", "High": "TINGGI"}
    label_text = label_map.get(predicted_class, predicted_class)

    prompt = f"""
Anda adalah konsultan bisnis senior spesialis UMKM Indonesia dengan pengalaman 20 tahun.
Berikan analisis mendalam dan rekomendasi strategis yang KONKRET, SPESIFIK, dan ACTIONABLE
dalam Bahasa Indonesia yang profesional namun mudah dipahami pelaku UMKM.

=== DATA UMKM ===
Nama UMKM     : {business_name}
Sektor Usaha  : {business_sector}
Lokasi        : {province}

=== HASIL PREDIKSI AI ===
Kategori Ketahanan Bisnis : {label_text} ({predicted_class})
Kepercayaan Model          : {confidence:.1%}
Estimasi Skor BR           : {br_score:.2f} / 5.00
Probabilitas Low/Med/High  : {probabilities.get('Low',0):.1%} / {probabilities.get('Medium',0):.1%} / {probabilities.get('High',0):.1%}

=== SKOR 7 KONSTRUK ===
{score_lines}

=== AREA TERLEMAH ===
1. {weak1}
2. {weak2}

=== INSTRUKSI ANALISIS ===
Buatlah laporan rekomendasi dengan format PERSIS sebagai berikut:

**🔍 RINGKASAN KONDISI UMKM**
[2-3 kalimat deskripsi kondisi saat ini berdasarkan data di atas, sebutkan nama UMKM dan sektornya]

**⚠️ AREA KRITIS YANG PERLU DITANGANI**
[Identifikasi 2-3 masalah utama berdasarkan skor terendah, jelaskan MENGAPA ini menjadi masalah untuk sektor {business_sector}]

**🎯 RENCANA AKSI 30 HARI PERTAMA**
[3 langkah konkret yang bisa langsung dilakukan minggu ini, spesifik untuk {business_sector} di {province}]

**📈 STRATEGI JANGKA MENENGAH (3-6 BULAN)**
[2-3 strategi untuk meningkatkan kategori ketahanan bisnis dari {label_text} ke level lebih tinggi]

**💰 PROGRAM PEMERINTAH YANG RELEVAN**
[2-3 program bantuan/subsidi pemerintah Indonesia yang cocok untuk kondisi UMKM ini, sertakan cara akses]

**🏆 TARGET PENINGKATAN SKOR**
[Skor target realistis untuk setiap konstruk dalam 6 bulan ke depan, dalam format tabel sederhana]

Pastikan rekomendasi:
- Spesifik untuk sektor {business_sector}
- Mempertimbangkan kondisi di {province}
- Menggunakan bahasa yang mudah dipahami pengusaha UMKM
- Mencantumkan langkah konkret yang bisa langsung dieksekusi
- TIDAK menggunakan istilah teknis yang rumit
""".strip()

    return prompt


# ============================================================
# PROMPT V2  (versi baru — SWOT lengkap + evidence-based)
# ============================================================

def build_umkm_prompt_v2(
    business_name: str,
    business_sector: str,
    province: str,
    predicted_class: str,
    confidence: float,
    br_score: float,
    scores: dict,
    probabilities: dict,
    indicators: dict = None,
    profile: dict = None,
) -> str:
    """
    PROMPT V2 — Versi baru: SWOT framework lengkap, evidence-based, anti-hallucination.
    Membutuhkan data individual indicators (DC1-BR7) dan profil bisnis lengkap
    untuk menghasilkan analisis yang benar-benar spesifik per UMKM.
    """
    ind  = indicators or {}
    prof = profile or {}

    label_map  = {"Low": "LOW", "Medium": "MEDIUM", "High": "HIGH"}
    label_text = label_map.get(predicted_class, predicted_class)

    # Format construct scores
    construct_block = "\n".join([
        f"  DigitalCapabilityScore:          {scores.get('DigitalCapabilityScore', 'N/A')}",
        f"  InnovationCapabilityScore:        {scores.get('InnovationCapabilityScore', 'N/A')}",
        f"  EntrepreneurialOrientationScore:  {scores.get('EntrepreneurialOrientationScore', 'N/A')}",
        f"  OrganizationalAgilityScore:       {scores.get('OrganizationalAgilityScore', 'N/A')}",
        f"  ResourceAccessScore:              {scores.get('ResourceAccessScore', 'N/A')}",
        f"  EnvironmentalDynamismScore:       {scores.get('EnvironmentalDynamismScore', 'N/A')}",
        f"  BusinessResilienceScore:          {scores.get('BusinessResilienceScore', br_score)}",
    ])

    # Format individual indicators
    dc_block = _fmt_indicators(DC_LABELS, ind)
    ic_block = _fmt_indicators(IC_LABELS, ind)
    eo_block = _fmt_indicators(EO_LABELS, ind)
    oa_block = _fmt_indicators(OA_LABELS, ind)
    ra_block = _fmt_indicators(RA_LABELS, ind)
    ed_block = _fmt_indicators(ED_LABELS, ind)
    br_block = _fmt_indicators(BR_LABELS, ind)

    # Business profile
    def _p(key, default="N/A"):
        v = prof.get(key, default)
        return str(v) if v not in (None, "", "nan") else "N/A"

    profile_block = "\n".join([
        f"  Business Name      : {business_name}",
        f"  Province           : {province}",
        f"  City               : {_p('city')}",
        f"  Business Sector    : {business_sector}",
        f"  Business Age       : {_p('business_age')} years",
        f"  Number of Employees: {_p('num_employees')}",
        f"  Annual Revenue     : {_p('annual_revenue')}",
        f"  Digital Sales %    : {_p('digital_sales_pct')}",
        f"  Owner Age          : {_p('owner_age')}",
        f"  Owner Gender       : {_p('owner_gender')}",
        f"  Education          : {_p('education')}",
        f"  Legal Status       : {_p('legal_status')}",
    ])

    prompt = f"""
Anda adalah tim ahli yang terdiri dari:
- Senior SME Business Analyst
- Business Resilience Consultant
- Strategic Management Consultant
- SWOT Analysis Specialist
- Machine Learning Interpretation Specialist

Tugas Anda: Analisis UMKM di bawah ini HANYA menggunakan data yang tersedia.
JANGAN menciptakan fakta, kondisi, pasar, pesaing, atau peluang yang tidak ada dalam data.

SETIAP pernyataan analitis HARUS dapat ditelusuri ke field data yang tersedia.

Prediksi Machine Learning bersifat FINAL. JANGAN mengubah atau menghitung ulang prediksi.

Tulis seluruh laporan dalam BAHASA INDONESIA. Gunakan bahasa formal namun mudah dipahami pemilik UMKM.

============================================================
DATA UMKM
============================================================

LEVEL 1 - PROFIL BISNIS
{profile_block}

LEVEL 2 - INDIKATOR KUESIONER INDIVIDUAL (Skala 1-5)

[Digital Capability]
{dc_block}

[Innovation Capability]
{ic_block}

[Entrepreneurial Orientation]
{eo_block}

[Organizational Agility]
{oa_block}

[Resource Access]
{ra_block}

[Environmental Dynamism]
{ed_block}

[Business Resilience Self-Assessment]
{br_block}

LEVEL 3 - SKOR KONSTRUK (Skala 1-5)
{construct_block}

LEVEL 4 - OUTPUT MACHINE LEARNING
  Predicted Category : {label_text}
  Confidence         : {confidence:.1%}
  Probability Low    : {probabilities.get('Low', 0):.1%}
  Probability Medium : {probabilities.get('Medium', 0):.1%}
  Probability High   : {probabilities.get('High', 0):.1%}

============================================================
ATURAN ANALITIS (WAJIB DIIKUTI)
============================================================

1. Setiap pernyataan utama harus mengutip kode indikator atau skor spesifik dari data di atas.
2. Bedakan antara:
   A. DATA TERAMATI - langsung dinyatakan dalam input
   B. INTERPRETASI ANALITIS - kesimpulan wajar dari data
   C. IMPLIKASI STRATEGIS - tindakan bisnis yang berasal dari analisis
   Jangan menyajikan (C) seolah-olah itu adalah (A).
3. JANGAN menciptakan: program pemerintah, tren pasar, kelemahan pesaing, peluang eksternal,
   opsi pembiayaan - kecuali ada dalam data.
4. Jika bukti peluang eksternal tidak cukup dari ED1-ED5, nyatakan:
   "Identifikasi peluang eksternal terbatas berdasarkan data yang tersedia."
5. SWOT harus mengikuti pembedaan internal/eksternal yang ketat:
   - Kekuatan & Kelemahan = INTERNAL (dari DC, IC, EO, OA, RA, BR)
   - Peluang & Ancaman = EKSTERNAL (dari ED1-ED5 dan konteks profil)
6. Prediksi ML adalah output model, bukan kepastian. Jangan pernah berkata "bisnis akan gagal."
7. Setiap rekomendasi harus mengikuti alur: DATA -> INTERPRETASI -> MASALAH -> STRATEGI -> AKSI

============================================================
FORMAT OUTPUT (gunakan PERSIS struktur ini)
============================================================

# ANALISIS KETAHANAN BISNIS UMKM

## 1. Profil Bisnis
[Uraikan profil UMKM berdasarkan data Level 1 saja]

## 2. Konteks Demografis dan Sosial Pemilik
[Analisis Owner Age, Gender, Education, Legal Status sebagai informasi kontekstual saja.
JANGAN jadikan variabel ini sebagai bukti langsung ketahanan bisnis]

## 3. Skala dan Konteks Operasional Bisnis
[Analisis Business Age, Employees, Revenue, Digital Sales % menggunakan bahasa hati-hati]

## 4. Analisis Konstruk

### 4.1 Digital Capability
[Skor rata-rata + indikator DC terkuat dan terlemah + makna bisnis + relevansi ke ketahanan]

### 4.2 Innovation Capability
[Skor rata-rata + indikator IC terkuat dan terlemah + makna bisnis + relevansi ke ketahanan]

### 4.3 Entrepreneurial Orientation
[Skor rata-rata + indikator EO terkuat dan terlemah + makna bisnis + relevansi ke ketahanan]

### 4.4 Organizational Agility
[Skor rata-rata + indikator OA terkuat dan terlemah + makna bisnis + relevansi ke ketahanan]

### 4.5 Resource Access
[Skor rata-rata + indikator RA terkuat dan terlemah + makna bisnis + relevansi ke ketahanan]

### 4.6 Environmental Dynamism
[Skor rata-rata + indikator ED terkuat dan terlemah - ini menjadi basis Ancaman dan Peluang]

## 5. Analisis Ketahanan Bisnis (BR)
[Analisis semua BR1-BR7 - identifikasi dimensi terkuat dan terlemah - hubungkan ke konstruk predictor]

## 6. Interpretasi Prediksi Machine Learning
[Jelaskan makna prediksi {label_text} dengan confidence {confidence:.1%} menggunakan variabel aktual.
Sampaikan bahwa ini adalah output model, bukan kepastian masa depan]

## 7. Analisis SWOT

### Kekuatan (Strengths)
[Setiap Kekuatan memuat: Faktor | Bukti | Kode Indikator | Skor | Makna Strategis]

### Kelemahan (Weaknesses)
[Setiap Kelemahan memuat: Faktor | Bukti | Kode Indikator | Skor | Makna Strategis]

### Peluang (Opportunities)
[Hanya dari ED1-ED5 dan konteks profil. Jika bukti tidak cukup, nyatakan keterbatasan data]

### Ancaman (Threats)
[Hanya dari ED1-ED5 dan konteks profil. Setiap Ancaman memuat: Faktor | Bukti | Kode Indikator | Skor | Makna Strategis]

## 8. Matriks Strategi SWOT

### Strategi SO (Kekuatan x Peluang)
[Bagaimana kekuatan yang ada dapat dimanfaatkan untuk menangkap peluang? Sebutkan faktor SWOT yang mendasarinya]

### Strategi WO (Kelemahan x Peluang)
[Bagaimana kelemahan dapat diperbaiki untuk memanfaatkan peluang? Sebutkan faktor SWOT yang mendasarinya]

### Strategi ST (Kekuatan x Ancaman)
[Bagaimana kekuatan yang ada dapat mengurangi dampak ancaman? Sebutkan faktor SWOT yang mendasarinya]

### Strategi WT (Kelemahan x Ancaman)
[Bagaimana kelemahan dapat diminimalkan agar tidak diperparah ancaman? Sebutkan faktor SWOT yang mendasarinya]

## 9. Rekomendasi Prioritas
[Maksimal 5 prioritas, urutkan berdasarkan: keparahan kelemahan, relevansi prediksi, eksposur ancaman, kelayakan, dampak bisnis]

Prioritas 1: ...
Prioritas 2: ...
Prioritas 3: ...

## 10. Rencana Aksi

### 0-3 Bulan
[Masalah | Bukti | Tindakan Direkomendasikan | Area Penanggung Jawab | Target Perbaikan]

### 3-12 Bulan
[Masalah | Bukti | Tindakan Direkomendasikan | Area Penanggung Jawab | Target Perbaikan]

### 1-3 Tahun
[Masalah | Bukti | Tindakan Direkomendasikan | Area Penanggung Jawab | Target Perbaikan]

## 11. Risiko Utama
[Identifikasi risiko kritis berdasarkan kombinasi kelemahan internal dan ancaman eksternal dari data]

## 12. Kesimpulan Eksekutif
[Ringkasan kondisi bisnis saat ini, temuan kunci SWOT, dan arah strategis utama - berdasarkan data, bukan asumsi]
""".strip()

    return prompt


# ============================================================
# UNIFIED PROMPT DISPATCHER
# ============================================================

def build_umkm_prompt(
    business_name: str,
    business_sector: str,
    province: str,
    predicted_class: str,
    confidence: float,
    br_score: float,
    scores: dict,
    probabilities: dict,
    indicators: dict = None,
    profile: dict = None,
) -> str:
    """
    Dispatcher - memanggil versi prompt sesuai PROMPT_VERSION.
    Ubah PROMPT_VERSION di atas file untuk berganti versi.
    """
    common = dict(
        business_name=business_name,
        business_sector=business_sector,
        province=province,
        predicted_class=predicted_class,
        confidence=confidence,
        br_score=br_score,
        scores=scores,
        probabilities=probabilities,
        indicators=indicators,
        profile=profile,
    )
    if PROMPT_VERSION == "v1":
        return build_umkm_prompt_v1(**common)
    else:
        return build_umkm_prompt_v2(**common)


def get_gemini_recommendation(
    business_name: str,
    business_sector: str,
    province: str,
    predicted_class: str,
    confidence: float,
    br_score: float,
    scores: dict,
    probabilities: dict,
    indicators: dict = None,   # individual DC1-BR7 scores
    profile: dict = None,      # city, business_age, employees, revenue, etc.
) -> dict:
    """
    Kirim request ke Gemini API dan kembalikan rekomendasi.

    Returns:
        dict dengan keys:
        - success (bool)
        - content (str) — teks rekomendasi dari Gemini
        - error (str)   — pesan error jika gagal
        - model (str)   — nama model yang digunakan
        - prompt_version (str) — versi prompt yang digunakan
    """

    if not GEMINI_AVAILABLE:
        return {
            "success": False,
            "content": "",
            "error": "Package google-generativeai belum terinstall. Jalankan: pip install google-generativeai",
            "model": None,
        }

    api_key = get_api_key()
    if not api_key:
        return {
            "success": False,
            "content": "",
            "error": "API Key Gemini belum dikonfigurasi.",
            "model": None,
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature":       0.4,   # lebih rendah = lebih konsisten & faktual
                "top_p":             0.9,
                "top_k":             40,
                "max_output_tokens": 8192,
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )

        prompt = build_umkm_prompt(
            business_name=business_name,
            business_sector=business_sector,
            province=province,
            predicted_class=predicted_class,
            confidence=confidence,
            br_score=br_score,
            scores=scores,
            probabilities=probabilities,
            indicators=indicators,
            profile=profile,
        )

        response = model.generate_content(prompt)
        return {
            "success":        True,
            "content":        response.text,
            "error":          None,
            "model":          "gemini-2.0-flash",
            "prompt_version": PROMPT_VERSION,
        }

    except Exception as e:
        error_msg = str(e)
        # User-friendly error messages
        if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
            error_msg = "API Key tidak valid. Periksa kembali API Key Anda di file .env"
        elif "quota" in error_msg.lower():
            error_msg = "Quota API Gemini habis. Coba lagi dalam beberapa menit."
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            error_msg = "Tidak dapat terhubung ke Gemini. Periksa koneksi internet Anda."

        return {
            "success":        False,
            "content":        "",
            "error":          error_msg,
            "model":          "gemini-2.0-flash",
            "prompt_version": PROMPT_VERSION,
        }


def render_gemini_section(
    business_name: str,
    business_sector: str,
    province: str,
    result: dict,
    scores: dict,
    indicators: dict = None,   # opsional: individual DC1-BR7
    profile: dict = None,      # opsional: profil lengkap bisnis
):
    """
    Render tombol dan hasil rekomendasi Gemini di Streamlit.
    Dipanggil dari app.py setelah hasil prediksi ML ditampilkan.
    """
    st.markdown("---")
    st.markdown("## 🤖 Rekomendasi AI dari Google Gemini")

    if not GEMINI_AVAILABLE:
        st.warning(
            "⚠️ Package `google-generativeai` belum terinstall.\n\n"
            "Jalankan di terminal VSCode:\n```\npip install google-generativeai\n```"
        )
        return

    api_key = get_api_key()
    if not api_key:
        st.markdown("""
        <div style="background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.4);
                    border-radius:10px;padding:1.2rem;margin:0.5rem 0;">
            <strong>⚠️ API Key Gemini Belum Dikonfigurasi</strong><br><br>
            Untuk mengaktifkan rekomendasi AI dari Gemini:
            <ol style="margin-top:0.5rem;color:#c9d1d9;">
                <li>Dapatkan API key gratis di: <strong>https://aistudio.google.com/app/apikey</strong></li>
                <li>Buat file <code>.env</code> di folder proyek</li>
                <li>Isi dengan: <code>GEMINI_API_KEY=AIzaSy...key_anda...</code></li>
                <li>Restart Streamlit</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return

    # Tombol generate rekomendasi Gemini
    col1, col2 = st.columns([3, 1])
    with col1:
        pv_badge = "v1 - Simple" if PROMPT_VERSION == "v1" else "v2 - SWOT Evidence-Based"
        st.markdown(f"""
        <div style="background:rgba(88,166,255,0.08);border:1px solid rgba(88,166,255,0.3);
                    border-radius:10px;padding:1rem;">
            ✨ <strong>Powered by Google Gemini 2.0 Flash</strong>
            <span style="font-size:0.75rem;background:rgba(88,166,255,0.2);
                         border-radius:4px;padding:2px 6px;margin-left:6px;">Prompt {pv_badge}</span><br>
            <span style="font-size:0.85rem;color:#8b949e;">
            Analisis mendalam dan rekomendasi strategis yang dipersonalisasi
            berdasarkan data UMKM Anda oleh AI Google Gemini.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        generate_btn = st.button(
            "✨ Generate\nRekomendasi AI",
            key="gemini_btn",
            use_container_width=True,
            help="Klik untuk mendapatkan analisis mendalam dari Google Gemini AI"
        )

    if generate_btn:
        with st.spinner("🤖 Gemini sedang menganalisis data UMKM Anda... (10-30 detik)"):
            gemini_result = get_gemini_recommendation(
                business_name   = business_name,
                business_sector = business_sector,
                province        = province,
                predicted_class = result["predicted_class"],
                confidence      = result["confidence"],
                br_score        = result["br_score"],
                scores          = scores,
                probabilities   = result["probabilities"],
                indicators      = indicators,
                profile         = profile,
            )
        # Simpan ke session_state agar tidak hilang saat download diklik
        if gemini_result["success"]:
            st.session_state["gemini_result"] = gemini_result
            st.session_state["gemini_business_name"] = business_name
        else:
            st.session_state.pop("gemini_result", None)
            st.error(f"❌ Gagal mendapatkan rekomendasi: {gemini_result['error']}")

    # Tampilkan hasil dari session_state (persists saat download diklik)
    if st.session_state.get("gemini_result") and st.session_state["gemini_result"]["success"]:
        saved      = st.session_state["gemini_result"]
        saved_name = st.session_state.get("gemini_business_name", business_name)
        pv         = saved.get("prompt_version", "v2")

        st.markdown(f"""
        <div style="background:rgba(46,160,67,0.08);border:1px solid rgba(46,160,67,0.3);
                    border-radius:10px;padding:0.8rem 1rem;margin-bottom:1rem;">
            ✅ <strong>Rekomendasi berhasil digenerate oleh Gemini 2.0 Flash</strong>
            <span style="font-size:0.75rem;background:rgba(46,160,67,0.2);
                         border-radius:4px;padding:2px 6px;margin-left:6px;">Prompt {pv}</span>
        </div>
        """, unsafe_allow_html=True)

        # Area scroll untuk output panjang
        st.markdown(
            f"""<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                           padding:1.5rem 2rem;line-height:1.8;color:#e6edf3;font-size:0.95rem;
                           max-height:700px;overflow-y:auto;white-space:pre-wrap;">
            {saved['content'].replace('<','&lt;').replace('>','&gt;').replace(chr(10), '<br>')}
            </div>""",
            unsafe_allow_html=True
        )

        # Download — tidak menghilangkan hasil karena tersimpan di session_state
        st.download_button(
            label="📥 Download Rekomendasi Gemini (TXT)",
            data=saved["content"].encode("utf-8"),
            file_name=f"rekomendasi_gemini_{pv}_{saved_name.replace(' ','_')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_gemini_btn",
        )
