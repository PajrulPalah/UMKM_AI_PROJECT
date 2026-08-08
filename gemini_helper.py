# ============================================================
# gemini_helper.py
# Google Gemini AI Integration untuk UMKM Resilience System
# Model: gemini-3.5-flash (gratis, cepat)
# ============================================================

import os
import streamlit as st
from pathlib import Path

# ── Coba import google-generativeai ────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


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


def build_umkm_prompt(
    business_name: str,
    business_sector: str,
    province: str,
    predicted_class: str,
    confidence: float,
    br_score: float,
    scores: dict,
    probabilities: dict,
) -> str:
    """
    Bangun prompt lengkap untuk Gemini berdasarkan data UMKM.
    Prompt dirancang agar Gemini memberikan rekomendasi spesifik dan actionable.
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


def get_gemini_recommendation(
    business_name: str,
    business_sector: str,
    province: str,
    predicted_class: str,
    confidence: float,
    br_score: float,
    scores: dict,
    probabilities: dict,
) -> dict:
    """
    Kirim request ke Gemini API dan kembalikan rekomendasi.

    Returns:
        dict dengan keys:
        - success (bool)
        - content (str) — teks rekomendasi dari Gemini
        - error (str) — pesan error jika gagal
        - model (str) — nama model yang digunakan
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
        # Konfigurasi Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-3.5-flash",
            generation_config={
                "temperature":     0.7,
                "top_p":           0.9,
                "top_k":           40,
                "max_output_tokens": 2048,
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )

        prompt = build_umkm_prompt(
            business_name, business_sector, province,
            predicted_class, confidence, br_score,
            scores, probabilities
        )

        response = model.generate_content(prompt)
        return {
            "success": True,
            "content": response.text,
            "error":   None,
            "model":   "gemini-3.5-flash",
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
            "success": False,
            "content": "",
            "error":   error_msg,
            "model":   "gemini-3.5-flash",
        }


def render_gemini_section(
    business_name: str,
    business_sector: str,
    province: str,
    result: dict,
    scores: dict,
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
        st.markdown("""
        <div style="background:rgba(88,166,255,0.08);border:1px solid rgba(88,166,255,0.3);
                    border-radius:10px;padding:1rem;">
            ✨ <strong>Powered by Google Gemini 3.5 Flash</strong><br>
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
        with st.spinner("🤖 Gemini sedang menganalisis data UMKM Anda... (5-15 detik)"):
            gemini_result = get_gemini_recommendation(
                business_name    = business_name,
                business_sector  = business_sector,
                province         = province,
                predicted_class  = result["predicted_class"],
                confidence       = result["confidence"],
                br_score         = result["br_score"],
                scores           = scores,
                probabilities    = result["probabilities"],
            )

        if gemini_result["success"]:
            st.markdown("""
            <div style="background:rgba(46,160,67,0.08);border:1px solid rgba(46,160,67,0.3);
                        border-radius:10px;padding:0.8rem 1rem;margin-bottom:1rem;">
                ✅ <strong>Rekomendasi berhasil digenerate oleh Gemini 3.5 Flash</strong>
            </div>
            """, unsafe_allow_html=True)

            # Tampilkan rekomendasi
            st.markdown(
                f"""<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                               padding:1.5rem 2rem;line-height:1.8;color:#e6edf3;font-size:0.95rem;">
                {gemini_result['content'].replace(chr(10), '<br>')}
                </div>""",
                unsafe_allow_html=True
            )

            # Download rekomendasi
            st.download_button(
                label="📥 Download Rekomendasi Gemini (TXT)",
                data=gemini_result["content"].encode("utf-8"),
                file_name=f"rekomendasi_gemini_{business_name.replace(' ','_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.error(f"❌ Gagal mendapatkan rekomendasi: {gemini_result['error']}")
