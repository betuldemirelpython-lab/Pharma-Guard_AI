import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Load components
from utils import index_pdfs, get_vector_store, generate_pdf_report
from agents import PharmaOrchestrator

# Load environmental variables
load_dotenv()

# Streamlit App Configurations
st.set_page_config(
    page_title="Pharma-Guard AI - Akıllı İlaç Denetçisi",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Rich Dark Theme with Blue-Purple accents & Glassmorphism)
st.markdown("""
<style>
    /* Dark Theme Core overrides */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
        color: #F8FAFC;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    
    /* Main title accent gradient */
    .main-title {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Card Style (Glassmorphism) */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Dynamic Badge Style */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        margin-right: 5px;
    }
    .badge-success { background-color: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-danger { background-color: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-warning { background-color: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-info { background-color: rgba(59, 130, 246, 0.2); color: #3B82F6; border: 1px solid rgba(59, 130, 246, 0.3); }

    /* Micro-animations and custom buttons */
    .stButton>button {
        background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 {
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] strong, [data-testid="stSidebar"] div {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #CBD5E1 !important;
    }
</style>
""", unsafe_allow_html=True)

# App Logo and Headers
st.markdown("<div class='main-title'>🛡️ Pharma-Guard AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Yapay Zeka Destekli Akıllı İlaç Denetim ve Doğrulama Asistanı</div>", unsafe_allow_html=True)

# Sidebar - Settings, API Status and RAG Corpus Management
with st.sidebar:
    st.image("https://img.icons8.com/color/144/shield-with-medical-cross.png", width=80)
    st.markdown("### ⚙️ Sistem Durumu")
    
    # API Key check badges
    gemini_ok = "Aktif" if os.getenv("GEMINI_API_KEY") else "Pasif"
    groq_ok = "Aktif" if os.getenv("GROQ_API_KEY") else "Pasif"
    
    st.markdown(f"**Gemini API:** <span class='badge {'badge-success' if gemini_ok == 'Aktif' else 'badge-danger'}'>{gemini_ok}</span>", unsafe_allow_html=True)
    st.markdown(f"**Groq API:** <span class='badge {'badge-success' if groq_ok == 'Aktif' else 'badge-danger'}'>{groq_ok}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📚 Prospektüs Yönetimi (RAG)")
    
    # Show listed PDFs
    corpus_dir = os.path.join(os.getcwd(), "data", "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(corpus_dir) if f.lower().endswith(".pdf")]
    
    if pdf_files:
        st.write(f"İndekslenmiş Belgeler ({len(pdf_files)}):")
        for pdf in pdf_files:
            st.caption(f"📄 {pdf}")
    else:
        st.warning("Veritabanında henüz prospektüs bulunmamaktadır.")
        
    uploaded_pdf = st.file_uploader("Yeni Prospektüs Ekle (PDF)", type=["pdf"])
    if uploaded_pdf is not None:
        with open(os.path.join(corpus_dir, uploaded_pdf.name), "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.success(f"{uploaded_pdf.name} başarıyla yüklendi!")
        
        with st.spinner("Vektör veritabanı güncelleniyor..."):
            index_pdfs()
        st.success("İndeksleme tamamlandı! Yeniden yüklendi.")
        st.rerun()

    # Re-index manually
    if st.button("🔄 Veritabanını Yeniden İndeksle"):
        with st.spinner("İndeksleniyor..."):
            index_pdfs()
        st.success("Tüm prospektüsler başarıyla indekslendi!")

# Main grid layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📸 İlaç Kutusu Görseli Yükle")
    uploaded_image = st.file_uploader("İlaç Kutusunu Net ve Işıklı Bir Alanda Çekip Yükleyin", type=["png", "jpg", "jpeg"])
    
    image = None
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Yüklenen İlaç Görseli", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Denetim ve Analiz Süreci")
    
    # Manual Input Fallback Expander
    with st.expander("📝 Sadece İlaç Adı ile Sorgula (Yedek & Pratik Yöntem)"):
        st.info("API kotaları aşıldığında veya acil durumlarda sadece ilacın adını yazarak tüm denetim süreçlerini başlatabilirsiniz. Sistem etken madde, dozaj ve üretici bilgilerini yapay zeka ile otomatik tamamlayacaktır.")
        m_ilac_adi = st.text_input("İlaç Adı (Örn: Parol, Aspirin, Apranax)", "")
        use_manual = st.checkbox("Sadece Yazılan İlaç Adını Kullan")

    if st.button("🚀 Denetimi Başlat"):
        # Multi-agent orchestrator setup
        orchestrator = PharmaOrchestrator()
        
        # Determine manual data
        manual_data = None
        if use_manual:
            if not m_ilac_adi:
                st.error("Manuel analiz için lütfen ilaç adını yazın!")
                st.stop()
            manual_data = {
                "ilac_adi": m_ilac_adi,
                "etken_madde": "Otomatik",
                "dozaj": "Otomatik",
                "uretici_firma": "Otomatik",
                "yazi_okunuyor_mu": True,
                "guven_puani": 10
            }
        elif image is None:
            st.error("Lütfen sol taraftaki panelden bir ilaç kutusu fotoğrafı yükleyin veya yukarıdaki 'Sadece İlaç Adı ile Sorgula' kısmına ilaç adı yazıp onaylayın.")
            st.stop()
        
        # Progress indications
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown("🔍 **[Vision-Scanner]** İlaç bilgileri analiz ediliyor...")
        progress_bar.progress(20)
        
        # Run orchestrator
        with st.spinner("Pharma-Guard ajanları çalışıyor..."):
            final_report, logs = orchestrator.process_medicine(image, manual_data=manual_data)
            
        progress_bar.progress(100)
        status_text.markdown("✅ **Analiz Tamamlandı!**")
        
        # Display logs per agent
        if logs:
            st.markdown("#### 🕵️ Ajan Raporları")
            l_col1, l_col2, l_col3, l_col4 = st.columns(4)
            
            with l_col1:
                score = logs["vision"].get("guven_puani", 1) if "vision" in logs else 0
                st.metric("Vision Scanner", f"{score}/10", help="İlaç adı ve mg okuma doğruluğu")
            with l_col2:
                docs_found = logs["rag"].get("bulunan_belgeler_sayisi", 0) if "rag" in logs else 0
                st.metric("RAG Specialist", f"{docs_found} Belge", help="Bulunan prospektüs eşleşmesi")
            with l_col3:
                safety_score = logs["safety"].get("guven_puani", 1) if "safety" in logs else 0
                st.metric("Safety Auditor", f"{safety_score}/10", help="Uyuşmazlık ve risk denetim güveni")
            with l_col4:
                corp_score = logs["corporate"].get("guvenirlik_puani", 1) if "corporate" in logs else 0
                st.metric("Corporate Analyst", f"{corp_score}/10", help="Üretici firma güvenliği")
            
            # Check for warnings or blocks
            if logs.get("safety", {}).get("durum") == "VERİ UYUŞMAZLIĞI":
                st.error(f"🚨 **VERİ UYUŞMAZLIĞI ALARMI**: {logs['safety'].get('fark_detaylari')}")
            elif not logs.get("vision", {}).get("yazi_okunuyor_mu", True):
                st.warning("⚠️ **KURAL İHLALİ**: Görsel üzerindeki yazılar okunamıyor. Lütfen daha iyi ışık altında tekrar çekin.")
                st.error(f"Ajan Detay Hatası: {logs.get('vision')}")
        
        # Display Final synthesized Report
        st.markdown("---")
        st.markdown("### 📝 Nihai Denetim Raporu")
        st.markdown(final_report)
        
        # Generate and allow downloading PDF
        pdf_path = generate_pdf_report(final_report)
        
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Profesyonel PDF Raporu İndir",
                data=f,
                file_name="ilac_denetim_raporu.pdf",
                mime="application/pdf"
            )
    st.markdown("</div>", unsafe_allow_html=True)
