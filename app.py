import streamlit as st
import google.generativeai as genai
import json
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Chatbot Dana Desa 2026",
    page_icon="💰",
    layout="centered"
)

# --- 1. SETUP API KEY (OPSI B) ---
# Mencari API Key di Streamlit Secrets (untuk Deploy) atau manual input (untuk lokal)
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    with st.sidebar:
        st.title("🔑 Konfigurasi")
        api_key = st.text_input("Masukkan Gemini API Key (Lokal):", type="password")
        st.info("Jika sudah di-deploy, masukkan key ini di Dashboard Secrets Streamlit.")

if not api_key:
    st.warning("⚠️ API Key tidak ditemukan. Silakan konfigurasi di Secrets atau Sidebar.")
    st.stop()

# Konfigurasi AI Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. LOAD DATA PERATURAN ---
@st.cache_data
def load_context():
    file_name = 'PMK 7 2026 Tanpa Lampiran - Pengelolaan Dana Desa Tahun Anggaran 2026_llm.json'
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Menggabungkan data menjadi satu konteks teks
        context_parts = []
        for item in data:
            ctx = f"[{item['full_context']}] {item['text']}"
            context_parts.append(ctx)
        
        return "\n".join(context_parts)
    except FileNotFoundError:
        st.error(f"File {file_name} tidak ditemukan!")
        return None

peraturan_text = load_context()

if not peraturan_text:
    st.stop()

# --- 3. ANTARMUKA CHAT ---
st.title("🤖 Asisten Dana Desa 2026")
st.markdown("Tanyakan aturan terkait **PMK No. 7 Tahun 2026**. Data diambil langsung dari dokumen resmi.")

# Inisialisasi riwayat pesan
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input user
if prompt := st.chat_input("Contoh: Apa syarat penyaluran tahap I?"):
    # Simpan & Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respon AI
    with st.chat_message("assistant"):
        with st.spinner("Membaca peraturan..."):
            # Instruksi agar AI hanya menjawab berdasarkan data JSON
            system_instruction = f"""
            Anda adalah pakar hukum dari Kementerian Keuangan.
            Gunakan teks peraturan berikut untuk menjawab pertanyaan:
            
            {peraturan_text}
            
            Aturan jawaban:
            1. Jawablah hanya berdasarkan data di atas.
            2. Sebutkan nomor Pasal/Ayat dengan jelas (misal: Sesuai Pasal 5 Ayat 2...).
            3. Jika tidak ada di data, katakan informasi tersebut tidak diatur di PMK 7/2026.
            4. Gunakan bahasa Indonesia yang sopan dan profesional.
            """
            
            try:
                # Mengirim konteks + pertanyaan ke Gemini
                response = model.generate_content([system_instruction, prompt])
                answer = response.text
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Kesalahan: {str(e)}")