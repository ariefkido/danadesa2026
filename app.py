import streamlit as st
import google.generativeai as genai
import json

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Chatbot Dana Desa 2026",
    page_icon="💰",
    layout="centered"
)

# --- 1. SETUP API KEY (OPSI B) ---
# Mengambil dari Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    # Fallback jika dijalankan lokal tanpa secrets
    with st.sidebar:
        st.title("🔑 Konfigurasi")
        api_key = st.text_input("Masukkan Gemini API Key:", type="password")
        st.info("Dapatkan key di: aistudio.google.com")

if not api_key:
    st.warning("⚠️ API Key tidak ditemukan. Harap masukkan API Key untuk memulai.")
    st.stop()

# Konfigurasi AI Gemini
try:
    genai.configure(api_key=api_key)
    # Gunakan nama model yang lengkap: 'models/gemini-1.5-flash'
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
except Exception as e:
    st.error(f"Gagal konfigurasi Gemini: {e}")
    st.stop()

# --- 2. LOAD DATA PERATURAN ---
@st.cache_data
def load_context():
    file_name = 'PMK 7 2026 Tanpa Lampiran - Pengelolaan Dana Desa Tahun Anggaran 2026_llm.json'
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        context_parts = []
        for item in data:
            ctx = f"[{item['full_context']}] {item['text']}"
            context_parts.append(ctx)
        
        return "\n".join(context_parts)
    except FileNotFoundError:
        st.error(f"File {file_name} tidak ditemukan di repositori!")
        return None
    except Exception as e:
        st.error(f"Error memuat JSON: {e}")
        return None

peraturan_text = load_context()

# --- 3. ANTARMUKA CHAT ---
st.title("🤖 Asisten Dana Desa 2026")
st.caption("Menjawab berdasarkan PMK No. 7 Tahun 2026")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input user
if prompt := st.chat_input("Tanyakan sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            system_instruction = f"""
            Anda adalah pakar hukum keuangan negara. Jawablah hanya berdasarkan data PMK 7/2026 berikut:
            
            {peraturan_text if peraturan_text else "Data tidak tersedia."}
            
            Instruksi:
            - Jika jawaban ada, sebutkan Pasal/Ayat.
            - Jika tidak ada, katakan tidak diatur dalam PMK ini.
            - Gunakan Bahasa Indonesia formal.
            """
            
            try:
                # Menggunakan generate_content dengan struktur yang lebih aman
                response = model.generate_content(
                    f"{system_instruction}\n\nPertanyaan User: {prompt}"
                )
                
                if response.text:
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                else:
                    st.error("AI tidak memberikan respon. Coba ulangi pertanyaan.")
                    
            except Exception as e:
                # Menampilkan error yang lebih jelas jika model tidak ditemukan
                st.error(f"Terjadi kesalahan saat memanggil AI: {str(e)}")
                if "404" in str(e):
                    st.info("Tips: Pastikan API Key Anda aktif dan coba ganti model ke 'gemini-pro' jika 'gemini-1.5-flash' bermasalah di region Anda.")