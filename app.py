import streamlit as st
from google import genai
import json

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Chatbot Dana Desa 2026",
    page_icon="💰",
    layout="centered"
)

# --- 1. SETUP API KEY ---
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    with st.sidebar:
        st.title("🔑 Konfigurasi")
        api_key = st.text_input("Masukkan Gemini API Key:", type="password")
        st.info("Dapatkan key di: aistudio.google.com")

if not api_key:
    st.warning("⚠️ API Key tidak ditemukan. Harap masukkan API Key untuk memulai.")
    st.stop()

# Inisialisasi client (cara baru)
try:
    client = genai.Client(api_key=api_key)
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            system_instruction = f"""Anda adalah pakar hukum keuangan negara. Jawablah hanya berdasarkan data PMK 7/2026 berikut:

{peraturan_text if peraturan_text else "Data tidak tersedia."}

Instruksi:
- Jika jawaban ada, sebutkan Pasal/Ayat.
- Jika tidak ada, katakan tidak diatur dalam PMK ini.
- Gunakan Bahasa Indonesia formal."""

            try:
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=f"{system_instruction}\n\nPertanyaan User: {prompt}"
                )
                answer = response.text
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memanggil AI: {str(e)}")