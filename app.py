import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Load & preprocess dataset
# =========================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    
    # Split pertanyaan + variasi
    questions = []
    answers = []
    pasals = []
    
    for _, row in df.iterrows():
        q_variants = str(row['pertanyaan||variasi1||variasi2']).split('||')
        for q in q_variants:
            questions.append(q.strip())
            answers.append(row['jawaban'])
            pasals.append(row['pasal_referensi'])
    
    clean_df = pd.DataFrame({
        'question': questions,
        'answer': answers,
        'pasal': pasals
    })
    
    return clean_df

# =========================
# Build vectorizer
# =========================
@st.cache_resource
def build_vectorizer(questions):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)
    return vectorizer, vectors

# =========================
# Search function
# =========================
def get_best_answer(query, vectorizer, vectors, df, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, vectors).flatten()
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    
    for idx in top_indices:
        results.append({
            'question': df.iloc[idx]['question'],
            'answer': df.iloc[idx]['answer'],
            'pasal': df.iloc[idx]['pasal'],
            'score': similarities[idx]
        })
    
    return results

# =========================
# UI Streamlit
# =========================
st.set_page_config(page_title="Chatbot PMK 7/2026", layout="wide")

st.title("🤖 Chatbot PMK 7 Tahun 2026")
st.write("Tanyakan apa saja terkait peraturan Dana Desa")

# Load data
DATA_PATH = "faq_pmk7dd_fix.csv"
df = load_data(DATA_PATH)
vectorizer, vectors = build_vectorizer(df['question'])

# Chat input
user_input = st.text_input("Masukkan pertanyaan:")

if user_input:
    results = get_best_answer(user_input, vectorizer, vectors, df)
    
    st.subheader("Jawaban:")
    
    for i, res in enumerate(results):
        st.markdown(f"### 🔹 Hasil {i+1}")
        st.write(f"**Kemiripan:** {res['score']:.2f}")
        st.write(f"**Pertanyaan mirip:** {res['question']}")
        st.write(f"**Jawaban:** {res['answer']}")
        st.write(f"**Referensi:** {res['pasal']}")
        st.divider()

# =========================
# Sidebar info
# =========================
st.sidebar.title("Tentang")
st.sidebar.write("Chatbot ini menggunakan TF-IDF + Cosine Similarity tanpa API AI.")
st.sidebar.write("Dataset berasal dari PMK 7 Tahun 2026.")
