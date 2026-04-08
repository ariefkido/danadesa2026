import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    questions, answers, pasals = [], [], []

    for _, row in df.iterrows():
        q_variants = str(row['pertanyaan||variasi1||variasi2']).split('||')
        for q in q_variants:
            questions.append(q.strip())
            answers.append(row['jawaban'])
            pasals.append(row['pasal_referensi'])

    return pd.DataFrame({
        "question": questions,
        "answer": answers,
        "pasal": pasals
    })

# =========================
# VECTORIZE
# =========================
@st.cache_resource
def build_model(questions):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)
    return vectorizer, vectors

# =========================
# SEARCH
# =========================
def search(query, vectorizer, vectors, df, top_k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, vectors).flatten()

    idx = sims.argsort()[-top_k:][::-1]

    results = []
    for i in idx:
        results.append({
            "question": df.iloc[i]["question"],
            "answer": df.iloc[i]["answer"],
            "pasal": df.iloc[i]["pasal"],
            "score": sims[i]
        })
    return results

# =========================
# UI
# =========================
st.set_page_config(page_title="Chatbot PMK 7", layout="centered")

st.title("🤖 Chatbot PMK 7 Tahun 2026")
st.write("Tanyakan apa saja terkait Dana Desa")

df = load_data("faq_pmk7dd_fix.csv")
vectorizer, vectors = build_model(df["question"])

# SESSION STATE
if "selected_answer" not in st.session_state:
    st.session_state.selected_answer = None
    st.session_state.selected_question = None

# INPUT
query = st.text_input("Tanyakan sesuatu:")

# =========================
# SHOW SUGGESTION
# =========================
if query:
    results = search(query, vectorizer, vectors, df)

    st.subheader("🤔 Apakah ini maksud Anda?")

    cols = st.columns(len(results))

    for i, res in enumerate(results):
        with cols[i]:
            if st.button(res["question"], key=f"suggest_{i}"):
                st.session_state.selected_answer = res["answer"]
                st.session_state.selected_question = res["question"]
                st.session_state.selected_pasal = res["pasal"]
                st.rerun()

# =========================
# SHOW FINAL ANSWER
# =========================
if st.session_state.selected_answer:
    st.divider()
    st.markdown(f"### 📌 Pertanyaan terpilih")
    st.write(st.session_state.selected_question)

    st.markdown("### ✅ Jawaban")
    st.success(st.session_state.selected_answer)

    st.markdown("### 📖 Referensi Pasal")
    st.info(st.session_state.selected_pasal)