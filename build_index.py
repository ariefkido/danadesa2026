"""
build_index.py
Jalankan SEKALI secara lokal:
    python build_index.py

Output:
    faiss.index   → FAISS index file
    chunks.json   → metadata tiap chunk

Push kedua file tersebut ke repo GitHub bersama app.py.
"""

import json
import faiss
import numpy as np
from google import genai

# ── KONFIGURASI ──────────────────────────────────────────────────────────────
JSON_FILE   = "PMK 7 2026 Tanpa Lampiran - Pengelolaan Dana Desa Tahun Anggaran 2026_llm.json"
INDEX_FILE  = "faiss.index"
CHUNKS_FILE = "chunks.json"
EMBED_MODEL = "gemini-embedding-001"
GEMINI_API_KEY = input("Masukkan Gemini API Key: ").strip()
BATCH_SIZE  = 50   # Gemini embedding max 100 per request, pakai 50 agar aman
# ─────────────────────────────────────────────────────────────────────────────

client = genai.Client(api_key=GEMINI_API_KEY)

# 1. Load JSON
print("Loading JSON...")
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Buat chunks — gabung full_context + text sebagai teks yang di-embed
chunks = []
for item in data:
    text = f"[{item['full_context']}] {item['text']}".strip()
    if not text:
        continue
    chunks.append({
        "text": text,
        "full_context": item["full_context"],
        "pasal": item.get("pasal", ""),
        "ayat": item.get("ayat", ""),
        "bab": item.get("bab", ""),
        "judul_bab": item.get("judul_bab", ""),
    })

print(f"Total chunks: {len(chunks)}")

# 3. Embed dalam batch
def embed_batch(texts):
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model="text-embedding-004",  # tanpa prefix "models/"
            contents=text,
        )
        embeddings.append(response.embeddings[0].values)
    return embeddings

print("Embedding chunks (ini butuh beberapa menit)...")
all_embeddings = []
for i in range(0, len(chunks), BATCH_SIZE):
    batch_texts = [c["text"] for c in chunks[i:i+BATCH_SIZE]]
    batch_embeddings = embed_batch(batch_texts)
    all_embeddings.extend(batch_embeddings)
    print(f"  {min(i+BATCH_SIZE, len(chunks))}/{len(chunks)} chunks selesai")

# 4. Build FAISS index
print("Building FAISS index...")
dim = len(all_embeddings[0])
index = faiss.IndexFlatIP(dim)  # Inner Product = cosine similarity (setelah normalize)

vectors = np.array(all_embeddings, dtype="float32")
faiss.normalize_L2(vectors)
index.add(vectors)

# 5. Simpan
faiss.write_index(index, INDEX_FILE)
print(f"✅ Saved: {INDEX_FILE}")

with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
print(f"✅ Saved: {CHUNKS_FILE}")

print("\nDone! Push faiss.index dan chunks.json ke GitHub repo.")