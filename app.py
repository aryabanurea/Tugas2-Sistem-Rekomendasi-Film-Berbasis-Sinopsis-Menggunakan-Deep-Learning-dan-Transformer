import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Sistem Rekomendasi Film",
    page_icon=" ",
    layout="centered"
)

st.title(" Sistem Rekomendasi Film")
st.write("Content-Based Recommendation menggunakan **Transformer (BERT)**")

# ===============================
# LOAD DATASET
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/movies_metadata.csv", low_memory=False)

    # Ambil kolom penting
    df = df[['title', 'overview']]

    # Bersihkan data
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Normalisasi judul
    df['title_clean'] = df['title'].str.lower().str.strip()

    return df

try:
    df = load_data()
    st.success(f"Dataset berhasil dimuat ({df.shape[0]} film)")
except Exception as e:
    st.error("Dataset tidak ditemukan. Pastikan movies_metadata.csv satu folder dengan app.py")
    st.stop()

# ===============================
# LOAD MODEL TRANSFORMER
# ===============================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ===============================
# EMBEDDING FILM
# ===============================
@st.cache_resource
def generate_embeddings(texts):
    return model.encode(texts, show_progress_bar=True)

with st.spinner("🔄 Memproses embedding film (hanya sekali, tunggu)..."):
    movie_embeddings = generate_embeddings(df["overview"].tolist())

# ===============================
# FUNGSI REKOMENDASI (ANTI ERROR)
# ===============================
def recommend_movie(query, top_n=5):
    query = query.lower().strip()

    # Cari judul yang mengandung input user
    candidates = df[df["title_clean"].str.contains(query)]

    if len(candidates) == 0:
        return None

    # Ambil kandidat pertama
    idx = candidates.index[0]

    query_embedding = movie_embeddings[idx].reshape(1, -1)

    similarities = cosine_similarity(query_embedding, movie_embeddings)[0]

    # Ambil top-N selain dirinya sendiri
    similar_indices = similarities.argsort()[::-1][1 : top_n + 1]

    result = df.iloc[similar_indices][["title"]].copy()
    result["Skor Kemiripan"] = similarities[similar_indices]

    return result.reset_index(drop=True)

# ===============================
# INTERFACE USER
# ===============================
st.subheader("Cari Film")

user_input = st.text_input(
    "Masukkan judul film",
    placeholder="Contoh: Batman"
)

top_n = st.slider("Jumlah rekomendasi", 3, 10, 5)

if st.button(" Rekomendasikan"):
    if user_input.strip() == "":
        st.warning("Masukkan judul film terlebih dahulu.")
    else:
        hasil = recommend_movie(user_input, top_n)

        if hasil is None:
            st.error("Judul film tidak ditemukan dalam dataset.")
        else:
            st.subheader("🎥 Hasil Rekomendasi")
            st.table(hasil)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Sistem Rekomendasi Film • Transformer (BERT) • Content-Based Filtering")
