# recommend_app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- ダミーデータ ---
data = {
    'title': ['Inception', 'The Matrix', 'Interstellar', 'The Notebook', 'Titanic'],
    'description': [
        "A thief who steals corporate secrets through dream-sharing technology.",
        "A hacker learns about the true nature of his reality and his role in the war.",
        "A team travels through a wormhole in space to ensure humanity's survival.",
        "A romantic drama about a young couple in the 1940s.",
        "A love story aboard the ill-fated Titanic ship."
    ]
}
df = pd.DataFrame(data)

# --- TF-IDFの準備 ---
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description'])

# --- Streamlit アプリ ---
st.title("🎬 類似映画レコメンドアプリ")

user_input = st.text_area("あらすじや紹介文を入力してください:")

if user_input:
    input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:3]

    st.subheader("🔍 類似おすすめ作品:")
    for i in top_indices:
        st.write(f"**{df.iloc[i]['title']}** — 類似度: {similarities[i]:.2f}")
