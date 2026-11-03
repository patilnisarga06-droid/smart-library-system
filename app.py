import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Smart Library",
    page_icon="📚",
    layout="wide"
)

# -------------------- BOOK DATASET --------------------
books = {
    "Book_ID": list(range(1, 21)),
    "Title": [
        "Harry Potter", "The Hobbit", "The Lord of the Rings",
        "Percy Jackson", "The Hunger Games", "Twilight",
        "Pride and Prejudice", "Jane Eyre",
        "The Fault in Our Stars", "The Notebook",
        "Sherlock Holmes", "Murder on the Orient Express",
        "The Da Vinci Code", "Angels and Demons",
        "Atomic Habits", "The Power of Habit",
        "Rich Dad Poor Dad", "Think and Grow Rich",
        "The Alchemist", "Life of Pi"
    ],
    "Author": [
        "J K Rowling", "J R R Tolkien", "J R R Tolkien",
        "Rick Riordan", "Suzanne Collins", "Stephenie Meyer",
        "Jane Austen", "Charlotte Bronte", "John Green",
        "Nicholas Sparks", "Arthur Conan Doyle",
        "Agatha Christie", "Dan Brown", "Dan Brown",
        "James Clear", "Charles Duhigg", "Robert Kiyosaki",
        "Napoleon Hill", "Paulo Coelho", "Yann Martel"
    ],
    "Genre": [
        "Fantasy Magic Adventure", "Fantasy Adventure",
        "Fantasy Adventure", "Fantasy Mythology Adventure",
        "Science Fiction Adventure", "Romance Fantasy",
        "Romance Classic", "Romance Classic",
        "Romance Drama", "Romance Drama",
        "Mystery Detective", "Mystery Crime",
        "Mystery Thriller", "Mystery Thriller",
        "Self Help Motivation", "Self Help Psychology",
        "Finance Business", "Finance Motivation",
        "Philosophy Adventure", "Adventure Drama"
    ],
    "Description": [
        "A young wizard discovers magic and attends a school of wizardry.",
        "A hobbit goes on an exciting journey with dwarves and a wizard.",
        "A group of heroes fights evil to save the magical world.",
        "A young demigod experiences adventures based on Greek mythology.",
        "A girl fights for survival in a dangerous competition.",
        "A romantic story between a human girl and a vampire.",
        "A classic story about love marriage and social expectations.",
        "A woman faces challenges in life love and society.",
        "Two teenagers fall in love while dealing with illness.",
        "A beautiful love story about two people and their relationship.",
        "A detective solves complicated mysteries using intelligence.",
        "A detective investigates a mysterious murder on a train.",
        "A professor solves mysterious puzzles and secret codes.",
        "A thrilling story involving secret societies and mysteries.",
        "A book about building good habits and improving daily life.",
        "Explains how habits influence human behaviour and success.",
        "A book explaining financial education and wealth creation.",
        "A motivational book about success and achieving goals.",
        "A young man travels to discover his dreams and purpose.",
        "A boy survives a shipwreck with a tiger in the ocean."
    ]
}

df = pd.DataFrame(books)
df["Features"] = df["Genre"] + " " + df["Description"]

# -------------------- MACHINE LEARNING MODEL --------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Features"])
similarity_matrix = cosine_similarity(tfidf_matrix)

def recommend_books(book_title, number=5):
    book_index = df.index[df["Title"] == book_title][0]
    scores = list(enumerate(similarity_matrix[book_index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    scores = [item for item in scores if item[0] != book_index][:number]

    recommendations = []
    for index, score in scores:
        recommendations.append({
            "Title": df.iloc[index]["Title"],
            "Author": df.iloc[index]["Author"],
            "Genre": df.iloc[index]["Genre"],
            "Similarity": round(float(score) * 100, 1)
        })
    return pd.DataFrame(recommendations)

# -------------------- CUSTOM DESIGN --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4f7fb 0%, #eaf0f8 100%);
}
.main .block-container {
    padding-top: 2rem;
    max-width: 1150px;
}
.hero {
    background: linear-gradient(110deg, #142d4e, #285b83);
    padding: 34px 38px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 { color: white; margin-bottom: 8px; font-size: 2.35rem; }
.hero p { color: #e4edf6; font-size: 1.05rem; margin-bottom: 0; }
.section-title {
    color: #183653;
    font-size: 1.45rem;
    font-weight: 700;
    margin: 18px 0 12px 0;
}
.book-card {
    background: white;
    border: 1px solid #e0e8f0;
    border-radius: 14px;
    padding: 17px;
    min-height: 150px;
    box-shadow: 0 3px 12px rgba(20,45,78,0.06);
    margin-bottom: 12px;
}
.book-card h4 { color: #183653; margin: 0 0 8px 0; }
.book-card p { color: #526579; margin: 4px 0; font-size: 0.92rem; }
.metric-box {
    background: #17365D;
    border: 1px solid #285b83;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

.metric-box h2 {
    color: white !important;
    font-size: 32px;
}

.metric-box p {
    color: white !important;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
    <h1>📚 Smart Library Management System</h1>
    <p>Machine Learning Based Book Recommendation • Discover your next favourite read</p>
</div>
""", unsafe_allow_html=True)

# -------------------- SUMMARY --------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric-box"><h2>20</h2><p>Books in catalogue</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-box"><h2>5</h2><p>Recommendations per search</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-box"><h2>ML</h2><p>TF-IDF + Cosine Similarity</p></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">🔎 Find books you may enjoy</div>', unsafe_allow_html=True)
left, right = st.columns([2, 1])
with left:
    selected_book = st.selectbox("Choose a book", df["Title"].tolist())
with right:
    st.write("")
    st.write("")
    show_recommendations = st.button("✨ Recommend Books", type="primary", use_container_width=True)

if show_recommendations:
    selected = df[df["Title"] == selected_book].iloc[0]
    st.markdown('<div class="section-title">📖 Selected Book</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="book-card">
        <h4>{selected['Title']}</h4>
        <p><b>Author:</b> {selected['Author']}</p>
        <p><b>Genre:</b> {selected['Genre']}</p>
        <p>{selected['Description']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">⭐ Recommended for you</div>', unsafe_allow_html=True)
    recs = recommend_books(selected_book)
    cols = st.columns(2)
    for i, row in recs.iterrows():
        with cols[i % 2]:
            st.markdown(f"""
            <div class="book-card">
                <h4>📘 {row['Title']}</h4>
                <p><b>Author:</b> {row['Author']}</p>
                <p><b>Genre:</b> {row['Genre']}</p>
                <p><b>Content similarity:</b> {row['Similarity']}%</p>
            </div>
            """, unsafe_allow_html=True)
    st.caption("Similarity indicates how closely the book descriptions and genres match. It is not a user rating.")

st.markdown("---")
with st.expander("📚 Browse all books in the catalogue"):
    st.dataframe(df[["Title", "Author", "Genre", "Description"]], use_container_width=True, hide_index=True)

st.markdown(
    "<p style='text-align:center;color:#718096;font-size:0.85rem;'>Smart Library • Mini Project Prototype</p>",
    unsafe_allow_html=True
)
