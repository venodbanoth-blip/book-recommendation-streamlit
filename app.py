import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BookWise | Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Main Background ---------- */

    .stApp {
        background:
            linear-gradient(
                rgba(15, 23, 42, 0.93),
                rgba(30, 41, 59, 0.96)
            ),
            url("https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=2000&q=80");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }


    /* ---------- Main Content ---------- */

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- Header ---------- */

    .hero {
        padding: 42px 35px;
        border-radius: 24px;
        margin-bottom: 28px;

        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 59, 0.96),
                rgba(51, 65, 85, 0.92)
            );

        border: 1px solid rgba(255,255,255,0.12);

        box-shadow:
            0 20px 50px rgba(0,0,0,0.35);

        text-align: center;
    }

    .hero h1 {
        color: white;
        font-size: 44px;
        margin-bottom: 10px;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 18px;
        margin-bottom: 0;
    }


    /* ---------- Search Card ---------- */

    .search-card {
        background: rgba(255,255,255,0.96);
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 30px;

        box-shadow:
            0 12px 35px rgba(0,0,0,0.20);
    }


    /* ---------- Recommendation Card ---------- */

    .book-card {
        background: rgba(255,255,255,0.97);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 18px;

        border: 1px solid rgba(255,255,255,0.5);

        box-shadow:
            0 10px 28px rgba(0,0,0,0.18);

        transition: all 0.25s ease;
    }

    .book-card:hover {
        transform: translateY(-4px);

        box-shadow:
            0 16px 38px rgba(0,0,0,0.28);
    }


    /* ---------- Book Text ---------- */

    .rank {
        display: inline-block;

        background: #0f172a;
        color: white;

        padding: 5px 12px;
        border-radius: 20px;

        font-size: 13px;
        font-weight: 700;

        margin-bottom: 8px;
    }

    .book-title {
        color: #0f172a;
        font-size: 21px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .book-author {
        color: #475569;
        font-size: 15px;
        margin-bottom: 5px;
    }

    .book-publisher {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .book-isbn {
        color: #64748b;
        font-size: 13px;
        font-family: monospace;
    }


    /* ---------- Section Header ---------- */

    .section-title {
        color: white;
        font-size: 28px;
        font-weight: 800;
        margin-top: 15px;
        margin-bottom: 20px;
    }


    /* ---------- Info Cards ---------- */

    .info-card {
        background: rgba(255,255,255,0.95);
        border-radius: 18px;
        padding: 22px;
        height: 100%;

        box-shadow:
            0 10px 25px rgba(0,0,0,0.16);
    }

    .info-card h3 {
        color: #0f172a;
        margin-bottom: 10px;
    }

    .info-card p {
        color: #475569;
        line-height: 1.6;
    }


    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0f172a,
                #1e293b
            );
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }


    /* ---------- Button ---------- */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 48px;

        font-size: 16px;
        font-weight: 700;

        border: none;

        background: #0f172a;
        color: white;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #334155;
        color: white;
        transform: translateY(-1px);
    }


    /* ---------- Input ---------- */

    div[data-baseweb="input"] {
        border-radius: 12px;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #cbd5e1;
        margin-top: 45px;
        padding-top: 25px;

        border-top: 1px solid rgba(255,255,255,0.15);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ARTIFACT DIRECTORY
# ============================================================

ARTIFACT_DIR = os.path.join(
    os.path.dirname(__file__),
    "model_artifacts"
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    required_files = [
        "tfidf_vectorizer.pkl",
        "tfidf_matrix.pkl",
        "isbn_to_index.pkl",
        "knn_model.pkl",
        "user_book_sparse.pkl",
        "user_to_index.pkl",
        "svd_model.pkl",
        "user_factors.pkl",
        "book_factors.pkl",
        "train_user_ids.pkl",
        "train_isbn_ids.pkl",
        "popularity_rank.pkl",
        "book_info.pkl",
        "train_history.pkl",
        "hybrid_config.json"
    ]

    missing_files = [
        file_name
        for file_name in required_files
        if not os.path.exists(
            os.path.join(ARTIFACT_DIR, file_name)
        )
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model artifact files: "
            + ", ".join(missing_files)
        )

    artifacts = {}

    artifacts["tfidf_vectorizer"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "tfidf_vectorizer.pkl"
        )
    )

    artifacts["tfidf_matrix"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "tfidf_matrix.pkl"
        )
    )

    artifacts["isbn_to_index"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "isbn_to_index.pkl"
        )
    )

    artifacts["knn_model"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "knn_model.pkl"
        )
    )

    artifacts["user_book_sparse"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "user_book_sparse.pkl"
        )
    )

    artifacts["user_to_index"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "user_to_index.pkl"
        )
    )

    artifacts["svd_model"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "svd_model.pkl"
        )
    )

    artifacts["user_factors"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "user_factors.pkl"
        )
    )

    artifacts["book_factors"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "book_factors.pkl"
        )
    )

    artifacts["train_user_ids"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "train_user_ids.pkl"
        )
    )

    artifacts["train_isbn_ids"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "train_isbn_ids.pkl"
        )
    )

    artifacts["popularity_rank"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "popularity_rank.pkl"
        )
    )

    artifacts["book_info"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "book_info.pkl"
        )
    )

    artifacts["train_history"] = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "train_history.pkl"
        )
    )

    with open(
        os.path.join(
            ARTIFACT_DIR,
            "hybrid_config.json"
        ),
        "r",
        encoding="utf-8"
    ) as file:

        artifacts["hybrid_config"] = json.load(file)

    return artifacts


# ============================================================
# LOAD ARTIFACTS SAFELY
# ============================================================

try:

    artifacts = load_artifacts()

except Exception as error:

    st.error("❌ Unable to load the recommendation system.")

    st.code(str(error))

    st.stop()


# ============================================================
# ASSIGN ARTIFACTS
# ============================================================

tfidf_vectorizer = artifacts["tfidf_vectorizer"]
tfidf_matrix = artifacts["tfidf_matrix"]
isbn_to_index = artifacts["isbn_to_index"]

knn_model = artifacts["knn_model"]
user_book_sparse = artifacts["user_book_sparse"]
user_to_index = artifacts["user_to_index"]

svd_model = artifacts["svd_model"]
user_factors = artifacts["user_factors"]
book_factors = artifacts["book_factors"]

train_user_ids = artifacts["train_user_ids"]
train_isbn_ids = artifacts["train_isbn_ids"]

popularity_rank = artifacts["popularity_rank"]

book_info = artifacts["book_info"]
train_history = artifacts["train_history"]

hybrid_config = artifacts["hybrid_config"]


# ============================================================
# BOOK INFORMATION
# ============================================================

required_book_columns = [
    "ISBN",
    "Book-Title",
    "Book-Author",
    "Publisher",
    "Image-URL-M"
]

missing_book_columns = [
    column
    for column in required_book_columns
    if column not in book_info.columns
]

if missing_book_columns:

    st.error(
        "book_info.pkl is missing columns: "
        + ", ".join(missing_book_columns)
    )

    st.stop()


# ============================================================
# BOOK MAPPINGS
# ============================================================

title_map = dict(
    zip(
        book_info["ISBN"],
        book_info["Book-Title"]
    )
)

author_map = dict(
    zip(
        book_info["ISBN"],
        book_info["Book-Author"]
    )
)

publisher_map = dict(
    zip(
        book_info["ISBN"],
        book_info["Publisher"]
    )
)

image_map = dict(
    zip(
        book_info["ISBN"],
        book_info["Image-URL-M"]
    )
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def clean_image_url(url):

    if pd.isna(url):
        return ""

    url = str(url).strip()

    if not url:
        return ""

    if url.startswith("http://"):
        url = url.replace(
            "http://",
            "https://",
            1
        )

    if not url.startswith(
        ("http://", "https://")
    ):
        return ""

    return url


# ============================================================
# POPULARITY RECOMMENDATION
# ============================================================

def recommend_popularity(
    user_id,
    top_n=10
):

    history = set(
        train_history[
            train_history["User-ID"] == user_id
        ]["ISBN"]
    )

    result = popularity_rank[
        ~popularity_rank["ISBN"].isin(history)
    ].head(top_n)

    return result["ISBN"].tolist()


# ============================================================
# CONTENT-BASED RECOMMENDATION
# ============================================================

def recommend_content(
    user_id,
    top_n=100
):

    history = set(
        train_history[
            train_history["User-ID"] == user_id
        ]["ISBN"]
    )

    history = [
        isbn
        for isbn in history
        if isbn in isbn_to_index
    ]

    if not history:
        return []

    history = history[:5]

    scores = {}

    for isbn in history:

        book_index = isbn_to_index[isbn]

        similarity = cosine_similarity(
            tfidf_matrix[book_index],
            tfidf_matrix
        ).flatten()

        candidate_count = min(
            101,
            len(similarity)
        )

        candidate_indices = np.argpartition(
            similarity,
            -candidate_count
        )[-candidate_count:]

        candidate_indices = candidate_indices[
            np.argsort(
                similarity[candidate_indices]
            )[::-1]
        ]

        for candidate_index in candidate_indices:

            if candidate_index >= len(train_isbn_ids):
                continue

            candidate_isbn = train_isbn_ids[
                candidate_index
            ]

            if candidate_isbn in history:
                continue

            score = float(
                similarity[candidate_index]
            )

            if (
                candidate_isbn not in scores
                or score > scores[candidate_isbn]
            ):
                scores[candidate_isbn] = score

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        isbn
        for isbn, score in ranked[:top_n]
    ]


# ============================================================
# COLLABORATIVE FILTERING
# ============================================================

def recommend_collaborative(
    user_id,
    top_n=100
):

    if user_id not in user_to_index:
        return []

    user_index = user_to_index[user_id]

    distances, indices = knn_model.kneighbors(
        user_book_sparse[user_index],
        return_distance=True
    )

    history = set(
        train_history[
            train_history["User-ID"] == user_id
        ]["ISBN"]
    )

    scores = {}

    for distance, neighbor_index in zip(
        distances[0],
        indices[0]
    ):

        neighbor_user = train_user_ids[
            neighbor_index
        ]

        if neighbor_user == user_id:
            continue

        similarity = max(
            0,
            1 - distance
        )

        neighbor_books = train_history[
            train_history["User-ID"] == neighbor_user
        ]

        for _, row in neighbor_books.iterrows():

            isbn = row["ISBN"]

            if isbn in history:
                continue

            rating = row["Book-Rating"]

            scores[isbn] = (
                scores.get(isbn, 0)
                + similarity * rating
            )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        isbn
        for isbn, score in ranked[:top_n]
    ]


# ============================================================
# SVD RECOMMENDATION
# ============================================================

def recommend_svd(
    user_id,
    top_n=100
):

    if user_id not in user_to_index:
        return []

    user_index = user_to_index[user_id]

    user_vector = user_factors[user_index]

    predicted_scores = np.dot(
        user_vector,
        book_factors
    )

    rated_indices = set(
        user_book_sparse[user_index].indices
    )

    ranked_indices = np.argsort(
        predicted_scores
    )[::-1]

    recommendations = []

    for book_index in ranked_indices:

        if book_index in rated_indices:
            continue

        if book_index >= len(train_isbn_ids):
            continue

        recommendations.append(
            train_isbn_ids[book_index]
        )

        if len(recommendations) >= top_n:
            break

    return recommendations


# ============================================================
# HYBRID RECOMMENDATION
# ============================================================

def recommend_hybrid(
    user_id,
    top_n=10
):

    content = recommend_content(
        user_id,
        top_n=100
    )

    collaborative = recommend_collaborative(
        user_id,
        top_n=100
    )

    svd = recommend_svd(
        user_id,
        top_n=100
    )

    popularity = popularity_rank[
        "ISBN"
    ].head(100).tolist()

    history = set(
        train_history[
            train_history["User-ID"] == user_id
        ]["ISBN"]
    )

    candidates = (
        set(content)
        |
        set(collaborative)
        |
        set(svd)
        |
        set(popularity)
    )

    candidates -= history

    if not candidates:
        return []

    content_scores = {
        isbn: 1 - (i / 100)
        for i, isbn in enumerate(content)
    }

    collaborative_scores = {
        isbn: 1 - (i / 100)
        for i, isbn in enumerate(collaborative)
    }

    svd_scores = {
        isbn: 1 - (i / 100)
        for i, isbn in enumerate(svd)
    }

    popularity_scores = {
        isbn: 1 - (i / 100)
        for i, isbn in enumerate(popularity)
    }

    content_weight = hybrid_config.get(
        "content_weight",
        0.30
    )

    collaborative_weight = hybrid_config.get(
        "collaborative_weight",
        0.30
    )

    svd_weight = hybrid_config.get(
        "svd_weight",
        0.20
    )

    popularity_weight = hybrid_config.get(
        "popularity_weight",
        0.20
    )

    final_scores = {}

    for isbn in candidates:

        score = (
            content_weight
            * content_scores.get(isbn, 0)
            +
            collaborative_weight
            * collaborative_scores.get(isbn, 0)
            +
            svd_weight
            * svd_scores.get(isbn, 0)
            +
            popularity_weight
            * popularity_scores.get(isbn, 0)
        )

        final_scores[isbn] = score

    ranked = sorted(
        final_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        isbn
        for isbn, score in ranked[:top_n]
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2>📚 BookWise</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🤖 Recommendation Models")

    st.write("• Popularity-Based")
    st.write("• Content-Based")
    st.write("• Collaborative Filtering")
    st.write("• SVD")
    st.write("• Hybrid")

    st.markdown("---")

    st.markdown("### ⚙️ Hybrid Weights")

    st.write(
        f"Content: {hybrid_config.get('content_weight', 0.30):.0%}"
    )

    st.write(
        f"Collaborative: {hybrid_config.get('collaborative_weight', 0.30):.0%}"
    )

    st.write(
        f"SVD: {hybrid_config.get('svd_weight', 0.20):.0%}"
    )

    st.write(
        f"Popularity: {hybrid_config.get('popularity_weight', 0.20):.0%}"
    )

    st.markdown("---")

    st.caption(
        "Machine Learning Book Recommendation System"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>📚 BookWise</h1>

        <p>
            Discover your next favorite book with
            Machine Learning powered recommendations.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SEARCH AREA
# ============================================================

st.markdown(
    """
    <div class="search-card">
    """,
    unsafe_allow_html=True
)

st.markdown(
    "### 🔍 Find Your Recommendations"
)

st.write(
    "Enter your User ID to receive personalized book recommendations."
)

user_id_input = st.text_input(
    "User ID",
    placeholder="Example: 276704",
    label_visibility="collapsed"
)

get_recommendations = st.button(
    "🚀 Get My Recommendations",
    type="primary",
    use_container_width=True
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

if get_recommendations:

    if not user_id_input.strip():

        st.warning(
            "⚠️ Please enter a User ID."
        )

        st.stop()

    try:

        user_id = int(
            user_id_input.strip()
        )

    except ValueError:

        st.error(
            "❌ User ID must be a numeric value."
        )

        st.stop()

    if user_id not in user_to_index:

        st.warning(
            "⚠️ User ID not found in the trained recommendation system."
        )

        st.info(
            "Please enter a User ID that exists in the training data."
        )

        st.stop()

    with st.spinner(
        "🔮 Finding books for you..."
    ):

        recommendations = recommend_hybrid(
            user_id,
            top_n=10
        )

    if not recommendations:

        st.warning(
            "No recommendations are available for this user."
        )

        st.stop()

    st.markdown(
        '<div class="section-title">✨ Recommended For You</div>',
        unsafe_allow_html=True
    )

    for rank, isbn in enumerate(
        recommendations,
        start=1
    ):

        title = title_map.get(
            isbn,
            "Unknown Title"
        )

        author = author_map.get(
            isbn,
            "Unknown Author"
        )

        publisher = publisher_map.get(
            isbn,
            "Unknown Publisher"
        )

        image_url = clean_image_url(
            image_map.get(
                isbn,
                ""
            )
        )

        col1, col2 = st.columns(
            [1, 4],
            gap="large"
        )

        with col1:

            if image_url:

                st.image(
                    image_url,
                    width=150
                )

            else:

                st.markdown(
                    """
                    <div style="
                        width:150px;
                        height:210px;
                        border-radius:12px;
                        background:#e2e8f0;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:50px;
                    ">
                    📕
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:

            st.markdown(
                f"""
                <div class="book-card">

                    <div class="rank">
                        #{rank}
                    </div>

                    <div class="book-title">
                        {title}
                    </div>

                    <div class="book-author">
                        ✍️ <b>Author:</b> {author}
                    </div>

                    <div class="book-publisher">
                        🏢 <b>Publisher:</b> {publisher}
                    </div>

                    <div class="book-isbn">
                        ISBN: {isbn}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# INFORMATION SECTION
# ============================================================

if not get_recommendations:

    st.markdown(
        '<div class="section-title">🧠 How It Works</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="info-card">

            <h3>📖 Content-Based</h3>

            <p>
            Finds books with similar titles, authors
            and publishers using TF-IDF and cosine similarity.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">

            <h3>👥 Collaborative</h3>

            <p>
            Uses similar users and their book-rating
            behavior to discover relevant books.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="info-card">

            <h3>🤖 Hybrid AI</h3>

            <p>
            Combines content, collaborative, SVD and
            popularity signals into one recommendation ranking.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        📚 <b>BookWise</b> — Machine Learning Book Recommendation System

        <br><br>

        Built with Python • Scikit-learn • Streamlit • Machine Learning

    </div>
    """,
    unsafe_allow_html=True
)
