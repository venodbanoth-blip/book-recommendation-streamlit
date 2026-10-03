import os
import json
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BookWise | AI Book Recommendation System",
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

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            linear-gradient(
                rgba(8, 15, 30, 0.94),
                rgba(15, 23, 42, 0.97)
            ),
            url("https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=2200&q=85");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #080f1e 0%,
                #111827 50%,
                #172033 100%
            );

        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .sidebar-brand {
        text-align: center;
        padding: 15px 5px 22px 5px;
    }

    .sidebar-logo {
        font-size: 46px;
        margin-bottom: 5px;
    }

    .sidebar-title {
        font-size: 25px;
        font-weight: 800;
        color: white;
        letter-spacing: -0.5px;
    }

    .sidebar-subtitle {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 5px;
    }

    .sidebar-section {
        color: #94a3b8;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 22px;
        margin-bottom: 12px;
    }

    .model-item {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 9px 12px;
        margin-bottom: 7px;
        font-size: 13px;
    }

    .weight-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 2px;
        color: #cbd5e1;
        font-size: 13px;
    }

    .weight-value {
        color: white;
        font-weight: 700;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        padding: 58px 35px 52px 35px;

        border-radius: 28px;

        margin-bottom: 25px;

        background:
            linear-gradient(
                135deg,
                rgba(15, 23, 42, 0.97),
                rgba(30, 41, 59, 0.93)
            );

        border: 1px solid rgba(255,255,255,0.10);

        box-shadow:
            0 25px 70px rgba(0,0,0,0.35);

        text-align: center;
    }

    .hero::before {
        content: "📖";
        position: absolute;
        font-size: 130px;
        opacity: 0.035;
        right: 40px;
        top: -20px;
        transform: rotate(-12deg);
    }

    .hero::after {
        content: "📚";
        position: absolute;
        font-size: 110px;
        opacity: 0.035;
        left: 35px;
        bottom: -25px;
        transform: rotate(10deg);
    }

    .hero-icon {
        font-size: 48px;
        margin-bottom: 8px;
    }

    .hero h1 {
        color: white;
        font-size: 52px;
        line-height: 1.1;
        margin: 0;
        font-weight: 850;
        letter-spacing: -2px;
    }

    .hero-highlight {
        background: linear-gradient(
            90deg,
            #f8fafc,
            #cbd5e1
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 17px;
        margin: 15px auto 0 auto;
        max-width: 680px;
        line-height: 1.7;
    }

    .hero-badges {
        margin-top: 22px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        margin: 4px;
        border-radius: 20px;

        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.09);

        color: #cbd5e1;
        font-size: 12px;
        font-weight: 600;
    }


    /* ========================================================
       SEARCH PANEL
       ======================================================== */

    .search-panel {
        background: rgba(255,255,255,0.97);
        border-radius: 22px;
        padding: 25px 28px 20px 28px;

        box-shadow:
            0 18px 45px rgba(0,0,0,0.20);

        border: 1px solid rgba(255,255,255,0.55);

        margin-bottom: 32px;
    }

    .search-title {
        color: #0f172a;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 3px;
    }

    .search-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 15px;
    }

    div[data-baseweb="input"] {
        border-radius: 12px;
    }

    div[data-baseweb="input"] input {
        color: #0f172a !important;
        font-weight: 600;
    }

    .stButton > button {
        width: 100%;
        min-height: 46px;

        border-radius: 12px;

        font-size: 15px;
        font-weight: 750;

        border: 0;

        background: #0f172a;
        color: white;

        box-shadow:
            0 7px 18px rgba(15,23,42,0.22);

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #1e293b;
        color: white;
        transform: translateY(-2px);

        box-shadow:
            0 10px 25px rgba(15,23,42,0.30);
    }


    /* ========================================================
       SECTION HEADER
       ======================================================== */

    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;

        margin: 10px 0 18px 0;
    }

    .section-title {
        color: white;
        font-size: 29px;
        font-weight: 850;
        letter-spacing: -0.6px;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 3px;
    }


    /* ========================================================
       RECOMMENDATION CARD
       ======================================================== */

    .book-card {
        background: rgba(255,255,255,0.98);

        border-radius: 20px;

        padding: 14px;

        margin-bottom: 18px;

        border: 1px solid rgba(255,255,255,0.70);

        box-shadow:
            0 12px 32px rgba(0,0,0,0.18);

        min-height: 455px;

        transition:
            transform 0.22s ease,
            box-shadow 0.22s ease;
    }

    .book-card:hover {
        transform: translateY(-6px);

        box-shadow:
            0 20px 45px rgba(0,0,0,0.28);
    }

    .rank-row {
        display: flex;
        align-items: center;
        justify-content: space-between;

        margin-bottom: 10px;
    }

    .rank {
        display: inline-block;

        background: #0f172a;
        color: white;

        padding: 5px 10px;

        border-radius: 20px;

        font-size: 11px;
        font-weight: 800;
    }

    .ml-badge {
        color: #64748b;
        font-size: 10px;
        font-weight: 700;

        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .book-image-container {
        display: flex;
        justify-content: center;
        align-items: center;

        width: 100%;
        height: 245px;

        background:
            linear-gradient(
                145deg,
                #f1f5f9,
                #e2e8f0
            );

        border-radius: 15px;

        overflow: hidden;

        margin-bottom: 15px;
    }

    .book-image-container img {
        max-width: 155px;
        max-height: 225px;

        width: auto;
        height: auto;

        object-fit: contain;

        border-radius: 7px;

        box-shadow:
            0 9px 22px rgba(0,0,0,0.18);
    }

    .book-placeholder {
        width: 125px;
        height: 190px;

        border-radius: 9px;

        background: #cbd5e1;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 48px;

        box-shadow:
            0 9px 20px rgba(0,0,0,0.10);
    }

    .book-title {
        color: #0f172a;

        font-size: 17px;
        font-weight: 800;

        line-height: 1.35;

        height: 47px;

        overflow: hidden;

        margin-bottom: 8px;
    }

    .book-author {
        color: #475569;

        font-size: 12px;

        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;

        margin-bottom: 5px;
    }

    .book-publisher {
        color: #64748b;

        font-size: 11px;

        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;

        margin-bottom: 7px;
    }

    .book-isbn {
        color: #94a3b8;

        font-size: 10px;

        font-family: monospace;
    }


    /* ========================================================
       INFORMATION CARDS
       ======================================================== */

    .info-title {
        color: white;

        font-size: 27px;
        font-weight: 850;

        margin-top: 42px;
        margin-bottom: 18px;
    }

    .info-card {
        background: rgba(255,255,255,0.96);

        border-radius: 18px;

        padding: 22px;

        min-height: 205px;

        border: 1px solid rgba(255,255,255,0.60);

        box-shadow:
            0 12px 30px rgba(0,0,0,0.16);
    }

    .info-icon {
        font-size: 30px;
        margin-bottom: 10px;
    }

    .info-card h3 {
        color: #0f172a;

        font-size: 18px;

        margin: 0 0 8px 0;
    }

    .info-card p {
        color: #64748b;

        font-size: 13px;

        line-height: 1.65;

        margin: 0;
    }


    /* ========================================================
       STATS
       ======================================================== */

    .stat-card {
        background: rgba(255,255,255,0.055);

        border: 1px solid rgba(255,255,255,0.09);

        border-radius: 16px;

        padding: 18px;

        text-align: center;

        margin-top: 25px;
    }

    .stat-number {
        color: white;

        font-size: 24px;

        font-weight: 850;
    }

    .stat-label {
        color: #94a3b8;

        font-size: 11px;

        margin-top: 3px;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    .stAlert {
        border-radius: 12px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;

        color: #94a3b8;

        margin-top: 55px;

        padding-top: 25px;
        padding-bottom: 10px;

        border-top:
            1px solid rgba(255,255,255,0.10);

        font-size: 12px;

        line-height: 1.8;
    }

    .footer strong {
        color: #e2e8f0;
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
            os.path.join(
                ARTIFACT_DIR,
                file_name
            )
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

    st.error(
        "❌ Unable to load the recommendation system."
    )

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
# BOOK INFORMATION VALIDATION
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
# HELPER FUNCTIONS
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


def safe_text(value, default="Unknown"):

    if pd.isna(value):
        return default

    value = str(value).strip()

    if not value:
        return default

    return html.escape(value)


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
        <div class="sidebar-brand">

            <div class="sidebar-logo">📚</div>

            <div class="sidebar-title">
                BookWise
            </div>

            <div class="sidebar-subtitle">
                AI Book Recommendation System
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section">Recommendation Models</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="model-item">📈 Popularity-Based</div>
        <div class="model-item">📖 Content-Based TF-IDF</div>
        <div class="model-item">👥 Collaborative Filtering</div>
        <div class="model-item">🧮 SVD</div>
        <div class="model-item">🤖 Hybrid Recommendation</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">Hybrid Configuration</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="weight-item">
            <span>Content</span>
            <span class="weight-value">
                {hybrid_config.get("content_weight", 0.30):.0%}
            </span>
        </div>

        <div class="weight-item">
            <span>Collaborative</span>
            <span class="weight-value">
                {hybrid_config.get("collaborative_weight", 0.30):.0%}
            </span>
        </div>

        <div class="weight-item">
            <span>SVD</span>
            <span class="weight-value">
                {hybrid_config.get("svd_weight", 0.20):.0%}
            </span>
        </div>

        <div class="weight-item">
            <span>Popularity</span>
            <span class="weight-value">
                {hybrid_config.get("popularity_weight", 0.20):.0%}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-number">5</div>
            <div class="stat-label">ML Recommendation Models</div>
        </div>

        <div class="stat-card">
            <div class="stat-number">10</div>
            <div class="stat-label">Books Recommended</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption(
        "Built with Python • Scikit-learn • Streamlit"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-icon">
            📚
        </div>

        <h1>
            Book<span class="hero-highlight">Wise</span>
        </h1>

        <p>
            Discover books you'll love through
            Machine Learning powered recommendations
            based on reading behavior, content similarity,
            collaborative patterns and popularity.
        </p>

        <div class="hero-badges">
            <span class="hero-badge">
                🤖 Machine Learning
            </span>

            <span class="hero-badge">
                📖 Content Intelligence
            </span>

            <span class="hero-badge">
                👥 Collaborative Filtering
            </span>

            <span class="hero-badge">
                ⚡ Hybrid Recommendation
            </span>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SEARCH PANEL
# ============================================================

st.markdown(
    """
    <div class="search-panel">

        <div class="search-title">
            🔍 Find Your Next Read
        </div>

        <div class="search-description">
            Enter your User ID and let BookWise generate
            personalized book recommendations.
        </div>

    """,
    unsafe_allow_html=True
)

search_col1, search_col2 = st.columns(
    [3, 1],
    gap="medium"
)

with search_col1:

    user_id_input = st.text_input(
        "User ID",
        placeholder="Example: 276704",
        label_visibility="collapsed"
    )

with search_col2:

    get_recommendations = st.button(
        "🚀 Get Recommendations",
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
        "🔮 Analyzing your reading preferences..."
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
        """
        <div class="section-header">

            <div>
                <div class="section-title">
                    ✨ Recommended For You
                </div>

                <div class="section-subtitle">
                    Personalized results generated by the Hybrid ML recommendation system
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # DISPLAY BOOKS IN 3-COLUMN GRID
    # ========================================================

    for row_start in range(
        0,
        len(recommendations),
        3
    ):

        row_books = recommendations[
            row_start:row_start + 3
        ]

        columns = st.columns(
            3,
            gap="large"
        )

        for position, isbn in enumerate(
            row_books
        ):

            rank = row_start + position + 1

            title = safe_text(
                title_map.get(
                    isbn,
                    "Unknown Title"
                ),
                "Unknown Title"
            )

            author = safe_text(
                author_map.get(
                    isbn,
                    "Unknown Author"
                ),
                "Unknown Author"
            )

            publisher = safe_text(
                publisher_map.get(
                    isbn,
                    "Unknown Publisher"
                ),
                "Unknown Publisher"
            )

            isbn_display = safe_text(
                isbn,
                "Unknown ISBN"
            )

            image_url = clean_image_url(
                image_map.get(
                    isbn,
                    ""
                )
            )

            with columns[position]:

                st.markdown(
                    '<div class="book-card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="rank-row">

                        <span class="rank">
                            #{rank}
                        </span>

                        <span class="ml-badge">
                            Hybrid Match
                        </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if image_url:

                    st.markdown(
                        f"""
                        <div class="book-image-container">

                            <img
                                src="{html.escape(image_url, quote=True)}"
                                alt="Book cover"
                            />

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="book-image-container">

                            <div class="book-placeholder">
                                📕
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f"""
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
                        ISBN: {isbn_display}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


# ============================================================
# INFORMATION SECTION
# ============================================================

if not get_recommendations:

    st.markdown(
        """
        <div class="info-title">
            🧠 How BookWise Works
        </div>
        """,
        unsafe_allow_html=True
    )

    info_col1, info_col2, info_col3 = st.columns(
        3,
        gap="large"
    )

    with info_col1:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-icon">
                    📖
                </div>

                <h3>
                    Content-Based
                </h3>

                <p>
                    Uses TF-IDF and cosine similarity
                    to identify books with similar
                    content characteristics such as
                    title, author and publisher.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with info_col2:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-icon">
                    👥
                </div>

                <h3>
                    Collaborative Filtering
                </h3>

                <p>
                    Finds users with similar reading
                    behavior and uses their interactions
                    to discover books that may interest
                    the target user.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with info_col3:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-icon">
                    🤖
                </div>

                <h3>
                    Hybrid Intelligence
                </h3>

                <p>
                    Combines Content, Collaborative,
                    SVD and Popularity signals into
                    a single recommendation ranking.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TECHNOLOGY SECTION
# ============================================================

if not get_recommendations:

    st.markdown(
        """
        <div class="info-title">
            ⚙️ Technology Stack
        </div>
        """,
        unsafe_allow_html=True
    )

    tech1, tech2, tech3, tech4 = st.columns(
        4,
        gap="medium"
    )

    with tech1:

        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">🐍</div>
                <div class="stat-label">Python</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tech2:

        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">🧠</div>
                <div class="stat-label">Scikit-learn</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tech3:

        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">📊</div>
                <div class="stat-label">Machine Learning</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tech4:

        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">🚀</div>
                <div class="stat-label">Streamlit</div>
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

        📚 <strong>BookWise</strong>
        — Machine Learning Book Recommendation System

        <br>

        Built with Python • Scikit-learn • Streamlit • Machine Learning

        <br>

        Hybrid Recommendation • Content Similarity • Collaborative Filtering • SVD

    </div>
    """,
    unsafe_allow_html=True
)
