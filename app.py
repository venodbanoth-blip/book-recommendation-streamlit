import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BookWise | AI Book Recommendation",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"


# ============================================================
# LIGHT CHOCOLATE + BLACK THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ================================
       GLOBAL
       ================================ */

    .stApp {
        background-color: #F6EBDD;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ================================
       TEXT
       ================================ */

    h1, h2, h3, h4, h5, h6 {
        color: #1A1410 !important;
    }

    p, label {
        color: #3E3026 !important;
    }

    /* ================================
       SIDEBAR
       ================================ */

    section[data-testid="stSidebar"] {
        background-color: #E8D2BA;
        border-right: 1px solid #CDB092;
    }

    section[data-testid="stSidebar"] * {
        color: #1A1410 !important;
    }

    /* ================================
       HERO
       ================================ */

    .hero-card {
        background-color: #FFF8F0;
        border: 1px solid #D7BFA4;
        border-radius: 28px;
        padding: 45px;
        margin-bottom: 30px;
        box-shadow: 0 15px 40px rgba(70, 45, 25, 0.10);
    }

    .hero-title {
        font-size: 58px;
        font-weight: 850;
        letter-spacing: -2px;
        color: #17120E;
        margin: 0;
    }

    .hero-title-accent {
        color: #70452B;
    }

    .hero-subtitle {
        font-size: 20px;
        color: #675344;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    /* ================================
       BADGES
       ================================ */

    .badge {
        display: inline-block;
        background-color: #1A1410;
        color: #FFF8F0 !important;
        border-radius: 30px;
        padding: 8px 14px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 13px;
        font-weight: 600;
    }

    /* ================================
       SEARCH CARD
       ================================ */

    .search-card {
        background-color: #FFF8F0;
        border: 1px solid #D7BFA4;
        border-radius: 22px;
        padding: 25px;
        margin-bottom: 28px;
        box-shadow: 0 10px 28px rgba(70, 45, 25, 0.08);
    }

    /* ================================
       INPUT
       ================================ */

    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #BFA185 !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="input"] input {
        color: #17120E !important;
        background-color: #FFFFFF !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #8B7766 !important;
    }

    /* ================================
       BUTTON
       ================================ */

    .stButton > button {
        background-color: #1A1410;
        color: #FFF8F0;
        border: 1px solid #1A1410;
        border-radius: 12px;
        min-height: 48px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #70452B;
        color: #FFFFFF;
        border-color: #70452B;
        transform: translateY(-2px);
    }

    /* ================================
       BOOK CARD
       ================================ */

    .book-card {
        background-color: #FFF8F0;
        border: 1px solid #D7BFA4;
        border-radius: 20px;
        padding: 18px;
        margin-bottom: 20px;
        box-shadow: 0 10px 28px rgba(70, 45, 25, 0.08);
    }

    .rank-badge {
        display: inline-block;
        background-color: #E5CEB5;
        color: #4C301E !important;
        border-radius: 20px;
        padding: 6px 10px;
        font-size: 11px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .book-title-text {
        font-size: 19px;
        font-weight: 800;
        color: #17120E !important;
        line-height: 1.35;
        margin-top: 10px;
    }

    .book-author-text {
        font-size: 14px;
        color: #634F40 !important;
        margin-top: 7px;
    }

    .book-publisher-text {
        font-size: 13px;
        color: #806C5B !important;
        margin-top: 5px;
    }

    .book-isbn-text {
        font-size: 12px;
        color: #9A8877 !important;
        margin-top: 5px;
    }

    /* ================================
       INFO CARDS
       ================================ */

    .info-card {
        background-color: #FFF8F0;
        border: 1px solid #D7BFA4;
        border-radius: 20px;
        padding: 25px;
        min-height: 220px;
        box-shadow: 0 10px 25px rgba(70, 45, 25, 0.07);
    }

    .info-text {
        color: #675344 !important;
        line-height: 1.65;
        font-size: 14px;
    }

    /* ================================
       TECHNOLOGY CARDS
       ================================ */

    .tech-card {
        background-color: #E8D2BA;
        border: 1px solid #CDB092;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
    }

    /* ================================
       SIDEBAR BOX
       ================================ */

    .sidebar-box {
        background-color: #F6EBDD;
        border: 1px solid #CDB092;
        border-radius: 15px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    /* ================================
       FOOTER
       ================================ */

    .footer-line {
        border-top: 1px solid #D1B79A;
        margin-top: 45px;
        padding-top: 22px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    tfidf_vectorizer = joblib.load(
        ARTIFACT_DIR / "tfidf_vectorizer.pkl"
    )

    tfidf_matrix = joblib.load(
        ARTIFACT_DIR / "tfidf_matrix.pkl"
    )

    isbn_to_index = joblib.load(
        ARTIFACT_DIR / "isbn_to_index.pkl"
    )

    knn_model = joblib.load(
        ARTIFACT_DIR / "knn_model.pkl"
    )

    user_book_sparse = joblib.load(
        ARTIFACT_DIR / "user_book_sparse.pkl"
    )

    user_to_index = joblib.load(
        ARTIFACT_DIR / "user_to_index.pkl"
    )

    svd_model = joblib.load(
        ARTIFACT_DIR / "svd_model.pkl"
    )

    user_factors = joblib.load(
        ARTIFACT_DIR / "user_factors.pkl"
    )

    book_factors = joblib.load(
        ARTIFACT_DIR / "book_factors.pkl"
    )

    train_user_ids = joblib.load(
        ARTIFACT_DIR / "train_user_ids.pkl"
    )

    train_isbn_ids = joblib.load(
        ARTIFACT_DIR / "train_isbn_ids.pkl"
    )

    popularity_rank = joblib.load(
        ARTIFACT_DIR / "popularity_rank.pkl"
    )

    book_info = joblib.load(
        ARTIFACT_DIR / "book_info.pkl"
    )

    train_history = joblib.load(
        ARTIFACT_DIR / "train_history.pkl"
    )

    with open(
        ARTIFACT_DIR / "hybrid_config.json",
        "r"
    ) as f:
        hybrid_config = json.load(f)

    return (
        tfidf_vectorizer,
        tfidf_matrix,
        isbn_to_index,
        knn_model,
        user_book_sparse,
        user_to_index,
        svd_model,
        user_factors,
        book_factors,
        train_user_ids,
        train_isbn_ids,
        popularity_rank,
        book_info,
        train_history,
        hybrid_config
    )


# ============================================================
# LOAD
# ============================================================

try:

    (
        tfidf_vectorizer,
        tfidf_matrix,
        isbn_to_index,
        knn_model,
        user_book_sparse,
        user_to_index,
        svd_model,
        user_factors,
        book_factors,
        train_user_ids,
        train_isbn_ids,
        popularity_rank,
        book_info,
        train_history,
        hybrid_config
    ) = load_artifacts()

except Exception as e:

    st.error("Unable to load model artifacts.")
    st.exception(e)
    st.stop()


# ============================================================
# RECOMMENDATION FUNCTIONS
# EXACT HYBRID LOGIC
# ============================================================

def recommend_popularity(user_id, top_n=10):

    history = set(
        train_history[
            train_history["User-ID"] == user_id
        ]["ISBN"]
    )

    result = popularity_rank[
        ~popularity_rank["ISBN"].isin(history)
    ].head(top_n)

    return result["ISBN"].tolist()


def recommend_content(user_id, top_n=100):

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


def recommend_collaborative(user_id, top_n=100):

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


def recommend_svd(user_id, top_n=100):

    if user_id not in user_to_index:
        return []

    user_index = user_to_index[user_id]

    user_vector = user_factors[user_index]

    predicted_scores = np.dot(
        user_vector,
        book_factors
    )

    rated_indices = set(
        user_book_sparse[
            user_index
        ].indices
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


def recommend_hybrid(user_id, top_n=10):

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
        | set(collaborative)
        | set(svd)
        | set(popularity)
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

            + collaborative_weight
            * collaborative_scores.get(isbn, 0)

            + svd_weight
            * svd_scores.get(isbn, 0)

            + popularity_weight
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

    st.title("📚 BookWise")

    st.caption(
        "AI Book Recommendation System"
    )

    st.divider()

    st.subheader("🤖 Recommendation Models")

    st.markdown(
        """
        📈 **Popularity-Based**

        📖 **Content-Based TF-IDF**

        👥 **Collaborative Filtering**

        🧮 **SVD**

        🤖 **Hybrid Recommendation**
        """
    )

    st.divider()

    st.subheader("⚖️ Hybrid Weights")

    st.markdown(
        """
        | Component | Weight |
        |---|---:|
        | Content | **30%** |
        | Collaborative | **30%** |
        | SVD | **20%** |
        | Popularity | **20%** |
        """
    )

    st.divider()

    st.subheader("📊 System")

    st.metric(
        "Recommendation Models",
        "5"
    )

    st.metric(
        "Books per Recommendation",
        "10"
    )

    st.divider()

    st.caption(
        "BookWise • Machine Learning"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div class="hero-card">',
    unsafe_allow_html=True
)

st.markdown("### 📚")

st.markdown(
    """
    # Book<span class="hero-title-accent">Wise</span>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle">
        AI Book Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Discover books you'll love using Machine Learning "
    "powered recommendations based on reading behavior, "
    "content similarity, collaborative patterns and popularity."
)

st.markdown(
    """
    <span class="badge">🤖 Machine Learning</span>
    <span class="badge">📖 Content Intelligence</span>
    <span class="badge">👥 Collaborative Filtering</span>
    <span class="badge">⚡ Hybrid Recommendation</span>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# FIND YOUR NEXT READ
# ============================================================

st.header("🔍 Find Your Next Read")

st.write(
    "Enter your User ID and BookWise will generate "
    "personalized recommendations."
)

user_id_input = st.text_input(
    "User ID",
    placeholder="Example: 276704"
)

recommend_button = st.button(
    "✨ Get My Recommendations",
    type="primary",
    use_container_width=True
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

if recommend_button:

    if not user_id_input.strip():

        st.warning(
            "Please enter a User ID."
        )

    else:

        try:

            user_id = int(
                user_id_input.strip()
            )

        except ValueError:

            st.error(
                "User ID must be a number."
            )

            st.stop()

        if user_id not in user_to_index:

            st.warning(
                "User ID not found or no recommendations "
                "are available for this user."
            )

        else:

            with st.spinner(
                "Finding books you'll love..."
            ):

                recommendations = recommend_hybrid(
                    user_id,
                    top_n=10
                )

            if not recommendations:

                st.warning(
                    "No recommendations are available "
                    "for this user."
                )

            else:

                st.success(
                    f"✨ {len(recommendations)} personalized "
                    "recommendations generated using Hybrid Intelligence."
                )

                st.header("📚 Your Recommended Books")

                for start in range(
                    0,
                    len(recommendations),
                    3
                ):

                    row_books = recommendations[
                        start:start + 3
                    ]

                    columns = st.columns(3)

                    for offset, isbn in enumerate(
                        row_books
                    ):

                        with columns[offset]:

                            rank = start + offset + 1

                            book_rows = book_info[
                                book_info["ISBN"] == isbn
                            ]

                            if book_rows.empty:
                                continue

                            book = book_rows.iloc[0]

                            title = str(
                                book.get(
                                    "Book-Title",
                                    "Unknown Title"
                                )
                            ).strip()

                            author = str(
                                book.get(
                                    "Book-Author",
                                    "Unknown Author"
                                )
                            ).strip()

                            publisher = str(
                                book.get(
                                    "Publisher",
                                    "Unknown Publisher"
                                )
                            ).strip()

                            image_url = str(
                                book.get(
                                    "Image-URL-M",
                                    ""
                                )
                            ).strip()

                            st.markdown(
                                f"""
                                <div class="book-card">
                                    <div class="rank-badge">
                                        #{rank} • HYBRID MATCH
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            if image_url:

                                try:

                                    st.image(
                                        image_url,
                                        use_container_width=True
                                    )

                                except Exception:

                                    st.info(
                                        "📚 Cover image unavailable"
                                    )

                            else:

                                st.info(
                                    "📚 Cover image unavailable"
                                )

                            st.markdown(
                                f"""
                                <div class="book-title-text">
                                    {title}
                                </div>

                                <div class="book-author-text">
                                    ✍️ {author}
                                </div>

                                <div class="book-publisher-text">
                                    🏢 {publisher}
                                </div>

                                <div class="book-isbn-text">
                                    ISBN: {isbn}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            st.write("")


# ============================================================
# HOW BOOKWISE WORKS
# ============================================================

st.header("🧠 How BookWise Works")

st.write(
    "Four machine-learning signals work together to create "
    "the final Hybrid recommendation ranking."
)

info1, info2, info3 = st.columns(3)

with info1:

    st.markdown(
        """
        <div class="info-card">
        <h3>📖 Content-Based</h3>
        <div class="info-text">
        Uses TF-IDF and cosine similarity to identify
        books with similar content characteristics
        such as title, author and publisher.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with info2:

    st.markdown(
        """
        <div class="info-card">
        <h3>👥 Collaborative Filtering</h3>
        <div class="info-text">
        Finds users with similar reading behavior
        and uses their interactions to discover
        books that may interest the target user.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with info3:

    st.markdown(
        """
        <div class="info-card">
        <h3>🤖 Hybrid Intelligence</h3>
        <div class="info-text">
        Combines Content, Collaborative, SVD and
        Popularity signals into one final
        recommendation ranking.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.header("⚙️ Technology Stack")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:

    st.markdown(
        """
        <div class="tech-card">
        <h2>🐍</h2>
        <b>Python</b>
        </div>
        """,
        unsafe_allow_html=True
    )

with tech2:

    st.markdown(
        """
        <div class="tech-card">
        <h2>🧠</h2>
        <b>Scikit-learn</b>
        </div>
        """,
        unsafe_allow_html=True
    )

with tech3:

    st.markdown(
        """
        <div class="tech-card">
        <h2>📊</h2>
        <b>Machine Learning</b>
        </div>
        """,
        unsafe_allow_html=True
    )

with tech4:

    st.markdown(
        """
        <div class="tech-card">
        <h2>🚀</h2>
        <b>Streamlit</b>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-line"></div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "📚 BookWise — Machine Learning Book Recommendation System"
)

st.caption(
    "Built with Python • Scikit-learn • Streamlit"
)

st.caption(
    "Hybrid Recommendation • Content Similarity • "
    "Collaborative Filtering • SVD"
)
