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
# MODERN LIGHT CHOCOLATE + BLACK DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #F7EFE5;
        color: #17130F;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    p, span, div, label {
        color: #17130F;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #E9D5BF;
        border-right: 1px solid #D2B99D;
    }

    section[data-testid="stSidebar"] * {
        color: #17130F !important;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #17130F !important;
    }

    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        background: linear-gradient(
            135deg,
            #FFF9F2 0%,
            #E8D2BA 100%
        );

        border: 1px solid #D6B99A;
        border-radius: 28px;
        padding: 55px 55px 48px 55px;
        margin-bottom: 28px;

        box-shadow:
            0 18px 45px rgba(71, 45, 25, 0.10);
    }

    .hero-icon {
        font-size: 54px;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 58px;
        line-height: 1.05;
        font-weight: 850;
        letter-spacing: -2px;
        color: #17130F;
    }

    .hero-title span {
        color: #70452A;
    }

    .hero-description {
        max-width: 800px;
        margin-top: 16px;
        font-size: 18px;
        line-height: 1.7;
        color: #59483A;
    }

    .badge-row {
        margin-top: 25px;
    }

    .badge {
        display: inline-block;
        background: #17130F;
        color: #FFF8EF !important;
        border-radius: 999px;
        padding: 9px 15px;
        margin-right: 7px;
        margin-bottom: 7px;
        font-size: 13px;
        font-weight: 600;
    }

    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-title {
        font-size: 31px;
        font-weight: 800;
        color: #17130F;
        margin-top: 32px;
        margin-bottom: 7px;
        letter-spacing: -0.5px;
    }

    .section-description {
        font-size: 16px;
        color: #725F4F;
        margin-bottom: 20px;
    }

    /* ========================================================
       SEARCH CARD
       ======================================================== */

    .search-card {
        background: #FFF9F2;
        border: 1px solid #D8BFA5;
        border-radius: 22px;
        padding: 25px;
        margin-top: 15px;
        margin-bottom: 30px;
        box-shadow:
            0 12px 30px rgba(71, 45, 25, 0.08);
    }

    /* ========================================================
       INPUT
       ======================================================== */

    div[data-baseweb="input"] {
        background: #FFFFFF !important;
        border-radius: 12px !important;
        border: 1px solid #C8AA8D !important;
    }

    div[data-baseweb="input"] input {
        color: #17130F !important;
        background: #FFFFFF !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #8A7767 !important;
    }

    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        width: 100%;
        min-height: 48px;

        background: #17130F;
        color: #FFF9F2;

        border: 1px solid #17130F;
        border-radius: 12px;

        font-size: 15px;
        font-weight: 700;

        transition:
            transform 0.2s ease,
            background 0.2s ease;
    }

    .stButton > button:hover {
        background: #70452A;
        color: #FFFFFF;
        border-color: #70452A;
        transform: translateY(-2px);
    }

    /* ========================================================
       RECOMMENDATION CARDS
       ======================================================== */

    .book-card {
        background: #FFF9F2;
        border: 1px solid #D8BFA5;
        border-radius: 20px;

        padding: 17px;
        margin-bottom: 22px;

        box-shadow:
            0 12px 30px rgba(71, 45, 25, 0.09);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .book-card:hover {
        transform: translateY(-5px);

        box-shadow:
            0 18px 38px rgba(71, 45, 25, 0.15);
    }

    .rank {
        display: inline-block;

        background: #E9D5BF;
        color: #4A2F1D !important;

        border-radius: 999px;
        padding: 6px 10px;

        font-size: 11px;
        font-weight: 800;

        margin-bottom: 12px;
    }

    .book-title {
        font-size: 19px;
        line-height: 1.35;

        font-weight: 800;

        color: #17130F !important;

        margin-top: 12px;
        margin-bottom: 8px;
    }

    .book-author {
        font-size: 14px;
        color: #634F3F !important;
        margin-bottom: 5px;
    }

    .book-publisher {
        font-size: 13px;
        color: #806D5C !important;
        margin-bottom: 4px;
    }

    .book-isbn {
        font-size: 12px;
        color: #9A8877 !important;
    }

    /* ========================================================
       INFO CARDS
       ======================================================== */

    .info-card {
        background: #FFF9F2;
        border: 1px solid #D8BFA5;
        border-radius: 20px;
        padding: 25px;

        min-height: 210px;

        box-shadow:
            0 10px 25px rgba(71, 45, 25, 0.07);
    }

    .info-icon {
        font-size: 34px;
        margin-bottom: 10px;
    }

    .info-title {
        font-size: 19px;
        font-weight: 800;
        margin-bottom: 9px;
    }

    .info-text {
        font-size: 14px;
        line-height: 1.7;
        color: #6D5948 !important;
    }

    /* ========================================================
       TECHNOLOGY CARDS
       ======================================================== */

    .tech-card {
        background: #E9D5BF;
        border: 1px solid #D0B294;
        border-radius: 18px;

        padding: 22px;

        text-align: center;

        box-shadow:
            0 8px 20px rgba(71, 45, 25, 0.06);
    }

    .tech-icon {
        font-size: 30px;
        margin-bottom: 8px;
    }

    .tech-name {
        font-weight: 800;
        font-size: 15px;
    }

    /* ========================================================
       SIDEBAR CARDS
       ======================================================== */

    .sidebar-card {
        background: #F7EBDD;
        border: 1px solid #D2B99D;
        border-radius: 16px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .sidebar-title {
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .weight-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        font-size: 13px;
    }

    .weight-value {
        font-weight: 800;
        color: #70452A !important;
    }

    /* ========================================================
       SUCCESS MESSAGE
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        margin-top: 55px;
        padding-top: 25px;

        border-top: 1px solid #D6B99A;

        text-align: center;

        color: #806D5C !important;

        font-size: 13px;
        line-height: 1.8;
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

    st.markdown(
        """
        <div style="font-size:35px;">📚</div>

        <div style="
            font-size:27px;
            font-weight:850;
            margin-bottom:3px;
        ">
            BookWise
        </div>

        <div style="
            font-size:13px;
            color:#725F4F !important;
        ">
            AI Book Recommendation System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🤖 Recommendation Models")

    st.markdown(
        """
        <div class="sidebar-card">

        <div class="sidebar-title">
        📈 Popularity-Based
        </div>

        <div class="sidebar-title">
        📖 Content-Based TF-IDF
        </div>

        <div class="sidebar-title">
        👥 Collaborative Filtering
        </div>

        <div class="sidebar-title">
        🧮 SVD
        </div>

        <div class="sidebar-title">
        🤖 Hybrid Recommendation
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚖️ Hybrid Weights")

    st.markdown(
        """
        <div class="sidebar-card">

        <div class="weight-row">
            <span>Content</span>
            <span class="weight-value">30%</span>
        </div>

        <div class="weight-row">
            <span>Collaborative</span>
            <span class="weight-value">30%</span>
        </div>

        <div class="weight-row">
            <span>SVD</span>
            <span class="weight-value">20%</span>
        </div>

        <div class="weight-row">
            <span>Popularity</span>
            <span class="weight-value">20%</span>
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📊 System")

    st.metric(
        "Recommendation Models",
        "5"
    )

    st.metric(
        "Books per Recommendation",
        "10"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-icon">
            📚
        </div>

        <div class="hero-title">
            Book<span>Wise</span>
        </div>

        <div class="hero-description">
            Discover books you'll love using Machine Learning
            powered recommendations based on reading behavior,
            content similarity, collaborative patterns and
            popularity.
        </div>

        <div class="badge-row">

            <span class="badge">
                🤖 Machine Learning
            </span>

            <span class="badge">
                📖 Content Intelligence
            </span>

            <span class="badge">
                👥 Collaborative Filtering
            </span>

            <span class="badge">
                ⚡ Hybrid Recommendation
            </span>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SEARCH
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🔍 Find Your Next Read
    </div>

    <div class="section-description">
        Enter your User ID and BookWise will generate
        personalized recommendations.
    </div>
    """,
    unsafe_allow_html=True
)

user_id_input = st.text_input(
    "User ID",
    placeholder="Example: 276704",
    label_visibility="collapsed"
)

recommend_button = st.button(
    "✨ Get My Recommendations",
    type="primary"
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

                st.markdown(
                    """
                    <div class="section-title">
                        📚 Your Recommended Books
                    </div>
                    """,
                    unsafe_allow_html=True
                )

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

                                    <div class="rank">
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

                                    st.markdown(
                                        """
                                        <div style="
                                            height:300px;
                                            display:flex;
                                            align-items:center;
                                            justify-content:center;
                                            background:#E9D5BF;
                                            border-radius:15px;
                                            font-size:50px;
                                        ">
                                        📚
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                            else:

                                st.markdown(
                                    """
                                    <div style="
                                        height:300px;
                                        display:flex;
                                        align-items:center;
                                        justify-content:center;
                                        background:#E9D5BF;
                                        border-radius:15px;
                                        font-size:50px;
                                    ">
                                    📚
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
                                    ✍️ {author}
                                </div>

                                <div class="book-publisher">
                                    🏢 {publisher}
                                </div>

                                <div class="book-isbn">
                                    ISBN: {isbn}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            st.write("")


# ============================================================
# HOW BOOKWISE WORKS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🧠 How BookWise Works
    </div>

    <div class="section-description">
        Four machine-learning signals work together to create
        the final Hybrid recommendation ranking.
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                📖
            </div>

            <div class="info-title">
                Content-Based
            </div>

            <div class="info-text">
                Uses TF-IDF and cosine similarity
                to identify books with similar
                content characteristics such as
                title, author and publisher.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                👥
            </div>

            <div class="info-title">
                Collaborative Filtering
            </div>

            <div class="info-text">
                Finds users with similar reading
                behavior and uses their interactions
                to discover books that may interest
                the target user.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                🤖
            </div>

            <div class="info-title">
                Hybrid Intelligence
            </div>

            <div class="info-text">
                Combines Content, Collaborative,
                SVD and Popularity signals into
                one final recommendation ranking.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.markdown(
    """
    <div class="section-title">
        ⚙️ Technology Stack
    </div>
    """,
    unsafe_allow_html=True
)

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:

    st.markdown(
        """
        <div class="tech-card">
            <div class="tech-icon">🐍</div>
            <div class="tech-name">Python</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with tech2:

    st.markdown(
        """
        <div class="tech-card">
            <div class="tech-icon">🧠</div>
            <div class="tech-name">Scikit-learn</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with tech3:

    st.markdown(
        """
        <div class="tech-card">
            <div class="tech-icon">📊</div>
            <div class="tech-name">Machine Learning</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with tech4:

    st.markdown(
        """
        <div class="tech-card">
            <div class="tech-icon">🚀</div>
            <div class="tech-name">Streamlit</div>
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

        📚 <b>BookWise</b> —
        Machine Learning Book Recommendation System

        <br>

        Built with Python • Scikit-learn • Streamlit

        <br>

        Hybrid Recommendation • Content Similarity •
        Collaborative Filtering • SVD

    </div>
    """,
    unsafe_allow_html=True
)
