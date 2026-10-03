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
    page_title="BookWise — Discover Your Next Story",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"


# ============================================================
# MODERN WEBSITE THEME
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL WEBSITE
       ===================================================== */

    .stApp {
        background: #f7f4ef;
        color: #191817;
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 0.8rem;
        padding-bottom: 4rem;
    }

    /* =====================================================
       REMOVE STREAMLIT TOP SPACE
       ===================================================== */

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* =====================================================
       TYPOGRAPHY
       ===================================================== */

    h1 {
        color: #191817 !important;
        font-size: 3.2rem !important;
        line-height: 1.05 !important;
        font-weight: 800 !important;
        letter-spacing: -2px !important;
    }

    h2 {
        color: #191817 !important;
        font-size: 2rem !important;
        font-weight: 750 !important;
        letter-spacing: -1px !important;
    }

    h3 {
        color: #191817 !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
    }

    p {
        color: #5f5a55;
        line-height: 1.65;
    }

    /* =====================================================
       NAVIGATION
       ===================================================== */

    .nav-brand {
        font-size: 1.35rem;
        font-weight: 800;
        color: #191817;
    }

    .nav-tag {
        font-size: 0.78rem;
        color: #8a8178;
        letter-spacing: 1px;
    }

    /* =====================================================
       HERO
       ===================================================== */

    .hero-background {
        background:
            linear-gradient(
                90deg,
                rgba(25, 24, 23, 0.90) 0%,
                rgba(25, 24, 23, 0.72) 45%,
                rgba(25, 24, 23, 0.28) 100%
            ),
            url("https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=1800&q=85");

        background-size: cover;
        background-position: center;
        border-radius: 28px;
        min-height: 430px;
        padding: 60px;
        margin-top: 20px;
        margin-bottom: 45px;
        display: flex;
        align-items: center;
        box-shadow:
            0 20px 60px rgba(45, 35, 28, 0.14);
    }

    .hero-kicker {
        color: #e3b894;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 15px;
    }

    .hero-heading {
        color: #ffffff;
        font-size: 3.5rem;
        line-height: 1.04;
        font-weight: 800;
        letter-spacing: -2px;
        max-width: 680px;
        margin-bottom: 18px;
    }

    .hero-text {
        color: #eee6de;
        font-size: 1.05rem;
        max-width: 600px;
        line-height: 1.7;
    }

    /* =====================================================
       SECTION LABEL
       ===================================================== */

    .section-label {
        color: #b66b4d;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* =====================================================
       SEARCH PANEL
       ===================================================== */

    .search-panel {
        background: #ffffff;
        border-radius: 22px;
        padding: 26px 30px;
        border: 1px solid #ebe5de;
        box-shadow:
            0 12px 35px rgba(48, 38, 30, 0.07);
        margin-bottom: 40px;
    }

    /* =====================================================
       INPUT
       ===================================================== */

    .stTextInput input {
        background: #faf9f7 !important;
        color: #191817 !important;
        border: 1px solid #ded8d1 !important;
        border-radius: 13px !important;
        height: 50px !important;
        font-size: 16px !important;
    }

    .stTextInput input:focus {
        border-color: #b66b4d !important;
        box-shadow: 0 0 0 1px #b66b4d !important;
    }

    /* =====================================================
       BUTTON
       ===================================================== */

    .stButton > button {
        background: #191817 !important;
        color: #ffffff !important;
        border: 1px solid #191817 !important;
        border-radius: 13px !important;
        height: 50px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #b66b4d !important;
        border-color: #b66b4d !important;
        color: #ffffff !important;
    }

    /* =====================================================
       BOOK CARDS
       ===================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid #ebe5de !important;
        border-radius: 18px !important;
        box-shadow:
            0 8px 28px rgba(48, 38, 30, 0.055);
    }

    /* =====================================================
       BOOK IMAGE
       ===================================================== */

    [data-testid="stImage"] img {
        border-radius: 10px !important;
        box-shadow:
            0 10px 24px rgba(25, 24, 23, 0.13);
    }

    /* =====================================================
       CAPTIONS
       ===================================================== */

    .stCaption {
        color: #91887f !important;
    }

    /* =====================================================
       DIVIDERS
       ===================================================== */

    hr {
        border-color: #e5ded6 !important;
    }

    /* =====================================================
       ALERTS
       ===================================================== */

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* =====================================================
       METRICS — SMALL, CLEAN
       ===================================================== */

    [data-testid="stMetric"] {
        background: transparent;
        border: none;
        padding: 5px 0;
    }

    [data-testid="stMetricLabel"] {
        color: #91887f !important;
    }

    [data-testid="stMetricValue"] {
        color: #191817 !important;
    }

    /* =====================================================
       FOOTER
       ===================================================== */

    .footer-text {
        color: #9a928a;
        font-size: 0.8rem;
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

    st.error(
        "BookWise could not load the model artifacts."
    )

    st.exception(e)

    st.stop()


# ============================================================
# BOOK DATA CLEANING
# ============================================================

book_info = book_info.copy()

for column in [
    "ISBN",
    "Book-Title",
    "Book-Author",
    "Publisher",
    "Image-URL-S",
    "Image-URL-M",
    "Image-URL-L"
]:

    if column in book_info.columns:

        book_info[column] = (
            book_info[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )


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
# HYBRID MODEL
# UNCHANGED
# ============================================================

def recommend_hybrid(user_id, top_n=10):

    content = recommend_content(user_id, top_n=100)

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
# BOOK HELPERS
# ============================================================

def get_book_info(isbn):

    result = book_info[
        book_info["ISBN"] == str(isbn)
    ]

    if result.empty:
        return None

    return result.iloc[0]


def get_book_image(book):

    if book is None:
        return None

    for column in [
        "Image-URL-M",
        "Image-URL-L",
        "Image-URL-S"
    ]:

        if column in book.index:

            url = str(
                book[column]
            ).strip()

            if (
                url
                and url.lower() != "nan"
                and url.startswith("http")
            ):

                return url

    return None


# ============================================================
# TOP NAVIGATION
# ============================================================

nav1, nav2, nav3, nav4 = st.columns(
    [4, 1, 1, 1]
)

with nav1:

    st.markdown(
        "### 📚 BookWise"
    )

with nav2:

    home = st.button(
        "Discover",
        use_container_width=True
    )

with nav3:

    recommendations_page = st.button(
        "Recommendations",
        use_container_width=True
    )

with nav4:

    about_page = st.button(
        "About",
        use_container_width=True
    )


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "discover"


if home:

    st.session_state.page = "discover"


if recommendations_page:

    st.session_state.page = "recommendations"


if about_page:

    st.session_state.page = "about"


# ============================================================
# DISCOVER
# ============================================================

if st.session_state.page == "discover":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="hero-background">
            <div>
                <div class="hero-kicker">
                    PERSONALIZED BOOK DISCOVERY
                </div>

                <div class="hero-heading">
                    Find your next<br>
                    unforgettable story.
                </div>

                <div class="hero-text">
                    BookWise learns from reading patterns,
                    similar readers, book content and popularity
                    to create a personal bookshelf just for you.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # INTRO
    # --------------------------------------------------------

    st.markdown(
        "DISCOVER",
        help="Explore BookWise"
    )

    st.header(
        "Books chosen around you."
    )

    st.write(
        "Not just popular books. Not just similar books. "
        "BookWise combines multiple signals to create a "
        "personalized Top 10."
    )

    st.write("")

    c1, c2, c3 = st.columns(
        3,
        gap="large"
    )

    with c1:

        with st.container(border=True):

            st.subheader(
                "📖 Understand"
            )

            st.write(
                "Book content is analyzed using "
                "TF-IDF and cosine similarity."
            )

    with c2:

        with st.container(border=True):

            st.subheader(
                "👥 Connect"
            )

            st.write(
                "Similar readers help discover "
                "books you may enjoy."
            )

    with c3:

        with st.container(border=True):

            st.subheader(
                "✨ Personalize"
            )

            st.write(
                "Four signals are combined by "
                "the Hybrid recommendation engine."
            )

    st.write("")
    st.divider()

    # --------------------------------------------------------
    # HYBRID SECTION
    # --------------------------------------------------------

    st.markdown(
        "THE ENGINE"
    )

    st.header(
        "One recommendation. Four perspectives."
    )

    st.write(
        "The deployed Hybrid model combines the same "
        "trained recommendation components."
    )

    st.write("")

    h1, h2, h3, h4 = st.columns(4)

    with h1:

        st.metric(
            "Content",
            "30%"
        )

    with h2:

        st.metric(
            "Collaborative",
            "30%"
        )

    with h3:

        st.metric(
            "SVD",
            "20%"
        )

    with h4:

        st.metric(
            "Popularity",
            "20%"
        )

    st.write("")
    st.divider()

    # --------------------------------------------------------
    # CALL TO ACTION
    # --------------------------------------------------------

    st.header(
        "Ready to find your next book?"
    )

    st.write(
        "Use your User ID to generate your personalized "
        "bookshelf."
    )

    if st.button(
        "Build My Bookshelf →",
        use_container_width=False
    ):

        st.session_state.page = "recommendations"

        st.rerun()


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif st.session_state.page == "recommendations":

    st.markdown(
        "YOUR PERSONAL LIBRARY"
    )

    st.header(
        "What should you read next?"
    )

    st.write(
        "Enter your User ID and let BookWise build "
        "a personalized list from the trained Hybrid model."
    )

    st.write("")

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    with st.container(border=True):

        input_col, button_col = st.columns(
            [4, 1],
            gap="medium"
        )

        with input_col:

            user_id_text = st.text_input(
                "User ID",
                value="276704",
                placeholder="Enter your User ID",
                label_visibility="visible"
            )

        with button_col:

            st.write("")

            generate = st.button(
                "Find Books ✨",
                use_container_width=True
            )

    # --------------------------------------------------------
    # RECOMMEND
    # --------------------------------------------------------

    if generate:

        try:

            user_id = int(
                user_id_text.strip()
            )

        except ValueError:

            st.error(
                "Please enter a valid numeric User ID."
            )

            st.stop()

        if user_id not in user_to_index:

            st.warning(
                "This User ID is not available in "
                "the trained recommendation model."
            )

            st.info(
                "Try demo User ID 276704."
            )

            st.stop()

        with st.spinner(
            "Curating your bookshelf..."
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

            st.stop()

        st.session_state.recommendations = recommendations
        st.session_state.recommended_user = user_id

    # --------------------------------------------------------
    # SHOW RESULTS FROM SESSION
    # --------------------------------------------------------

    if (
        "recommendations" in st.session_state
        and "recommended_user" in st.session_state
    ):

        recommendations = st.session_state.recommendations
        user_id = st.session_state.recommended_user

        st.write("")
        st.divider()

        result_col1, result_col2 = st.columns(
            [4, 1]
        )

        with result_col1:

            st.markdown(
                "PERSONALIZED FOR YOU"
            )

            st.header(
                "Your next 10 reads."
            )

            st.caption(
                f"Curated for User {user_id} using the Hybrid model"
            )

        with result_col2:

            st.metric(
                "Books",
                len(recommendations)
            )

        st.write("")

        # ----------------------------------------------------
        # BOOK GRID
        # ----------------------------------------------------

        for row_start in range(
            0,
            len(recommendations),
            5
        ):

            row_books = recommendations[
                row_start:row_start + 5
            ]

            columns = st.columns(
                len(row_books),
                gap="large"
            )

            for index, isbn in enumerate(
                row_books
            ):

                position = row_start + index + 1

                with columns[index]:

                    book = get_book_info(
                        isbn
                    )

                    with st.container(
                        border=True
                    ):

                        if book is None:

                            st.write(
                                "📖"
                            )

                            st.caption(
                                f"#{position}"
                            )

                            st.subheader(
                                "Book information unavailable"
                            )

                            st.caption(
                                f"ISBN {isbn}"
                            )

                            continue

                        image_url = get_book_image(
                            book
                        )

                        if image_url:

                            try:

                                st.image(
                                    image_url,
                                    use_container_width=True
                                )

                            except Exception:

                                st.write(
                                    "📖 Cover unavailable"
                                )

                        else:

                            st.write(
                                "📖 Cover unavailable"
                            )

                        st.caption(
                            f"#{position}"
                        )

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
                                ""
                            )
                        ).strip()

                        if not title:
                            title = "Unknown Title"

                        if not author:
                            author = "Unknown Author"

                        st.subheader(
                            title[:65]
                        )

                        st.write(
                            author[:45]
                        )

                        if publisher:

                            st.caption(
                                publisher[:45]
                            )

                        st.caption(
                            f"ISBN {isbn}"
                        )

        st.write("")
        st.divider()

        # ----------------------------------------------------
        # MODEL TRANSPARENCY
        # ----------------------------------------------------

        st.markdown(
            "WHY THESE BOOKS?"
        )

        st.header(
            "Powered by the Hybrid engine."
        )

        st.write(
            "The final ranking uses the same four trained "
            "recommendation signals."
        )

        w1, w2, w3, w4 = st.columns(4)

        with w1:

            st.write("📖 Content")
            st.caption("30%")

        with w2:

            st.write("👥 Collaborative")
            st.caption("30%")

        with w3:

            st.write("🧠 SVD")
            st.caption("20%")

        with w4:

            st.write("🔥 Popularity")
            st.caption("20%")


# ============================================================
# ABOUT
# ============================================================

elif st.session_state.page == "about":

    st.markdown(
        "ABOUT BOOKWISE"
    )

    st.header(
        "A machine-learning approach to book discovery."
    )

    st.write(
        "BookWise combines several recommendation approaches "
        "to produce one personalized ranking."
    )

    st.write("")
    st.divider()

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        with st.container(border=True):

            st.subheader(
                "📖 Content-Based"
            )

            st.write(
                "TF-IDF and cosine similarity identify "
                "books with similar content."
            )

            st.caption(
                "Hybrid weight · 30%"
            )

        st.write("")

        with st.container(border=True):

            st.subheader(
                "👥 Collaborative Filtering"
            )

            st.write(
                "KNN-based user similarity identifies "
                "recommendations from similar readers."
            )

            st.caption(
                "Hybrid weight · 30%"
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "🧠 SVD"
            )

            st.write(
                "Latent factors model hidden relationships "
                "between users and books."
            )

            st.caption(
                "Hybrid weight · 20%"
            )

        st.write("")

        with st.container(border=True):

            st.subheader(
                "🔥 Popularity"
            )

            st.write(
                "Popular books provide an additional "
                "recommendation signal."
            )

            st.caption(
                "Hybrid weight · 20%"
            )

    st.write("")
    st.divider()

    st.header(
        "Hybrid configuration"
    )

    model_table = pd.DataFrame(
        {
            "Model Component": [
                "Content-Based",
                "Collaborative",
                "SVD",
                "Popularity"
            ],
            "Method": [
                "TF-IDF + Cosine Similarity",
                "KNN",
                "Latent Factors",
                "Popularity Ranking"
            ],
            "Weight": [
                "30%",
                "30%",
                "20%",
                "20%"
            ]
        }
    )

    st.dataframe(
        model_table,
        use_container_width=True,
        hide_index=True
    )

    st.write("")
    st.info(
        "The user interface does not modify the trained "
        "recommendation model or its Hybrid weights."
    )


# ============================================================
# FOOTER
# ============================================================

st.write("")
st.divider()

st.caption(
    "BOOKWISE  ·  Personalized Book Discovery"
)

st.caption(
    "Machine Learning  ·  Hybrid Recommendation  ·  Streamlit"
)
