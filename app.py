import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BookWise",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"


# ============================================================
# MODERN BOOKWISE THEME
# CSS IS ONLY FOR STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(184, 134, 80, 0.13),
                transparent 28%
            ),
            radial-gradient(
                circle at 10% 30%,
                rgba(111, 72, 45, 0.10),
                transparent 30%
            ),
            #17110e;

        color: #f4eadc;
    }

    .main .block-container {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {
        background: #211713;
        border-right: 1px solid #39271f;
    }

    [data-testid="stSidebar"] * {
        color: #f4eadc !important;
    }

    /* ======================================================
       TEXT
       ====================================================== */

    h1 {
        color: #f8efe4 !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px !important;
    }

    h2 {
        color: #f5e7d6 !important;
        font-weight: 750 !important;
        letter-spacing: -0.8px !important;
    }

    h3 {
        color: #f1dfcb !important;
        font-weight: 700 !important;
    }

    p {
        color: #cdbca9;
    }

    .stCaption {
        color: #9e8c7b !important;
    }

    /* ======================================================
       INPUT
       ====================================================== */

    .stTextInput input {
        background: #241914 !important;
        color: #f8efe4 !important;
        border: 1px solid #493229 !important;
        border-radius: 14px !important;
        height: 48px !important;
    }

    .stTextInput input:focus {
        border: 1px solid #c89a61 !important;
        box-shadow: 0 0 0 1px #c89a61 !important;
    }

    /* ======================================================
       BUTTON
       ====================================================== */

    .stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 14px;
        border: 1px solid #c89a61;
        background: #c89a61;
        color: #1b120d;
        font-weight: 750;
        font-size: 15px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #e0b77d;
        border-color: #e0b77d;
        color: #17100c;
        transform: translateY(-1px);
    }

    /* ======================================================
       RADIO
       ====================================================== */

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent;
        border-radius: 10px;
        padding: 8px 10px;
    }

    /* ======================================================
       CONTAINERS
       ====================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(38, 26, 20, 0.86);
        border: 1px solid #3c2a21 !important;
        border-radius: 18px !important;
    }

    /* ======================================================
       BOOK IMAGES
       ====================================================== */

    [data-testid="stImage"] img {
        border-radius: 12px;
        object-fit: cover;
        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.30);
    }

    /* ======================================================
       ALERTS
       ====================================================== */

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* ======================================================
       DIVIDER
       ====================================================== */

    hr {
        border-color: #38271f !important;
    }

    /* ======================================================
       METRICS
       ====================================================== */

    [data-testid="stMetric"] {
        background: #211713;
        border: 1px solid #3c2a21;
        border-radius: 14px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #a99682 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f5e7d6 !important;
    }

    /* ======================================================
       DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL ARTIFACTS
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

    st.error("Model artifacts could not be loaded.")

    st.exception(e)

    st.stop()


# ============================================================
# CLEAN BOOK INFORMATION
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
# EXACTLY UNCHANGED
# ============================================================

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
            content_weight * content_scores.get(isbn, 0)
            + collaborative_weight * collaborative_scores.get(isbn, 0)
            + svd_weight * svd_scores.get(isbn, 0)
            + popularity_weight * popularity_scores.get(isbn, 0)
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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 BookWise")

    st.caption(
        "Personalized book discovery"
    )

    st.divider()

    page = st.radio(
        "Explore",
        [
            "Discover",
            "Recommendations",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("HYBRID ENGINE")

    st.write(
        "Four recommendation signals working together."
    )

    st.caption(
        "Content 30%  ·  Collaborative 30%"
    )

    st.caption(
        "SVD 20%  ·  Popularity 20%"
    )

    st.divider()

    st.caption(
        "BookWise ML Recommendation System"
    )


# ============================================================
# DISCOVER PAGE
# ============================================================

if page == "Discover":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    st.title("📚 BookWise")

    st.subheader(
        "Find a story worth getting lost in."
    )

    st.write(
        "Personalized recommendations built from your "
        "reading behavior, similar readers, book content, "
        "and popularity patterns."
    )

    st.write("")

    hero_left, hero_right = st.columns(
        [3, 2],
        gap="large"
    )

    with hero_left:

        with st.container(border=True):

            st.caption("PERSONALIZED DISCOVERY")

            st.header(
                "Your next favorite book is waiting."
            )

            st.write(
                "BookWise combines Content-Based filtering, "
                "Collaborative Filtering, SVD and Popularity "
                "into one Hybrid recommendation engine."
            )

            st.write("")

            st.button(
                "Go to Recommendations →",
                key="discover_button",
                use_container_width=True
            )

    with hero_right:

        with st.container(border=True):

            st.caption("HYBRID MODEL")

            st.metric(
                "Recommendation Engine",
                "Hybrid"
            )

            st.write(
                "30% Content · 30% Collaborative · "
                "20% SVD · 20% Popularity"
            )

    st.write("")
    st.divider()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.header("One system. Four signals.")

    col1, col2 = st.columns(2, gap="large")

    with col1:

        with st.container(border=True):

            st.subheader("📖 Content-Based")

            st.write(
                "Understands book similarity through "
                "TF-IDF representations and cosine similarity."
            )

            st.caption("30% of Hybrid ranking")

        st.write("")

        with st.container(border=True):

            st.subheader("🧠 SVD")

            st.write(
                "Learns hidden patterns from user-book "
                "interactions using latent factors."
            )

            st.caption("20% of Hybrid ranking")

    with col2:

        with st.container(border=True):

            st.subheader("👥 Collaborative")

            st.write(
                "Finds users with similar reading behavior "
                "and uses their preferences."
            )

            st.caption("30% of Hybrid ranking")

        st.write("")

        with st.container(border=True):

            st.subheader("🔥 Popularity")

            st.write(
                "Adds popular books to improve candidate "
                "coverage and discovery."
            )

            st.caption("20% of Hybrid ranking")

    st.write("")
    st.divider()

    # --------------------------------------------------------
    # SAMPLE USER
    # --------------------------------------------------------

    st.header("Ready to discover?")

    st.write(
        "Use a trained User ID to generate a personalized "
        "Top 10 recommendation list."
    )

    if 276704 in user_to_index:

        st.info(
            "Demo User ID available: 276704"
        )


# ============================================================
# RECOMMENDATIONS PAGE
# ============================================================

elif page == "Recommendations":

    st.title("Your Recommendations")

    st.write(
        "Tell us who you are, and BookWise will build "
        "a personalized reading list."
    )

    st.write("")

    # --------------------------------------------------------
    # SEARCH AREA
    # --------------------------------------------------------

    with st.container(border=True):

        st.caption("PERSONALIZED SEARCH")

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
                "Recommend ✨",
                use_container_width=True
            )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if generate:

        try:

            user_id = int(
                user_id_text.strip()
            )

        except ValueError:

            st.error(
                "Please enter a numeric User ID."
            )

            st.stop()

        if user_id not in user_to_index:

            st.warning(
                "This User ID is not available in "
                "the trained recommendation model."
            )

            st.info(
                "Try the demo User ID: 276704"
            )

            st.stop()

        with st.spinner(
            "Curating your personal bookshelf..."
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

        # ----------------------------------------------------
        # RESULT INTRO
        # ----------------------------------------------------

        st.write("")

        result_col1, result_col2 = st.columns(
            [3, 1]
        )

        with result_col1:

            st.header(
                "Picked for you"
            )

            st.caption(
                f"Personalized reading list for User {user_id}"
            )

        with result_col2:

            st.metric(
                "Books Found",
                len(recommendations)
            )

        st.write("")

        # ----------------------------------------------------
        # HYBRID EXPLANATION
        # ----------------------------------------------------

        with st.container(border=True):

            st.caption(
                "POWERED BY HYBRID RECOMMENDATION"
            )

            st.subheader(
                "Four signals. One personalized ranking."
            )

            w1, w2, w3, w4 = st.columns(4)

            with w1:

                st.write("📖 Content")
                st.write("**30%**")

            with w2:

                st.write("👥 Collaborative")
                st.write("**30%**")

            with w3:

                st.write("🧠 SVD")
                st.write("**20%**")

            with w4:

                st.write("🔥 Popularity")
                st.write("**20%**")

        st.write("")
        st.divider()

        # ----------------------------------------------------
        # BOOKS
        # ----------------------------------------------------

        st.header(
            "Your personal bookshelf"
        )

        st.caption(
            "Top 10 recommendations generated by the Hybrid model"
        )

        st.write("")

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
                gap="medium"
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

                        if book is not None:

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
                                title[:70]
                            )

                            st.write(
                                f"**{author[:45]}**"
                            )

                            if publisher:

                                st.caption(
                                    publisher[:45]
                                )

                            st.caption(
                                f"ISBN {isbn}"
                            )

                        else:

                            st.caption(
                                f"#{position}"
                            )

                            st.subheader(
                                "Book information unavailable"
                            )

                            st.caption(
                                f"ISBN {isbn}"
                            )

        st.write("")
        st.divider()

        st.caption(
            "BookWise recommendations are generated from "
            "the trained Hybrid recommendation system."
        )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.title("About BookWise")

    st.write(
        "BookWise is a machine-learning recommendation "
        "system designed to combine multiple recommendation "
        "strategies into one personalized experience."
    )

    st.write("")
    st.divider()

    st.header("The Hybrid Engine")

    st.write(
        "The deployed recommendation engine combines "
        "four ranked candidate lists."
    )

    st.write("")

    col1, col2 = st.columns(2, gap="large")

    with col1:

        with st.container(border=True):

            st.subheader("📖 Content-Based")

            st.write(
                "TF-IDF representations are used to measure "
                "similarity between books."
            )

            st.caption(
                "Weight: 30%"
            )

        st.write("")

        with st.container(border=True):

            st.subheader("👥 Collaborative")

            st.write(
                "KNN identifies similar users from their "
                "book interaction patterns."
            )

            st.caption(
                "Weight: 30%"
            )

    with col2:

        with st.container(border=True):

            st.subheader("🧠 SVD")

            st.write(
                "Latent factors capture hidden relationships "
                "between users and books."
            )

            st.caption(
                "Weight: 20%"
            )

        st.write("")

        with st.container(border=True):

            st.subheader("🔥 Popularity")

            st.write(
                "Popularity ranking provides an additional "
                "candidate signal."
            )

            st.caption(
                "Weight: 20%"
            )

    st.write("")
    st.divider()

    st.header("Final Hybrid Weights")

    model_table = pd.DataFrame(
        {
            "Recommendation Method": [
                "Content-Based",
                "Collaborative",
                "SVD",
                "Popularity"
            ],
            "Technique": [
                "TF-IDF + Cosine Similarity",
                "KNN User Similarity",
                "Latent Factor Model",
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
    st.divider()

    st.info(
        "The deployed application loads the existing trained "
        "model artifacts. The recommendation logic and Hybrid "
        "weights are not modified by this interface."
    )


# ============================================================
# FOOTER
# ============================================================

st.write("")
st.divider()

footer_left, footer_right = st.columns(
    [3, 1]
)

with footer_left:

    st.caption(
        "📚 BOOKWISE"
    )

    st.caption(
        "Personalized book discovery powered by machine learning."
    )

with footer_right:

    st.caption(
        "Hybrid Recommendation System"
    )
