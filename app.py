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
    page_title="BookWise | Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODERN THEME
# CSS ONLY — NO VISIBLE HTML UI
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background: #f7f1e8;
        color: #171411;
    }

    /* Main content width */
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #eadcc9;
        border-right: 1px solid #d3c0aa;
    }

    [data-testid="stSidebar"] * {
        color: #171411 !important;
    }

    /* Headings */
    h1, h2, h3 {
        color: #171411 !important;
        letter-spacing: -0.5px;
    }

    /* Normal text */
    p, label, span {
        color: #171411;
    }

    /* Buttons */
    .stButton > button {
        background: #4b2e20;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #2f1c13;
        color: white;
        border: none;
    }

    /* Input fields */
    .stTextInput input {
        background: #fffaf3;
        color: #171411;
        border: 1px solid #cdb9a2;
        border-radius: 12px;
    }

    .stTextInput input:focus {
        border-color: #4b2e20;
        box-shadow: 0 0 0 1px #4b2e20;
    }

    /* Selectbox */
    [data-baseweb="select"] > div {
        background: #fffaf3;
        border-radius: 12px;
        border-color: #cdb9a2;
    }

    /* Cards */
    .book-card {
        background: #fffaf3;
        border: 1px solid #dfcfbd;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 18px;
        min-height: 430px;
        box-shadow: 0 5px 18px rgba(75, 46, 32, 0.08);
    }

    .book-card:hover {
        box-shadow: 0 8px 24px rgba(75, 46, 32, 0.14);
    }

    /* Divider */
    hr {
        border-color: #d8c5af;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #fffaf3;
        border: 1px solid #dfcfbd;
        border-radius: 14px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #705747 !important;
    }

    [data-testid="stMetricValue"] {
        color: #171411 !important;
    }

    /* Info boxes */
    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* Images */
    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    /* Footer spacing */
    .footer-space {
        height: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"


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
# DATA CLEANING FOR DISPLAY
# ============================================================

book_info = book_info.copy()

book_info["ISBN"] = (
    book_info["ISBN"]
    .fillna("")
    .astype(str)
    .str.strip()
)

for column in [
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
# BOOK INFORMATION
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

    possible_columns = [
        "Image-URL-M",
        "Image-URL-L",
        "Image-URL-S"
    ]

    for column in possible_columns:

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

    st.markdown("## 📚 BookWise")

    st.caption(
        "Personalized book discovery"
    )

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Choose a section",
        [
            "Home",
            "Get Recommendations",
            "About Model"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("Hybrid Model")

    st.write(
        "Your recommendations combine four "
        "different recommendation signals."
    )

    st.caption(
        "Content 30%  •  Collaborative 30%  "
        "•  SVD 20%  •  Popularity 20%"
    )

    st.divider()

    st.caption(
        "BookWise Recommendation System"
    )

    st.caption(
        "Machine Learning • Recommendation • Streamlit"
    )


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.title("📚 BookWise")

    st.subheader(
        "Discover your next great book."
    )

    st.write(
        "A personalized book recommendation system "
        "that combines content similarity, collaborative "
        "filtering, SVD, and popularity signals."
    )

    st.divider()

    # --------------------------------------------------------
    # HERO METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Recommendation Model",
            "Hybrid"
        )

    with col2:
        st.metric(
            "Recommendation Size",
            "Top 10"
        )

    with col3:
        st.metric(
            "Content Weight",
            "30%"
        )

    with col4:
        st.metric(
            "Collaborative Weight",
            "30%"
        )

    st.divider()

    st.header("How BookWise works")

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.subheader(
                "📖 Content-Based"
            )

            st.write(
                "Finds books that are similar to "
                "books the user has interacted with "
                "using TF-IDF and cosine similarity."
            )

            st.caption(
                "Hybrid contribution: 30%"
            )

    with c2:

        with st.container(border=True):

            st.subheader(
                "👥 Collaborative"
            )

            st.write(
                "Uses similar users and their reading "
                "history to identify books that may "
                "interest the current user."
            )

            st.caption(
                "Hybrid contribution: 30%"
            )

    c3, c4 = st.columns(2)

    with c3:

        with st.container(border=True):

            st.subheader(
                "🧠 SVD"
            )

            st.write(
                "Uses latent factors learned from "
                "the user-book interaction matrix "
                "to estimate book preferences."
            )

            st.caption(
                "Hybrid contribution: 20%"
            )

    with c4:

        with st.container(border=True):

            st.subheader(
                "🔥 Popularity"
            )

            st.write(
                "Adds highly rated and popular books "
                "as another signal in the final "
                "recommendation."
            )

            st.caption(
                "Hybrid contribution: 20%"
            )

    st.divider()

    st.header("Start discovering")

    st.write(
        "Enter a valid User ID in the "
        "**Get Recommendations** section."
    )

    if 276704 in user_to_index:

        st.info(
            "Try User ID 276704 — this user is "
            "available in the trained recommendation model."
        )


# ============================================================
# RECOMMENDATION PAGE
# ============================================================

elif page == "Get Recommendations":

    st.title("📚 Personalized Recommendations")

    st.write(
        "Enter your User ID to generate personalized "
        "book recommendations using the Hybrid model."
    )

    st.divider()

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    input_col, button_col = st.columns(
        [4, 1]
    )

    with input_col:

        user_id_text = st.text_input(
            "User ID",
            value="276704",
            placeholder="Example: 276704",
            help="Enter a User ID available in the trained model."
        )

    with button_col:

        st.write("")

        get_recommendations = st.button(
            "✨ Recommend",
            use_container_width=True
        )

    if get_recommendations:

        # ----------------------------------------------------
        # VALIDATE USER ID
        # ----------------------------------------------------

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
                "User ID not found in the trained recommendation model."
            )

            st.info(
                "Try User ID 276704."
            )

            st.stop()

        # ----------------------------------------------------
        # GENERATE RECOMMENDATIONS
        # ----------------------------------------------------

        with st.spinner(
            "Finding books personalized for you..."
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
        # SUCCESS HEADER
        # ----------------------------------------------------

        st.success(
            f"Found {len(recommendations)} "
            f"personalized recommendations for User {user_id}."
        )

        st.subheader(
            "✨ Recommended for You"
        )

        st.caption(
            "Powered by the Hybrid recommendation model"
        )

        st.divider()

        # ----------------------------------------------------
        # HYBRID MODEL INFORMATION
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown(
                "### 🧠 Hybrid Recommendation Engine"
            )

            st.write(
                "The final ranking combines four recommendation "
                "signals without changing the trained model."
            )

            w1, w2, w3, w4 = st.columns(4)

            with w1:
                st.metric(
                    "Content",
                    "30%"
                )

            with w2:
                st.metric(
                    "Collaborative",
                    "30%"
                )

            with w3:
                st.metric(
                    "SVD",
                    "20%"
                )

            with w4:
                st.metric(
                    "Popularity",
                    "20%"
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
                len(row_books)
            )

            for position, isbn in enumerate(
                row_books,
                start=row_start + 1
            ):

                with columns[
                    position - row_start - 1
                ]:

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

                                    st.info(
                                        "📖 Cover unavailable"
                                    )

                            else:

                                st.info(
                                    "📖 Cover unavailable"
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

                            st.caption(
                                f"Recommendation #{position}"
                            )

                            st.subheader(
                                title[:80]
                            )

                            st.write(
                                f"**Author:** {author[:55]}"
                            )

                            if publisher:

                                st.caption(
                                    f"Publisher: {publisher[:55]}"
                                )

                            st.caption(
                                f"ISBN: {isbn}"
                            )

                        else:

                            st.subheader(
                                f"Book #{position}"
                            )

                            st.caption(
                                f"ISBN: {isbn}"
                            )

        st.divider()

        st.caption(
            "Recommendations are generated from the trained "
            "Hybrid model using your existing model artifacts."
        )


# ============================================================
# ABOUT MODEL PAGE
# ============================================================

elif page == "About Model":

    st.title("🧠 About the Recommendation System")

    st.write(
        "BookWise uses multiple machine-learning recommendation "
        "approaches and combines their ranked results into one "
        "Hybrid recommendation."
    )

    st.divider()

    st.header("Hybrid Model Configuration")

    config_col1, config_col2 = st.columns(2)

    with config_col1:

        st.metric(
            "Content-Based",
            "30%"
        )

        st.write(
            "TF-IDF + cosine similarity"
        )

        st.metric(
            "Collaborative Filtering",
            "30%"
        )

        st.write(
            "KNN-based user similarity"
        )

    with config_col2:

        st.metric(
            "SVD",
            "20%"
        )

        st.write(
            "Latent-factor recommendation"
        )

        st.metric(
            "Popularity",
            "20%"
        )

        st.write(
            "Popularity-based ranking"
        )

    st.divider()

    st.header("Recommendation Flow")

    st.write(
        "1. Identify the user's previous book interactions."
    )

    st.write(
        "2. Generate candidate books using Content-Based filtering."
    )

    st.write(
        "3. Generate candidate books using Collaborative filtering."
    )

    st.write(
        "4. Generate candidate books using SVD."
    )

    st.write(
        "5. Add popular books as additional candidates."
    )

    st.write(
        "6. Combine the four ranking signals using the "
        "configured Hybrid weights."
    )

    st.write(
        "7. Remove books already present in the user's history."
    )

    st.write(
        "8. Return the final Top 10 recommendations."
    )

    st.divider()

    st.header("Model Architecture")

    architecture = pd.DataFrame(
        {
            "Component": [
                "Content-Based",
                "Collaborative",
                "SVD",
                "Popularity"
            ],
            "Method": [
                "TF-IDF + Cosine Similarity",
                "KNN User Similarity",
                "Truncated SVD",
                "Popularity Ranking"
            ],
            "Hybrid Weight": [
                "30%",
                "30%",
                "20%",
                "20%"
            ]
        }
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.info(
        "The recommendation model and its weights are loaded "
        "from the existing model_artifacts folder."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📚 BookWise • Machine Learning Book Recommendation System"
)

st.caption(
    "Hybrid Recommendation • Personalized Discovery • Streamlit"
)
