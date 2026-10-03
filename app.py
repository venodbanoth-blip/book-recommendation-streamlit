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

    .stApp {
        background: #f7f4ef;
        color: #191817;
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 1rem;
        padding-bottom: 4rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    h1 {
        color: #191817 !important;
        font-size: 3rem !important;
        line-height: 1.05 !important;
        font-weight: 800 !important;
        letter-spacing: -1.8px !important;
    }

    h2 {
        color: #191817 !important;
        font-size: 2rem !important;
        font-weight: 750 !important;
        letter-spacing: -1px !important;
    }

    h3 {
        color: #191817 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }

    p {
        color: #5f5a55;
        line-height: 1.6;
    }

    .stCaption {
        color: #91887f !important;
    }

    .stTextInput input {
        background: #ffffff !important;
        color: #191817 !important;
        border: 1px solid #ded8d1 !important;
        border-radius: 13px !important;
        min-height: 50px !important;
        font-size: 16px !important;
    }

    .stTextInput input:focus {
        border-color: #b66b4d !important;
        box-shadow: 0 0 0 1px #b66b4d !important;
    }

    .stButton > button {
        background: #191817 !important;
        color: white !important;
        border: 1px solid #191817 !important;
        border-radius: 13px !important;
        min-height: 50px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #b66b4d !important;
        border-color: #b66b4d !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid #ebe5de !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 28px rgba(48, 38, 30, 0.055);
    }

    [data-testid="stImage"] img {
        border-radius: 12px !important;
    }

    [data-testid="stMetric"] {
        background: transparent;
        border: none;
    }

    [data-testid="stMetricLabel"] {
        color: #91887f !important;
    }

    [data-testid="stMetricValue"] {
        color: #191817 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    hr {
        border-color: #e5ded6 !important;
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
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "discover"

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "recommended_user" not in st.session_state:
    st.session_state.recommended_user = None


# ============================================================
# TOP NAVIGATION
# ============================================================

nav1, nav2, nav3, nav4 = st.columns(
    [5, 1, 1, 1]
)

with nav1:

    st.markdown("### 📚 BookWise")

with nav2:

    if st.button(
        "Discover",
        use_container_width=True
    ):

        st.session_state.page = "discover"
        st.rerun()

with nav3:

    if st.button(
        "Recommendations",
        use_container_width=True
    ):

        st.session_state.page = "recommendations"
        st.rerun()

with nav4:

    if st.button(
        "About",
        use_container_width=True
    ):

        st.session_state.page = "about"
        st.rerun()


# ============================================================
# DISCOVER PAGE
# ============================================================

if st.session_state.page == "discover":

    # ========================================================
    # HERO IMAGE
    # ========================================================

    hero_image = (
        "https://images.unsplash.com/"
        "photo-1507842217343-583bb7270b66"
        "?auto=format&fit=crop&w=1800&q=85"
    )

    try:

        st.image(
            hero_image,
            use_container_width=True
        )

    except Exception:

        st.info(
            "Book discovery starts here."
        )


    # ========================================================
    # HERO TEXT
    # ========================================================

    st.caption(
        "PERSONALIZED BOOK DISCOVERY"
    )

    st.title(
        "Find your next unforgettable story."
    )

    st.write(
        "BookWise learns from reading patterns, "
        "similar readers, book content and popularity "
        "to create a personal bookshelf just for you."
    )

    st.write("")


    # ========================================================
    # CALL TO ACTION
    # ========================================================

    if st.button(
        "Start Discovering →",
        use_container_width=False
    ):

        st.session_state.page = "recommendations"
        st.rerun()


    st.write("")
    st.divider()


    # ========================================================
    # DISCOVERY SECTION
    # ========================================================

    st.caption(
        "HOW BOOKWISE WORKS"
    )

    st.header(
        "Books chosen around you."
    )

    st.write(
        "BookWise combines multiple recommendation signals "
        "to create a personalized Top 10."
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
                "TF-IDF and cosine similarity "
                "find books with similar content."
            )

    with c2:

        with st.container(border=True):

            st.subheader(
                "👥 Connect"
            )

            st.write(
                "KNN collaborative filtering "
                "uses patterns from similar readers."
            )

    with c3:

        with st.container(border=True):

            st.subheader(
                "✨ Personalize"
            )

            st.write(
                "Multiple recommendation signals "
                "are combined by the Hybrid model."
            )


    st.write("")
    st.divider()


    # ========================================================
    # HYBRID MODEL
    # ========================================================

    st.caption(
        "THE RECOMMENDATION ENGINE"
    )

    st.header(
        "One recommendation. Four perspectives."
    )

    st.write(
        "The deployed Hybrid model combines "
        "four trained recommendation components."
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


    # ========================================================
    # FINAL CTA
    # ========================================================

    st.header(
        "Ready to find your next book?"
    )

    st.write(
        "Enter your User ID and build your personalized "
        "bookshelf."
    )

    if st.button(
        "Build My Bookshelf →",
        use_container_width=False
    ):

        st.session_state.page = "recommendations"
        st.rerun()


# ============================================================
# RECOMMENDATIONS PAGE
# ============================================================

elif st.session_state.page == "recommendations":

    st.caption(
        "YOUR PERSONAL LIBRARY"
    )

    st.title(
        "What should you read next?"
    )

    st.write(
        "Enter your User ID and let BookWise create "
        "your personalized bookshelf."
    )

    st.write("")


    # ========================================================
    # USER INPUT
    # ========================================================

    with st.container(border=True):

        input_col, button_col = st.columns(
            [4, 1],
            gap="medium"
        )

        with input_col:

            user_id_text = st.text_input(
                "User ID",
                value="276704",
                placeholder="Enter your User ID"
            )

        with button_col:

            st.write("")

            generate = st.button(
                "Find Books ✨",
                use_container_width=True
            )


    # ========================================================
    # GENERATE RECOMMENDATIONS
    # ========================================================

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


        st.session_state.recommendations = (
            recommendations
        )

        st.session_state.recommended_user = (
            user_id
        )


    # ========================================================
    # DISPLAY RECOMMENDATIONS
    # ========================================================

    if (
        st.session_state.recommendations
        and st.session_state.recommended_user
    ):

        recommendations = (
            st.session_state.recommendations
        )

        user_id = (
            st.session_state.recommended_user
        )


        st.write("")
        st.divider()


        result_col1, result_col2 = st.columns(
            [4, 1]
        )

        with result_col1:

            st.caption(
                "PERSONALIZED FOR YOU"
            )

            st.header(
                "Your next 10 reads."
            )

            st.caption(
                f"Curated for User {user_id} "
                "using the Hybrid model"
            )

        with result_col2:

            st.metric(
                "Books",
                len(recommendations)
            )


        st.write("")


        # ====================================================
        # BOOK GRID
        # ====================================================

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


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.caption(
            "WHY THESE BOOKS?"
        )

        st.header(
            "Powered by the Hybrid engine."
        )

        st.write(
            "The final ranking combines the same "
            "four trained recommendation signals."
        )

        st.write("")


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
# ABOUT PAGE
# ============================================================

elif st.session_state.page == "about":

    st.caption(
        "ABOUT BOOKWISE"
    )

    st.title(
        "A machine-learning approach to book discovery."
    )

    st.write(
        "BookWise combines several recommendation "
        "approaches to produce one personalized ranking."
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
