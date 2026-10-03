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
    page_title="BookWise - AI Book Recommendation System",
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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f9fc;
    }

    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .main-title span {
        color: #ff4b4b;
    }

    .subtitle {
        font-size: 19px;
        color: #666666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .section-text {
        color: #666666;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .badge {
        display: inline-block;
        padding: 8px 14px;
        margin: 4px;
        border-radius: 20px;
        background-color: #eef2ff;
        color: #333333;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-box {
        padding: 18px;
        border-radius: 12px;
        background-color: white;
        border: 1px solid #e6e6e6;
        text-align: center;
        margin-bottom: 15px;
    }

    .metric-number {
        font-size: 28px;
        font-weight: 750;
    }

    .metric-label {
        color: #777777;
        font-size: 14px;
    }

    .book-rank {
        font-size: 13px;
        font-weight: 700;
        color: #ff4b4b;
        margin-bottom: 8px;
    }

    .book-title {
        font-size: 20px;
        font-weight: 750;
        margin-top: 8px;
        margin-bottom: 5px;
    }

    .book-author {
        font-size: 15px;
        color: #555555;
        margin-bottom: 5px;
    }

    .book-meta {
        font-size: 13px;
        color: #777777;
    }

    .info-box {
        padding: 20px;
        border-radius: 12px;
        background-color: white;
        border: 1px solid #e6e6e6;
        min-height: 180px;
    }

    .info-title {
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .info-text {
        color: #666666;
        font-size: 14px;
        line-height: 1.6;
    }

    .tech-box {
        padding: 20px;
        border-radius: 12px;
        background-color: white;
        border: 1px solid #e6e6e6;
        text-align: center;
    }

    .footer {
        text-align: center;
        color: #777777;
        padding: 30px 0 10px 0;
        font-size: 14px;
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

    artifacts_loaded = True

except Exception as e:

    artifacts_loaded = False

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
        isbn for isbn in history
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

    st.markdown("## 📚 BookWise")

    st.caption(
        "AI Book Recommendation System"
    )

    st.divider()

    st.markdown("### Recommendation Models")

    st.write("📈 Popularity-Based")
    st.write("📖 Content-Based TF-IDF")
    st.write("👥 Collaborative Filtering")
    st.write("🧮 SVD")
    st.write("🤖 Hybrid Recommendation")

    st.divider()

    st.markdown("### Hybrid Weights")

    st.write("Content — **30%**")
    st.write("Collaborative — **30%**")
    st.write("SVD — **20%**")
    st.write("Popularity — **20%**")

    st.divider()

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
# HERO SECTION
# ============================================================

st.markdown(
    '<div style="font-size:55px;">📚</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">
        Book<span>Wise</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI Book Recommendation System
    </div>
    """,
    unsafe_allow_html=True
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


# ============================================================
# SEARCH SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Find Your Next Read</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-text">
        Enter your User ID and BookWise will generate
        personalized recommendations.
    </div>
    """,
    unsafe_allow_html=True
)

user_id_input = st.text_input(
    "Enter User ID",
    placeholder="Example: 276704"
)

recommend_button = st.button(
    "✨ Get Recommendations",
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
                "Generating personalized recommendations..."
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
                    f"Found {len(recommendations)} "
                    "personalized recommendations."
                )

                st.markdown(
                    "### 📚 Recommended Books"
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
                            )

                            author = str(
                                book.get(
                                    "Book-Author",
                                    "Unknown Author"
                                )
                            )

                            publisher = str(
                                book.get(
                                    "Publisher",
                                    "Unknown Publisher"
                                )
                            )

                            image_url = str(
                                book.get(
                                    "Image-URL-M",
                                    ""
                                )
                            ).strip()

                            st.markdown(
                                f"**#{rank} • HYBRID MATCH**"
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
                                f"### {title}"
                            )

                            st.write(
                                f"**Author:** {author}"
                            )

                            st.write(
                                f"**Publisher:** {publisher}"
                            )

                            st.caption(
                                f"ISBN: {isbn}"
                            )

                            st.divider()


# ============================================================
# HOW BOOKWISE WORKS
# ============================================================

st.markdown(
    '<div class="section-title">🧠 How BookWise Works</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="info-box">

        <div style="font-size:32px;">📖</div>

        <div class="info-title">
        Content-Based
        </div>

        <div class="info-text">
        Uses TF-IDF and cosine similarity
        to find books with similar content
        characteristics such as title, author
        and publisher.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="info-box">

        <div style="font-size:32px;">👥</div>

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
        <div class="info-box">

        <div style="font-size:32px;">🤖</div>

        <div class="info-title">
        Hybrid Intelligence
        </div>

        <div class="info-text">
        Combines Content, Collaborative,
        SVD and Popularity signals into
        a single recommendation ranking.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Technology Stack</div>',
    unsafe_allow_html=True
)

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:

    st.markdown(
        """
        <div class="tech-box">
        <div style="font-size:32px;">🐍</div>
        <b>Python</b>
        </div>
        """,
        unsafe_allow_html=True
    )

with tech2:

    st.markdown(
        """
        <div class="tech-box">
        <div style="font-size:32px;">🧠</div>
        <b>Scikit-learn</b>
        </div>
        """,
        unsafe_allow_html=True
    )

with tech3:

    st.markdown(
        """
        <div class="tech-box">
        <div style="font-size:32px;">📊</div>
        <b>Machine Learning</b>
        </div>
        """,
        unsafe_allow_html=True
    )

with tech4:

    st.markdown(
        """
        <div class="tech-box">
        <div style="font-size:32px;">🚀</div>
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
    <div class="footer">

    📚 <b>BookWise</b> —
    Machine Learning Book Recommendation System

    <br><br>

    Built with Python • Scikit-learn • Streamlit • Machine Learning

    <br><br>

    Hybrid Recommendation • Content Similarity •
    Collaborative Filtering • SVD

    </div>
    """,
    unsafe_allow_html=True
)
