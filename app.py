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
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# ARTIFACT DIRECTORY
# ============================================================

ARTIFACT_DIR = "model_artifacts"


# ============================================================
# LOAD SAVED MODEL ARTIFACTS
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
            "The following model files are missing:\n"
            + "\n".join(missing_files)
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

    artifacts["popularity_rank"] = pd.read_pickle(
        os.path.join(
            ARTIFACT_DIR,
            "popularity_rank.pkl"
        )
    )

    artifacts["book_info"] = pd.read_pickle(
        os.path.join(
            ARTIFACT_DIR,
            "book_info.pkl"
        )
    )

    artifacts["train_history"] = pd.read_pickle(
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
        "r"
    ) as file:
        artifacts["config"] = json.load(file)

    return artifacts


# ============================================================
# LOAD ARTIFACTS
# ============================================================

try:

    artifacts = load_artifacts()

except Exception as error:

    st.error(
        "Model artifacts could not be loaded."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# EXTRACT ARTIFACTS
# ============================================================

tfidf_matrix = artifacts["tfidf_matrix"]
isbn_to_index = artifacts["isbn_to_index"]

knn_model = artifacts["knn_model"]

user_book_sparse = artifacts["user_book_sparse"]
user_to_index = artifacts["user_to_index"]

user_factors = artifacts["user_factors"]
book_factors = artifacts["book_factors"]

train_user_ids = artifacts["train_user_ids"]
train_isbn_ids = artifacts["train_isbn_ids"]

popularity_rank = artifacts["popularity_rank"]

book_info = artifacts["book_info"]
train_history = artifacts["train_history"]

config = artifacts["config"]


# ============================================================
# BOOK INFORMATION MAPPINGS
# ============================================================

book_info = (
    book_info
    .drop_duplicates("ISBN")
    .copy()
)

isbn_to_title = (
    book_info
    .set_index("ISBN")["Book-Title"]
    .to_dict()
)

isbn_to_author = (
    book_info
    .set_index("ISBN")["Book-Author"]
    .to_dict()
)

isbn_to_publisher = (
    book_info
    .set_index("ISBN")["Publisher"]
    .to_dict()
)


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

            if candidate_index >= len(
                train_isbn_ids
            ):
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
# COLLABORATIVE FILTERING RECOMMENDATION
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
            train_history["User-ID"]
            == neighbor_user
        ]

        for _, row in neighbor_books.iterrows():

            isbn = row["ISBN"]

            if isbn in history:
                continue

            scores[isbn] = (
                scores.get(isbn, 0)
                +
                similarity * row["Book-Rating"]
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

    user_vector = user_factors[
        user_index
    ]

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

        isbn = train_isbn_ids[
            book_index
        ]

        recommendations.append(isbn)

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

    popularity = (
        popularity_rank[
            "ISBN"
        ]
        .head(100)
        .tolist()
    )

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
        isbn: 1 - (index / 100)
        for index, isbn
        in enumerate(content)
    }

    collaborative_scores = {
        isbn: 1 - (index / 100)
        for index, isbn
        in enumerate(collaborative)
    }

    svd_scores = {
        isbn: 1 - (index / 100)
        for index, isbn
        in enumerate(svd)
    }

    popularity_scores = {
        isbn: 1 - (index / 100)
        for index, isbn
        in enumerate(popularity)
    }

    final_scores = {}

    for isbn in candidates:

        score = (
            0.30
            * content_scores.get(isbn, 0)
            +
            0.30
            * collaborative_scores.get(isbn, 0)
            +
            0.20
            * svd_scores.get(isbn, 0)
            +
            0.20
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
        for isbn, score
        in ranked[:top_n]
    ]


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.title("📚 Book Recommendation System")

st.write(
    "Get personalized book recommendations "
    "using a Hybrid Machine Learning "
    "Recommendation System."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.write(
        "**Selected Model:** Hybrid"
    )

    st.write(
        "**Recommendation Count:** 10"
    )

    st.write(
        "**Model Components:**"
    )

    st.write(
        "Content-Based: 30%"
    )

    st.write(
        "Collaborative Filtering: 30%"
    )

    st.write(
        "SVD: 20%"
    )

    st.write(
        "Popularity: 20%"
    )


# ============================================================
# USER INPUT
# ============================================================

user_id_input = st.text_input(
    "Enter User ID",
    placeholder="Example: 276704"
)


# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

if st.button(
    "Get Recommendations",
    type="primary"
):

    if not user_id_input.strip():

        st.warning(
            "Please enter a User ID."
        )

        st.stop()

    try:

        user_id = int(
            user_id_input.strip()
        )

    except ValueError:

        st.error(
            "User ID must contain numbers only."
        )

        st.stop()

    if user_id not in user_to_index:

        st.error(
            "User ID not found in the training data."
        )

        st.info(
            "Please enter a User ID that exists "
            "in the trained recommendation system."
        )

        st.stop()

    with st.spinner(
        "Generating personalized recommendations..."
    ):

        try:

            recommendations = recommend_hybrid(
                user_id,
                top_n=10
            )

        except Exception as error:

            st.error(
                "An error occurred while generating "
                "recommendations."
            )

            st.code(str(error))

            st.stop()

    if not recommendations:

        st.warning(
            "No recommendations are available "
            "for this user."
        )

    else:

        st.success(
            "Recommendations generated successfully!"
        )

        st.subheader(
            "📖 Recommended Books"
        )

        recommendation_data = []

        for rank, isbn in enumerate(
            recommendations,
            start=1
        ):

            recommendation_data.append({
                "Rank": rank,
                "Book Title": isbn_to_title.get(
                    isbn,
                    "Title Not Available"
                ),
                "Author": isbn_to_author.get(
                    isbn,
                    "Author Not Available"
                ),
                "Publisher": isbn_to_publisher.get(
                    isbn,
                    "Publisher Not Available"
                ),
                "ISBN": isbn
            })

        recommendation_df = pd.DataFrame(
            recommendation_data
        )

        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.caption(
    "Book Recommendation System | "
    "Hybrid Machine Learning Model"
)