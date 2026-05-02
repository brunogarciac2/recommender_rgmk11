from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Config
# ----------------------------
OUTPUT_DIR = Path("outputs")
RATINGS_PATH = OUTPUT_DIR / "ratings_sample_150k.csv"
MOVIE_PROFILES_PATH = OUTPUT_DIR / "movie_profiles.csv"

POSITIVE_RATING_THRESHOLD = 4.0
MIN_POSITIVE_RATINGS_PER_USER = 5
DEFAULT_TOP_K = 10
EXPLANATION_NEIGHBOURS = 2


class TFIDFRecommender:
    def __init__(
        self,
        ratings_path: Path = RATINGS_PATH,
        movie_profiles_path: Path = MOVIE_PROFILES_PATH,
        positive_threshold: float = POSITIVE_RATING_THRESHOLD,
        min_positive_ratings_per_user: int = MIN_POSITIVE_RATINGS_PER_USER,
        max_features: int = 5000,
    ) -> None:
        self.ratings_path = ratings_path
        self.movie_profiles_path = movie_profiles_path
        self.positive_threshold = positive_threshold
        self.min_positive_ratings_per_user = min_positive_ratings_per_user
        self.max_features = max_features

        self.ratings: Optional[pd.DataFrame] = None
        self.movie_profiles: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.movie_tfidf_matrix: Optional[csr_matrix] = None

        self.movieid_to_idx: dict[int, int] = {}
        self.idx_to_movieid: dict[int, int] = {}

    # ----------------------------
    # Loading + fitting
    # ----------------------------
    def load_data(self) -> None:
        self.ratings = pd.read_csv(self.ratings_path)
        self.movie_profiles = pd.read_csv(self.movie_profiles_path)

        self.movie_profiles["profile_text"] = self.movie_profiles["profile_text"].fillna("")
        self.movie_profiles["title"] = self.movie_profiles["title"].fillna("Unknown Title")
        self.movie_profiles["genres"] = self.movie_profiles["genres"].fillna("")

        self.movieid_to_idx = {
            movie_id: idx for idx, movie_id in enumerate(self.movie_profiles["movieId"].tolist())
        }
        self.idx_to_movieid = {
            idx: movie_id for movie_id, idx in self.movieid_to_idx.items()
        }

    def fit(self) -> None:
        if self.movie_profiles is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english"
        )
        self.movie_tfidf_matrix = self.vectorizer.fit_transform(
            self.movie_profiles["profile_text"]
        )

    # ----------------------------
    # User helpers
    # ----------------------------
    def get_positive_user_ratings(self, user_id: int) -> pd.DataFrame:
        if self.ratings is None:
            raise ValueError("Ratings not loaded.")

        user_ratings = self.ratings[self.ratings["userId"] == user_id].copy()
        positives = user_ratings[user_ratings["rating"] >= self.positive_threshold].copy()
        return positives

    def user_is_eligible(self, user_id: int) -> bool:
        positives = self.get_positive_user_ratings(user_id)
        return len(positives) >= self.min_positive_ratings_per_user

    def get_seen_movie_ids(self, user_id: int) -> set[int]:
        if self.ratings is None:
            raise ValueError("Ratings not loaded.")

        seen = self.ratings[self.ratings["userId"] == user_id]["movieId"].tolist()
        return set(seen)

    def build_user_profile(self, user_id: int) -> csr_matrix:
        if self.movie_tfidf_matrix is None:
            raise ValueError("Model not fit. Call fit() first.")

        positives = self.get_positive_user_ratings(user_id)

        if len(positives) < self.min_positive_ratings_per_user:
            raise ValueError(
                f"User {user_id} has only {len(positives)} positive ratings. "
                f"Need at least {self.min_positive_ratings_per_user}."
            )

        liked_movie_ids = [
            movie_id for movie_id in positives["movieId"].tolist()
            if movie_id in self.movieid_to_idx
        ]

        if not liked_movie_ids:
            raise ValueError(f"User {user_id} has no liked movies with available profiles.")

        liked_indices = [self.movieid_to_idx[movie_id] for movie_id in liked_movie_ids]
        liked_matrix = self.movie_tfidf_matrix[liked_indices]

        user_profile = liked_matrix.mean(axis=0)
        return csr_matrix(user_profile)

    # ----------------------------
    # Recommendation logic
    # ----------------------------
    def recommend(self, user_id: int, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
        if self.movie_profiles is None or self.movie_tfidf_matrix is None:
            raise ValueError("Data/model not ready. Call load_data() and fit() first.")

        user_profile = self.build_user_profile(user_id)
        seen_movie_ids = self.get_seen_movie_ids(user_id)

        similarities = cosine_similarity(user_profile, self.movie_tfidf_matrix).flatten()

        candidates = self.movie_profiles[["movieId", "title", "genres"]].copy()
        candidates["score"] = similarities
        candidates = candidates[~candidates["movieId"].isin(seen_movie_ids)].copy()
        candidates = candidates.sort_values("score", ascending=False).head(top_k).copy()

        candidates["explanation"] = candidates["movieId"].apply(
            lambda movie_id: self.generate_explanation(user_id, movie_id)
        )

        return candidates.reset_index(drop=True)

    # ----------------------------
    # Explanation helpers
    # ----------------------------
    def _format_title_list(self, titles: List[str]) -> str:
        titles = [t for t in titles if t]
        if not titles:
            return ""
        if len(titles) == 1:
            return titles[0]
        if len(titles) == 2:
            return f"{titles[0]} and {titles[1]}"
        return ", ".join(titles[:-1]) + f", and {titles[-1]}"

    def _clean_genres_for_explanation(self, genres: str, max_genres: int = 3) -> str:
        if not genres or pd.isna(genres) or genres == "(no genres listed)":
            return ""
        genre_list = [g.strip() for g in str(genres).split("|") if g.strip()]
        return ", ".join(genre_list[:max_genres])

    # ----------------------------
    # Explanations
    # ----------------------------
    def generate_explanation(self, user_id: int, recommended_movie_id: int) -> str:
        if self.movie_profiles is None or self.movie_tfidf_matrix is None:
            return "No explanation available."

        positives = self.get_positive_user_ratings(user_id)
        liked_movie_ids = [
            movie_id for movie_id in positives["movieId"].tolist()
            if movie_id in self.movieid_to_idx
        ]

        if recommended_movie_id not in self.movieid_to_idx or not liked_movie_ids:
            return "Recommended based on similarity to films you rated positively."

        rec_idx = self.movieid_to_idx[recommended_movie_id]
        rec_vector = self.movie_tfidf_matrix[rec_idx]

        liked_indices = [self.movieid_to_idx[movie_id] for movie_id in liked_movie_ids]
        liked_vectors = self.movie_tfidf_matrix[liked_indices]

        sims = cosine_similarity(rec_vector, liked_vectors).flatten()
        top_liked_positions = np.argsort(sims)[::-1][:EXPLANATION_NEIGHBOURS]
        top_liked_movie_ids = [liked_movie_ids[pos] for pos in top_liked_positions]

        liked_titles = []
        for movie_id in top_liked_movie_ids:
            row = self.movie_profiles[self.movie_profiles["movieId"] == movie_id]
            if not row.empty:
                liked_titles.append(row.iloc[0]["title"])

        rec_row = self.movie_profiles[self.movie_profiles["movieId"] == recommended_movie_id]
        if rec_row.empty:
            return "Recommended based on similarity to films you rated positively."

        rec_genres = self._clean_genres_for_explanation(rec_row.iloc[0]["genres"])

        if len(liked_titles) >= 2 and rec_genres:
            return (
                f"Recommended because it is close in style to {liked_titles[0]} and "
                f"{liked_titles[1]}, especially in {rec_genres.lower()}."
            )

        if len(liked_titles) == 1 and rec_genres:
            return (
                f"Recommended because it is similar to {liked_titles[0]} and shares "
                f"{rec_genres.lower()} characteristics."
            )

        if len(liked_titles) >= 2:
            liked_text = self._format_title_list(liked_titles[:2])
            return f"Recommended because it is similar to films you liked, especially {liked_text}."

        if len(liked_titles) == 1:
            return f"Recommended because it is similar to {liked_titles[0]}."

        return "Recommended based on similarity to films you rated positively."

    # ----------------------------
    # Utility methods
    # ----------------------------
    def get_eligible_user_ids(self) -> List[int]:
        if self.ratings is None:
            raise ValueError("Ratings not loaded.")

        positive_counts = (
            self.ratings[self.ratings["rating"] >= self.positive_threshold]
            .groupby("userId")
            .size()
        )

        eligible_users = positive_counts[
            positive_counts >= self.min_positive_ratings_per_user
        ].index.tolist()

        return eligible_users


def main() -> None:
    recommender = TFIDFRecommender()
    recommender.load_data()
    recommender.fit()

    eligible_users = recommender.get_eligible_user_ids()
    print(f"Eligible users with at least {MIN_POSITIVE_RATINGS_PER_USER} positive ratings: {len(eligible_users)}")

    if not eligible_users:
        print("No eligible users found.")
        return

    demo_user = eligible_users[0]
    print(f"\nDemo recommendations for user {demo_user}:\n")

    recs = recommender.recommend(demo_user, top_k=10)

    seen_movie_ids = recommender.get_seen_movie_ids(demo_user)
    already_seen_check = recs["movieId"].isin(seen_movie_ids).any()
    print(f"Any recommended movies already seen by user {demo_user}? {already_seen_check}\n")

    print(recs[["movieId", "title", "score", "explanation"]].to_string(index=False))


if __name__ == "__main__":
    main()