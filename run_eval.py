import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from rs1_tfidf import TFIDFRecommender
from rs2_sbert import SBERTRecommender


TOP_K = 10
EVAL_USER_LIMIT = 300
MIN_POSITIVES_FOR_EVAL = 6
POSITIVE_THRESHOLD = 4.0
RANDOM_STATE = 42

N_HOLDOUT_POSITIVES = 3
N_NEGATIVE_SAMPLES = 100


def ndcg_at_k(recommended_ids: List[int], relevant_ids: List[int], k: int = 10) -> float:
    recommended_ids = recommended_ids[:k]
    dcg = 0.0
    for i, movie_id in enumerate(recommended_ids):
        if movie_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended_ids: List[int], relevant_ids: List[int], k: int = 10) -> float:
    recommended_ids = recommended_ids[:k]
    return 1.0 if any(movie_id in relevant_ids for movie_id in recommended_ids) else 0.0


def novelty_at_k(recommended_ids: List[int], item_popularity: dict[int, float], k: int = 10) -> float:
    vals = []
    for movie_id in recommended_ids[:k]:
        pop = item_popularity.get(movie_id, 1e-12)
        pop = max(pop, 1e-12)
        vals.append(-math.log2(pop))
    return float(np.mean(vals)) if vals else 0.0


def build_item_popularity(ratings: pd.DataFrame) -> dict[int, float]:
    n_users = ratings["userId"].nunique()
    item_user_counts = ratings.groupby("movieId")["userId"].nunique()
    popularity = (item_user_counts / n_users).to_dict()
    return popularity


def get_eval_users(ratings: pd.DataFrame) -> List[int]:
    positives = ratings[ratings["rating"] >= POSITIVE_THRESHOLD]
    positive_counts = positives.groupby("userId").size()
    return positive_counts[positive_counts >= MIN_POSITIVES_FOR_EVAL].index.tolist()


def hold_out_positive_items(
    ratings: pd.DataFrame,
    user_id: int,
    n_holdout: int = N_HOLDOUT_POSITIVES,
) -> Tuple[List[int], pd.DataFrame]:
    user_rows = ratings[ratings["userId"] == user_id].copy()
    positive_rows = user_rows[user_rows["rating"] >= POSITIVE_THRESHOLD].copy()

    n_holdout = min(n_holdout, len(positive_rows) - 1)
    if n_holdout <= 0:
        raise ValueError(f"User {user_id} does not have enough positive items to hold out.")

    held_out_rows = positive_rows.iloc[-n_holdout:]
    held_out_movie_ids = held_out_rows["movieId"].astype(int).tolist()

    modified_ratings = ratings.drop(index=held_out_rows.index).copy()
    return held_out_movie_ids, modified_ratings


def score_candidate_movies(recommender, user_id: int, candidate_movie_ids: List[int]) -> pd.DataFrame:
    """
    Score only a restricted candidate set.
    Works for both RS1 and RS2.
    """
    user_profile = recommender.build_user_profile(user_id)
    candidate_movie_ids = [mid for mid in candidate_movie_ids if mid in recommender.movieid_to_idx]

    candidate_indices = [recommender.movieid_to_idx[mid] for mid in candidate_movie_ids]

    if isinstance(recommender, TFIDFRecommender):
        candidate_matrix = recommender.movie_tfidf_matrix[candidate_indices]
        sims = cosine_similarity(user_profile, candidate_matrix).flatten()
    else:
        candidate_matrix = recommender.movie_embeddings[candidate_indices]
        sims = cosine_similarity(user_profile, candidate_matrix).flatten()

    rows = []
    for movie_id, score in zip(candidate_movie_ids, sims):
        row = recommender.movie_profiles[recommender.movie_profiles["movieId"] == movie_id]
        if not row.empty:
            rows.append({
                "movieId": movie_id,
                "title": row.iloc[0]["title"],
                "genres": row.iloc[0]["genres"],
                "score": float(score),
            })

    result = pd.DataFrame(rows).sort_values("score", ascending=False).head(TOP_K).reset_index(drop=True)
    return result


def sample_negative_items(
    ratings: pd.DataFrame,
    user_id: int,
    held_out_movie_ids: List[int],
    all_movie_ids: List[int],
    n_negatives: int = N_NEGATIVE_SAMPLES,
) -> List[int]:
    seen_movie_ids = set(ratings[ratings["userId"] == user_id]["movieId"].tolist())
    exclude = seen_movie_ids.union(set(held_out_movie_ids))

    negatives = [mid for mid in all_movie_ids if mid not in exclude]

    if len(negatives) <= n_negatives:
        return negatives

    return random.sample(negatives, n_negatives)


def evaluate_recommender(recommender, rs_name: str, user_ids: List[int], item_popularity: dict[int, float]) -> dict:
    ndcgs = []
    hit_rates = []
    novelties = []
    evaluated_users = 0

    original_ratings = recommender.ratings.copy()
    all_movie_ids = recommender.movie_profiles["movieId"].astype(int).tolist()

    for user_id in user_ids:
        try:
            held_out_movie_ids, modified_ratings = hold_out_positive_items(original_ratings, user_id)

            # temporarily replace ratings so the profile is built without held-out positives
            recommender.ratings = modified_ratings

            if not recommender.user_is_eligible(user_id):
                continue

            negative_movie_ids = sample_negative_items(
                ratings=modified_ratings,
                user_id=user_id,
                held_out_movie_ids=held_out_movie_ids,
                all_movie_ids=all_movie_ids,
                n_negatives=N_NEGATIVE_SAMPLES,
            )

            candidate_movie_ids = held_out_movie_ids + negative_movie_ids
            recs = score_candidate_movies(recommender, user_id, candidate_movie_ids)
            recommended_ids = recs["movieId"].tolist()

            ndcgs.append(ndcg_at_k(recommended_ids, held_out_movie_ids, k=TOP_K))
            hit_rates.append(hit_rate_at_k(recommended_ids, held_out_movie_ids, k=TOP_K))
            novelties.append(novelty_at_k(recommended_ids, item_popularity, k=TOP_K))
            evaluated_users += 1

        except Exception:
            continue

    recommender.ratings = original_ratings

    return {
        "RS": rs_name,
        "Users evaluated": evaluated_users,
        "HitRate@10": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "NDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "Novelty@10": float(np.mean(novelties)) if novelties else 0.0,
    }


def main() -> None:
    random.seed(RANDOM_STATE)

    print("Loading RS1...")
    rs1 = TFIDFRecommender()
    rs1.load_data()
    rs1.fit()

    print("Loading RS2...")
    rs2 = SBERTRecommender()
    rs2.load_data()
    rs2.fit_or_load_embeddings()

    ratings = rs1.ratings.copy()
    item_popularity = build_item_popularity(ratings)

    eval_users = get_eval_users(ratings)
    random.shuffle(eval_users)
    eval_users = eval_users[:EVAL_USER_LIMIT]

    print(f"\nUsers selected for evaluation: {len(eval_users)}")

    rs1_results = evaluate_recommender(rs1, "RS1_TFIDF", eval_users, item_popularity)
    rs2_results = evaluate_recommender(rs2, "RS2_SBERT", eval_users, item_popularity)

    results_df = pd.DataFrame([rs1_results, rs2_results])

    print("\nEvaluation Summary")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()