import argparse

from rs1_tfidf import TFIDFRecommender
from rs2_sbert import SBERTRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Movie recommender CLI")
    parser.add_argument(
        "--rs",
        type=str,
        required=True,
        choices=["rs1", "rs2"],
        help="Which recommender system to use: rs1 (TF-IDF) or rs2 (SBERT)"
    )
    parser.add_argument(
        "--user",
        type=int,
        required=True,
        help="User ID to generate recommendations for"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of recommendations to return"
    )

    args = parser.parse_args()

    if args.rs == "rs1":
        recommender = TFIDFRecommender()
    else:
        recommender = SBERTRecommender()

    print(f"\nLoading {args.rs.upper()} recommender...")
    recommender.load_data()

    if args.rs == "rs1":
        recommender.fit()
    else:
        recommender.fit_or_load_embeddings()

    if not recommender.user_is_eligible(args.user):
        print(
            f"User {args.user} does not have at least "
            f"{recommender.min_positive_ratings_per_user} positive ratings."
        )
        return

    recs = recommender.recommend(args.user, top_k=args.topk)

    seen_movie_ids = recommender.get_seen_movie_ids(args.user)
    already_seen_check = recs["movieId"].isin(seen_movie_ids).any()

    print(f"\nRecommendations for user {args.user} using {args.rs.upper()}:")
    print(f"Any recommended movies already seen? {already_seen_check}\n")

    print(recs[["movieId", "title", "score", "explanation"]].to_string(index=False))


if __name__ == "__main__":
    main()