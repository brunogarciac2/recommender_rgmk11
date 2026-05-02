from pathlib import Path
import pandas as pd


# ----------------------------
# Config
# ----------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
SAMPLE_SIZE = 150_000


# ----------------------------
# Helpers
# ----------------------------
def clean_genres(genres: str) -> str:
    """Convert MovieLens pipe-separated genres into space-separated text."""
    if pd.isna(genres) or genres == "(no genres listed)":
        return ""
    return genres.replace("|", " ").strip().lower()


def clean_title(title: str) -> str:
    """Basic title cleanup."""
    if pd.isna(title):
        return ""
    return str(title).strip().lower()


def clean_tag(tag: str) -> str:
    """Basic tag cleanup."""
    if pd.isna(tag):
        return ""
    return str(tag).strip().lower()


def aggregate_tags(tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all tags for each movie into a single text string.
    Duplicate tags per movie are removed.
    """
    tags_df = tags_df.copy()
    tags_df["tag"] = tags_df["tag"].map(clean_tag)
    tags_df = tags_df[tags_df["tag"] != ""]

    agg = (
        tags_df.groupby("movieId")["tag"]
        .apply(lambda s: " ".join(sorted(set(s))))
        .reset_index()
        .rename(columns={"tag": "tags_text"})
    )
    return agg


def build_movie_profiles(movies_df: pd.DataFrame, tags_agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one text profile per movie:
    title + genres + tags
    """
    movies = movies_df.copy()
    movies["title_clean"] = movies["title"].map(clean_title)
    movies["genres_text"] = movies["genres"].map(clean_genres)

    movie_profiles = movies.merge(tags_agg_df, on="movieId", how="left")
    movie_profiles["tags_text"] = movie_profiles["tags_text"].fillna("")

    movie_profiles["profile_text"] = (
        movie_profiles["title_clean"] + " "
        + movie_profiles["genres_text"] + " "
        + movie_profiles["tags_text"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return movie_profiles[
        ["movieId", "title", "genres", "title_clean", "genres_text", "tags_text", "profile_text"]
    ]


def sample_ratings(ratings_df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    """
    Randomly sample n ratings from the full ratings file.
    """
    if len(ratings_df) < n:
        raise ValueError(f"Requested sample size {n}, but ratings file only has {len(ratings_df)} rows.")
    return ratings_df.sample(n=n, random_state=random_state).copy()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"
    tags_path = DATA_DIR / "tags.csv"

    print("Loading CSV files...")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    tags = pd.read_csv(tags_path)

    print(f"Full ratings shape: {ratings.shape}")
    print(f"Movies shape: {movies.shape}")
    print(f"Tags shape: {tags.shape}")

    print(f"\nSampling {SAMPLE_SIZE:,} ratings...")
    ratings_sample = sample_ratings(ratings, SAMPLE_SIZE, RANDOM_STATE)

    sample_out_path = OUTPUT_DIR / "ratings_sample_150k.csv"
    ratings_sample.to_csv(sample_out_path, index=False)
    print(f"Saved sampled ratings to: {sample_out_path}")

    print("\nAggregating tags per movie...")
    tags_agg = aggregate_tags(tags)
    tags_agg_out_path = OUTPUT_DIR / "movie_tags_aggregated.csv"
    tags_agg.to_csv(tags_agg_out_path, index=False)
    print(f"Saved aggregated tags to: {tags_agg_out_path}")

    print("\nBuilding movie text profiles...")
    movie_profiles = build_movie_profiles(movies, tags_agg)
    movie_profiles_out_path = OUTPUT_DIR / "movie_profiles.csv"
    movie_profiles.to_csv(movie_profiles_out_path, index=False)
    print(f"Saved movie profiles to: {movie_profiles_out_path}")

    print("\nMerging sampled ratings with movie profiles...")
    ratings_with_profiles = ratings_sample.merge(
        movie_profiles[["movieId", "title", "genres", "profile_text"]],
        on="movieId",
        how="left",
    )

    merged_out_path = OUTPUT_DIR / "ratings_with_movie_profiles.csv"
    ratings_with_profiles.to_csv(merged_out_path, index=False)
    print(f"Saved merged ratings+profiles to: {merged_out_path}")

    print("\nDone.")
    print("\nSummary:")
    print(f"- Sampled ratings: {len(ratings_sample):,}")
    print(f"- Users in sample: {ratings_sample['userId'].nunique():,}")
    print(f"- Movies in sample: {ratings_sample['movieId'].nunique():,}")
    print(f"- Movie profiles built: {len(movie_profiles):,}")


if __name__ == "__main__":
    main()