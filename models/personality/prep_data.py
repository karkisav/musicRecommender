import numpy as np
import pandas as pd
from itertools import combinations
from config import (
    GENRE_COLUMNS, GENRE_LIKE_THRESHOLD,
    TRAIT_COLUMNS, TRAIT_NAMES,
    ADD_INTERACTIONS,
)

def load_survey(path: str) -> pd.DataFrame:
    """Load survey data from CSV and return as DataFrame."""
    df = pd.read_csv(path)
    return df

def compute_trait_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trait scores by averaging relevant survey columns."""
    trait_scores = {}
    for trait, cols in TRAIT_COLUMNS.items():
        subset = df[cols].copy()
        # Handle missing values by filling with trait mean (or could drop rows)
        subset = subset.fillna(subset.mean())
        trait_scores[trait] = subset.mean(axis=1)
    return pd.DataFrame(trait_scores, columns=TRAIT_NAMES)

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add pairwise interaction terms between traits."""
    for trait1, trait2 in combinations(TRAIT_NAMES, 2):
        interaction_col = f"{trait1}_x_{trait2}"
        df[interaction_col] = df[trait1] * df[trait2]
    return df

def build_genre_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert genre ratings to binary labels based on threshold."""
    genre_labels = {}
    for genre in GENRE_COLUMNS:
        if genre in df.columns:
            genre_labels[genre] = (df[genre] >= GENRE_LIKE_THRESHOLD).astype(int)
        else:
            print(f"Warning: Genre column '{genre}' not found in data.")
            genre_labels[genre] = np.zeros(len(df), dtype=int)  # default to 0 if missing
    return pd.DataFrame(genre_labels, columns=GENRE_COLUMNS)

def normalize_traits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Normalize all scores to 0–1 range."""
    mins = df.min()
    maxs = df.max()
    normalized = (df - mins) / (maxs - mins + 1e-8)  # add small epsilon to avoid division by zero
    return normalized, mins, maxs

def prep_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_survey(path)
    trait_scores = compute_trait_scores(df)
    if ADD_INTERACTIONS:
        trait_scores = add_interactions(trait_scores)
    X, x_mins, x_maxs = normalize_traits(trait_scores)
    Y = build_genre_labels(df)          # single call, remove the dead one above
    return X, Y, x_mins, x_maxs

if __name__ == "__main__":
    import sys
 
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/sauravkarki/repos/music/dataset/responses.csv"
    df = load_survey(path)
 
    X = compute_trait_scores(df)
    Y = build_genre_labels(df)
 
    print("\n--- Trait score distributions (raw, before normalization) ---")
    print(X.describe().round(2))
 
    print("\n--- Genre label prevalence (% of respondents who like each genre) ---")
    prevalence = Y.mean().sort_values(ascending=False) * 100
    for genre, pct in prevalence.items():
        bar = "█" * int(pct / 2)
        print(f"  {genre:<22} {pct:5.1f}%  {bar}")
 
    print(f"\nTotal samples: {len(X)}")
    print(f"Feature count: {X.shape[1]} {'(5 traits + 10 interactions)' if ADD_INTERACTIONS else '(5 traits only)'}")
    print(f"Label count:   {Y.shape[1]} genres")
    print(f"Avg genres liked per person: {Y.sum(axis=1).mean():.1f}")