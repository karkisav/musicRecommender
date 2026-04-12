# predict.py
# Loads the saved model and converts 10 OCEAN quiz answers into ranked seed genres.
# This is the file your application calls at inference time.
#
# Usage:
#   from predict import Predictor
#   p = Predictor("personality_model.pkl")
#   genres = p.predict(answers=[3, 4, 5, 2, 3, 4, 5, 3, 2, 1], top_n=5)

import pickle
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from config import TRAIT_QUESTION_MAP, TRAIT_NAMES, ADD_INTERACTIONS


class Predictor:
    def __init__(self, model_path: str = "personality_model.pkl"):
        with open(model_path, "rb") as f:
            artefacts = pickle.load(f)

        self.model          = artefacts["model"]
        self.thresholds     = artefacts["thresholds"]
        self.x_mins         = pd.Series(artefacts["x_mins"])
        self.x_maxs         = pd.Series(artefacts["x_maxs"])
        self.feature_names  = artefacts["feature_names"]
        self.genre_columns  = artefacts["genre_columns"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, answers: list, top_n: int = 5) -> list:
        """
        Takes 10 quiz answers (each 1–5) and returns the top_n seed genres
        ranked by predicted affinity probability.

        Args:
            answers : list of 10 ints/floats in range [1, 5]
                      ordered as Q1–Q10 matching TRAIT_QUESTION_MAP
            top_n   : how many genres to return (default 5)

        Returns:
            List of dicts: [{"genre": str, "probability": float}, ...]
            sorted highest probability first.
        """
        if len(answers) != 10:
            raise ValueError(f"Expected 10 answers, got {len(answers)}")
        if not all(1 <= a <= 5 for a in answers):
            raise ValueError("All answers must be between 1 and 5")

        X = self._answers_to_features(answers)
        probas = self._get_probabilities(X)

        ranked = sorted(
            zip(self.genre_columns, probas),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"genre": genre, "probability": round(float(prob), 3)}
            for genre, prob in ranked[:top_n]
        ]

    def predict_with_scores(self, answers: list) -> dict:
        """
        Returns full output: trait scores, all genre probabilities, and
        which genres cross the tuned threshold (binary prediction).
        Useful for debugging and building UI displays.
        """
        if len(answers) != 10:
            raise ValueError(f"Expected 10 answers, got {len(answers)}")

        trait_scores = self._compute_raw_traits(answers)
        X = self._answers_to_features(answers)
        probas = self._get_probabilities(X)

        genres = [
            {
                "genre": genre,
                "probability": round(float(prob), 3),
                "predicted": int(prob >= self.thresholds[i]),
            }
            for i, (genre, prob) in enumerate(zip(self.genre_columns, probas))
        ]
        genres.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "trait_scores": trait_scores,
            "genres": genres,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_raw_traits(self, answers: list) -> dict:
        """Maps Q1–Q10 answers to raw trait scores via broadcast-and-average."""
        question_keys = [f"Q{i}" for i in range(1, 11)]
        answer_map = dict(zip(question_keys, answers))

        trait_scores = {}
        for trait, questions in TRAIT_QUESTION_MAP.items():
            q_means = [answer_map[q] for q in questions]
            trait_scores[trait] = round(float(np.mean(q_means)), 3)
        return trait_scores

    def _answers_to_features(self, answers: list) -> np.ndarray:
        """
        Converts 10 quiz answers → normalised feature vector matching
        exactly what the model was trained on.
        """
        # Step 1: broadcast each answer to its mapped columns and average per trait
        question_keys = [f"Q{i}" for i in range(1, 11)]
        answer_map = dict(zip(question_keys, answers))

        trait_vals = {}
        for trait, questions in TRAIT_QUESTION_MAP.items():
            q_means = [answer_map[q] for q in questions]
            trait_vals[trait] = np.mean(q_means)

        X = pd.Series({t: trait_vals[t] for t in TRAIT_NAMES})

        # Step 2: add interaction terms if the model was trained with them
        if ADD_INTERACTIONS:
            X_df = X.to_frame().T
            for t1, t2 in combinations(TRAIT_NAMES, 2):
                col = f"{t1}_x_{t2}"
                X_df[col] = X_df[t1] * X_df[t2]
            X = X_df.iloc[0]

        # Step 3: normalise using the same min/max from training
        X_norm = (X - self.x_mins) / (self.x_maxs - self.x_mins + 1e-8)

        # Align to exact feature order the model expects
        X_aligned = X_norm[self.feature_names]
        return X_aligned.values.reshape(1, -1)

    def _get_probabilities(self, X: np.ndarray) -> list:
        """Returns raw probability for the positive class per genre."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probas = [
                est.predict_proba(X)[0, 1]
                for est in self.model.estimators_
            ]
        return probas


# ---------------------------------------------------------------------------
# Run directly to test with a manual set of answers
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    model_path = sys.argv[1] if len(sys.argv) > 1 else "personality_model.pkl"
    p = Predictor(model_path)

    print("Enter your answers for Q1–Q10 (1=strongly disagree, 5=strongly agree)")
    print("Q1: I love exploring new ideas, books, or art")
    print("Q2: I enjoy learning foreign languages or understanding how the mind works")
    print("Q3: I plan tasks and hate leaving things unfinished")
    print("Q4: People can count on me — I follow through on what I say")
    print("Q5: I recharge by being around people — socialising energises me")
    print("Q6: I have a lot of physical energy — I love dancing or moving to music")
    print("Q7: I genuinely care about others' feelings and go out of my way to help")
    print("Q8: I donate to causes and care about children and communities")
    print("Q9: My mood changes often — I feel things intensely and get frustrated easily")
    print("Q10: I often worry about how life is going and dwell on my struggles")
    print()

    try:
        raw = input("Enter 10 answers separated by spaces: ")
        answers = [int(x) for x in raw.strip().split()]
    except (ValueError, EOFError):
        # Default test case: high openness, high extraversion, low neuroticism
        print("Using default test answers: [4, 5, 3, 4, 5, 4, 4, 3, 2, 2]")
        answers = [4, 5, 3, 4, 5, 4, 4, 3, 2, 2]

    result = p.predict_with_scores(answers)

    print("\n--- Trait scores ---")
    for trait, score in result["trait_scores"].items():
        bar = "█" * int(score * 4)
        print(f"  {trait:<20} {score:.2f}  {bar}")

    print("\n--- Seed genres (ranked by probability) ---")
    for g in result["genres"]:
        flag = "✓" if g["predicted"] else " "
        bar = "█" * int(g["probability"] * 20)
        print(f"  {flag} {g['genre']:<22} {g['probability']:.3f}  {bar}")