import pickle
import sys
from typing import Any, cast

import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, train_test_split, KFold
from sklearn.multioutput import MultiOutputClassifier
import warnings

from config import GENRE_COLUMNS
from prep_data import prep_data

## Helpers ##

def find_best_thresholds(model, X_val: np.ndarray, Y_val: np.ndarray) -> list:
    """
        For each genre, sweep thresholds [0.2 .. 0.7] and pick the one that
        maximises F1 on the validation set. Returns a list of floats, one per genre.
    """
    probas = np.array([est.predict_proba(X_val)[:, 1] for est in model.estimators_]).T
    thresholds = []

    for i, genre in enumerate(GENRE_COLUMNS):
        best_f1 = 0
        best_thresh = 0.5  # default
        for thresh in np.linspace(0.2, 0.7, num=11):
            preds = (probas[:, i] >= thresh).astype(int)
            f = f1_score(Y_val[:, i], preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thresh = thresh
        thresholds.append(round(best_thresh, 2))
        print(f"  {genre:<25} best threshold: {best_thresh:.2f}  (val F1: {best_f1:.3f})")
    return thresholds

def evaluate(model, X_test: np.ndarray, Y_test: np.ndarray, thresholds: list):
    """
    Prints a per-genre classification report using the tuned thresholds.
    """
    probas = np.array([est.predict_proba(X_test)[:, 1] for est in model.estimators_]).T
    Y_pred = np.zeros_like(probas, dtype=int)
    for i, thresh in enumerate(thresholds):
        Y_pred[:, i] = (probas[:, i] >= thresh).astype(int)
 
    print("\n--- Per-genre results (test set) ---")
    for i, genre in enumerate(GENRE_COLUMNS):
        report = cast(
            dict[str, Any],
            classification_report(
                Y_test[:, i], Y_pred[:, i],
                target_names=["no", "yes"],
                output_dict=True,
                labels=[0, 1],
                zero_division=0,
            ),
        )
        yes_report = cast(dict[str, float], report["yes"])
        p  = yes_report["precision"]
        r  = yes_report["recall"]
        f1 = yes_report["f1-score"]
        sup = int(yes_report["support"])
        print(f"  {genre:<25}  P: {p:.2f}  R: {r:.2f}  F1: {f1:.2f}  (n={sup})")
 
    macro_f1 = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    print(f"\n  Macro F1 (all genres): {macro_f1:.3f}")
    return macro_f1

## Main training loop ##

def train(survey_path: str):
    X, Y, x_mins, x_maxs = prep_data(survey_path)
    X_arr = X.values
    Y_arr = Y.values

    feature_names = X.columns.tolist()
    print(f"features ({len(feature_names)}): {feature_names}")
    print(f"labels ({Y.shape[1]}): {Y.columns.tolist()}")

    # 2. Train / val / test split  (60 / 20 / 20)
    #    Stratify on Rock (most common genre) as the anchor label.
    rock_idx = GENRE_COLUMNS.index("Rock")

    X_tmp, X_test, Y_tmp, Y_test = train_test_split(X_arr, Y_arr, test_size=0.2, random_state=42, stratify=Y_arr[:, rock_idx])
    X_train, X_val, Y_train, Y_val = train_test_split(X_tmp, Y_tmp, test_size=0.25, random_state=42, stratify=Y_tmp[:, rock_idx])  # 0.25 x 0.8 = 0.2

    print(f"Split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Train a multi-label classifier (one binary classifier per genre)
    base_clf = lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        random_state=42,
        verbose=-1,
    )

    # 4. Wrap in MultiOutputClassifier (one LightGBM per genre)
    model = MultiOutputClassifier(cast(BaseEstimator, base_clf), n_jobs=-1)

    # 5. 5 fold cross-validation on the training set (stratify on Rock)
    print("\nPerforming 5-fold cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, Y_train, cv=cv, scoring="f1_macro", n_jobs=-1) # groups=Y_train[:, rock_idx]
    print(f"CV Macro F1 scores: {cv_scores.round(3)}  (mean: {cv_scores.mean():.3f})")

    # 6. Fit on full training set, then tune thresholds on validation set
    print("\nTraining final model on full training set and tuning thresholds on validation set...")
    model.fit(X_train, Y_train)
    thresholds = find_best_thresholds(model, X_val, Y_val)

    # 7. Evaluate on held-out test set
    macro_f1 = evaluate(model, X_test, Y_test, thresholds)

    # 8. Retain on full data for production
    print("\nRetraining on full dataset for production model...")
    model.fit(X_arr, Y_arr)

    # 9. Save everything needed for inference
    artefacts = {
        "model": model,
        "thresholds": thresholds,
        "feature_names": feature_names,
        "x_mins": x_mins,
        "x_maxs": x_maxs,
        "genre_columns": GENRE_COLUMNS,
    }
    with open("personality_model.pkl", "wb") as f:
        pickle.dump(artefacts, f)

    print("\nDone. Artefacts saved to personality_model.pkl")
    print(f"Final test set Macro F1: {macro_f1:.3f}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/sauravkarki/repos/music/dataset/responses.csv"
    train(path)