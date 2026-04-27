# PulsePersona System Report

Date: April 24, 2026

## Scope Reviewed

This report is based on a full read-through of the important code and data flow across:

- [README.md](README.md)
- [requirements.txt](requirements.txt)
- [download_dataset.sh](download_dataset.sh)
- [api/main.py](api/main.py)
- [api/schemas.py](api/schemas.py)
- [api/spotify.py](api/spotify.py)
- [models/personality/config.py](models/personality/config.py)
- [models/personality/prep_data.py](models/personality/prep_data.py)
- [models/personality/train.py](models/personality/train.py)
- [models/personality/predict.py](models/personality/predict.py)
- [models/personality/usage.py](models/personality/usage.py)
- [web/index.html](web/index.html)
- [web/app.js](web/app.js)
- [web/styles.css](web/styles.css)
- [dataset/columns.csv](dataset/columns.csv)
- [dataset/responses.csv](dataset/responses.csv)
- [explore.ipynb](explore.ipynb)

The serialized model artifact was also inspected at:

- [models/personality/personality_model.pkl](models/personality/personality_model.pkl)

## 1) What Model Is Used in Production

The live API uses a multi-label classification setup:

- MultiOutputClassifier
- Base estimator: LightGBM LGBMClassifier
- One binary classifier per genre (17 total genres)
- Feature vector size: 15
  - 5 Big Five trait scores
  - 10 pairwise trait interaction terms

Implementation sources:

- Training: [models/personality/train.py](models/personality/train.py)
- Inference: [models/personality/predict.py](models/personality/predict.py)

Artifact-level confirmation from the current model file:

- model_type: MultiOutputClassifier
- base_estimator_type: LGBMClassifier
- genres: 17
- features: 15
- thresholds_count: 17

## 2) Why This Model Choice Fits the Problem

The code indicates several intentional choices:

- The output is multi-label, not single-class. A user can like multiple genres.
- Per-genre class imbalance is handled with LightGBM settings (is_unbalance=True).
- Personality-to-genre relations are likely nonlinear; boosting handles this better than strict linear assumptions.
- Thresholds are tuned per genre on validation data instead of forcing 0.5 for all labels.

This is visible in:

- [models/personality/train.py](models/personality/train.py)
- [models/personality/predict.py](models/personality/predict.py)

## 3) How Prediction Is Made (End-to-End Logic)

### 3.1 Training Data and Label Construction

From [models/personality/prep_data.py](models/personality/prep_data.py):

1. Load survey data from [dataset/responses.csv](dataset/responses.csv).
2. Build Big Five proxy trait scores by averaging mapped survey columns.
3. Optionally add pairwise interactions (enabled in config).
4. Normalize features with min-max scaling.
5. Convert genre ratings to binary labels using threshold >= 4.

Config details are defined in:

- [models/personality/config.py](models/personality/config.py)

### 3.2 Model Training Pipeline

From [models/personality/train.py](models/personality/train.py):

1. Split into train/validation/test with stratification anchor on Rock.
2. Train MultiOutputClassifier(LGBMClassifier).
3. Run cross-validation (macro F1 scoring).
4. Tune a separate decision threshold per genre using validation F1 sweep.
5. Evaluate on held-out test set.
6. Retrain on full dataset for production artifact.
7. Save model + preprocessing metadata to personality_model.pkl.

Saved artifact fields include:

- model
- thresholds
- feature_names
- x_mins
- x_maxs
- genre_columns

### 3.3 Inference at Runtime

From [models/personality/predict.py](models/personality/predict.py):

1. Accept exactly 10 answers (1 to 5).
2. Convert answers Q1 to Q10 into 5 trait means.
3. Add interaction terms to match training feature space.
4. Apply min-max normalization with saved x_mins/x_maxs.
5. Align feature order exactly to saved feature_names.
6. Get per-genre probabilities from each estimator.
7. Mark predicted = 1 when probability >= tuned threshold for that genre.
8. Sort genres by probability descending.

### 3.4 API Prediction Route

From [api/main.py](api/main.py):

- POST /predict/full calls Predictor.predict_with_scores.
- Response model is defined in [api/schemas.py](api/schemas.py).
- It returns:
  - trait_scores
  - genres list with genre, probability, predicted

## 4) How It Ties to Frontend and Survey

Frontend implementation:

- Structure: [web/index.html](web/index.html)
- Logic: [web/app.js](web/app.js)
- Styling: [web/styles.css](web/styles.css)

Flow:

1. User answers a 10-question Likert-style quiz (scores 1 to 5).
2. Frontend sends answers to POST /predict/full.
3. Backend returns trait scores and ranked genre probabilities.
4. Frontend renders:
   - trait bars
   - top genre list
   - explanation text based on strongest trait/top genre

Important detail:

- If backend prediction fails, frontend falls back to a local heuristic predictor in [web/app.js](web/app.js) so the UI still returns a result.

## 5) Spotify API Usage and Integration

Spotify integration lives in:

- [api/spotify.py](api/spotify.py)
- [api/main.py](api/main.py)

### 5.1 Auth and Session Flow

- OAuth login URL generation
- Code exchange for access token
- Cookie-based session key storage in memory
- User profile retrieval for user id

Endpoints used in backend:

- GET /spotify/login
- GET /spotify/login-redirect
- GET /spotify/callback
- GET /spotify/me
- POST /spotify/create-playlist

### 5.2 Spotify Web API Calls Used

- Accounts:
  - /authorize
  - /api/token
- User/profile:
  - /v1/me
  - /v1/me/top/tracks
  - /v1/me/top/artists
- Recommendations and search:
  - /v1/recommendations/available-genre-seeds
  - /v1/recommendations
  - /v1/search (fallback)
- Playlist operations:
  - /v1/users/{user_id}/playlists
  - /v1/playlists/{playlist_id}/tracks

### 5.3 Playlist Construction Logic

From [api/main.py](api/main.py):

1. Predict ranked genres from quiz answers.
2. Take top model genres and map them to Spotify genre seeds.
3. Pull user top tracks/artists for personalization context.
4. Prefer discovery-oriented recommendations by filtering out familiar tracks/artists.
5. If recommendations are unavailable or fail, fall back to genre search.
6. Create a playlist and add selected track URIs.
7. Return playlist metadata and diagnostics.

Genre mapping is defined in [api/spotify.py](api/spotify.py) via MODEL_TO_SPOTIFY_GENRE.

## 6) Input Validation and Contracts

Schema constraints in [api/schemas.py](api/schemas.py):

- answers must be exactly 10 integers in [1, 5]
- top_n is bounded [1, 10]
- playlist limit is bounded [10, 100]

These constraints keep frontend payloads consistent with predictor expectations.

## 7) Outcome for the User

There are two practical outcomes.

### 7.1 Prediction Outcome

The user gets:

- A Big Five trait profile
- Ranked genre probabilities
- Binary genre predictions using tuned thresholds

### 7.2 Spotify Outcome

If connected, the user gets:

- A newly created playlist in their Spotify account
- A direct playlist URL
- Additional metadata showing which seeds were used and fallback source

Returned fields from playlist route include:

- playlist_id
- playlist_url
- seed_genres_model
- seed_genres_spotify
- seed_tracks
- seed_artists
- recommendation_source
- tracks_added

## 8) Notebook Findings vs Production Path

The notebook [explore.ipynb](explore.ipynb) is an exploratory Phase 1 baseline.

Recovered notebook outputs indicate:

- Dataset size: 1010 rows, 150 columns
- Music columns found: 17 of 17
- Duplicate rows: 0
- Suitability verdict: Accept
- Baseline regression comparison:
  - RandomForest MultiOutput: MAE 1.0283, RMSE 1.2424
  - RidgeCV MultiOutput: MAE 0.9782, RMSE 1.1758

Production has moved beyond this baseline to a binary multi-label classification pipeline with LightGBM and per-genre thresholds, which is aligned with the final application objective.

## 9) Summary

The architecture is coherent and production-oriented:

- Survey answers -> Big Five proxies -> trained multi-label genre model
- Predicted genres -> Spotify seed mapping -> recommendations/search fallback
- Final output -> on-platform Spotify playlist and interpretable trait/genre UI

In short, the system predicts what genres a user is likely to like from personality signals, then turns those predictions into an actionable playlist outcome using Spotify APIs and robust fallback logic.
