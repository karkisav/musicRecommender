# Music Recommender: Personality-Based Genre Prediction

This project develops a music recommender system that predicts music genre preferences based on an individual's personality traits. Utilizing the Young People Survey dataset, the system currently employs a baseline machine learning model to map Big Five personality proxy features to genre likelihoods.

## Features

-   **Personality Trait Extraction**: Derives Big Five personality scores (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) from survey responses.
-   **Multi-Label Genre Prediction**: Predicts the likelihood of a user enjoying various music genres using a multi-output LightGBM classifier.
-   **Tunable Thresholds**: Optimizes genre-specific classification thresholds to maximize prediction accuracy.
-   **Interactive Prediction CLI**: Allows users to input 10 quiz answers and receive immediate genre recommendations.
-   **Model Persistence**: Saves trained models and associated artifacts for easy deployment and inference.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have Python 3.8+ installed. The project dependencies are listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/karkisav/music.git
    cd music
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the dataset:**

    The project uses the Young People Survey dataset. You can download and prepare it using the provided script:

    ```bash
    ./download_dataset.sh
    ```
    This script will download `young-people-survey.zip` and extract `responses.csv` into a `dataset/` directory.

## Usage

### Frontend Prototype (Website UI)

A frontend prototype has been added in the `web/` folder. It includes:

- A 10-question personality quiz UI
- Animated progress and transitions
- Results view for trait profile and top genres
- API integration hook to a backend endpoint (`POST /predict/full`)
- Local fallback scoring if the backend is not running yet

To run it quickly:

1. Open `web/index.html` in your browser.
2. Complete the quiz and view recommendations.

If you run a backend at `http://127.0.0.1:8000`, the UI will automatically use real model predictions. Otherwise, it uses fallback client-side scoring so the experience still works during frontend development.

### Full Web App (Model + Spotify Auth + Playlist Generation)

The project now includes a FastAPI backend in `api/` and a frontend in `web/`.

#### What it does

- Uses 10-question personality input (1-5 each) from the frontend
- Runs your trained model (`predict_with_scores`) to get ranked seed genres
- Maps seed genres to Spotify-supported genre seeds
- Uses Spotify OAuth (user login) to access user top artists/tracks
- Combines model seed genres + user profile seeds to get recommendations
- Creates a playlist in the user account and adds recommended tracks

#### Spotify Developer Setup

1. Create an app in Spotify Developer Dashboard.
2. Set Redirect URI to:

    `http://127.0.0.1:8000/spotify/callback`

3. Set environment variables before running backend:

```bash
set SPOTIFY_CLIENT_ID=your_client_id
set SPOTIFY_CLIENT_SECRET=your_client_secret
set SPOTIFY_REDIRECT_URI=http://127.0.0.1:8000/spotify/callback
```

#### Run the app

1. Make sure the model exists:

```bash
python models/personality/train.py
```

2. Start FastAPI server:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

3. Open this URL in browser:

    `http://127.0.0.1:8000/`

4. Complete the smiley quiz and see seed genres.
5. Click **Connect Spotify**, sign in, then click **Create My Playlist**.

### Training the Model

To train the personality-to-genre prediction model, run the `train.py` script. This will process the dataset, train a LightGBM multi-output classifier, evaluate its performance, and save the trained model artifacts to `personality_model.pkl`.

```bash
python models/personality/train.py
```

### Making Predictions

You can use the `Predictor` class in `predict.py` to get genre recommendations. The script also includes an interactive command-line interface for quick testing.

#### Interactive CLI

Run the `predict.py` script directly to answer 10 personality-related questions and receive ranked genre predictions:

```bash
python models/personality/predict.py
```

#### Programmatic Usage

To integrate the predictor into your own application:

```python
from models.personality.predict import Predictor

# Initialize the predictor with the path to the trained model
p = Predictor("personality_model.pkl")

# Example answers for Q1-Q10 (each 1-5)
# These questions map to Big Five personality traits as defined in config.py
# Q1: Openness (ideas/art), Q2: Openness (languages/psychology)
# Q3: Conscientiousness (planning), Q4: Conscientiousness (reliability)
# Q5: Extraversion (socializing), Q6: Extraversion (energy/dancing)
# Q7: Agreeableness (empathy/help), Q8: Agreeableness (charity/community)
# Q9: Neuroticism (mood swings/frustration), Q10: Neuroticism (worry/struggles)
answers = [4, 5, 3, 4, 5, 4, 4, 3, 2, 2] # Example: High Openness, High Extraversion, Low Neuroticism

# Get top N genre recommendations
top_genres = p.predict(answers=answers, top_n=5)
print("Top 5 Recommended Genres:")
for genre_info in top_genres:
    print(f"- {genre_info['genre']}: {genre_info['probability']:.2f}")

# Get full prediction details, including trait scores and all genre probabilities
full_result = p.predict_with_scores(answers=answers)
print("\nFull Prediction Details:")
print("Trait Scores:", full_result["trait_scores"])
print("Genres with Probabilities and Predictions:")
for genre_data in full_result["genres"]:
    print(f"- {genre_data['genre']}: Probability={genre_data['probability']:.2f}, Predicted={bool(genre_data['predicted'])}")
```

## Project Structure

```
.github/
├── workflows/
│   └── python-app.yml
├── README.md
├── download_dataset.sh
├── explore.ipynb
├── models/
│   └── personality/
│       ├── config.py
│       ├── personality_model.pkl  # Generated after training
│       ├── predict.py
│       ├── prep_data.py
│       ├── train.py
│       └── usage.py
├── requirements.txt
└── young-people-survey.zip
```

-   `config.py`: Defines mappings for personality traits to survey questions, lists target music genres, and sets model parameters.
-   `prep_data.py`: Handles data loading, feature engineering (trait score computation, interaction terms), and data normalization.
-   `train.py`: Script for training the multi-label classification model, including data splitting, cross-validation, and model serialization.
-   `predict.py`: Contains the `Predictor` class for loading the trained model and making genre predictions, along with an interactive CLI.
-   `explore.ipynb`: Jupyter notebook for initial data exploration and analysis.
-   `download_dataset.sh`: Script to download and extract the dataset.
-   `requirements.txt`: Lists Python package dependencies.

## Development Roadmap

### Current Progress

-   Implemented Big Five personality proxy feature extraction.
-   Developed a baseline multi-output LightGBM regressor for genre preference prediction.
-   Established a robust training and evaluation pipeline with cross-validation and threshold tuning.
-   Created an inference module with an interactive CLI for predictions.

### Future Enhancements

-   **Advanced Feature Engineering**: Explore more sophisticated methods for deriving personality features.
-   **Model Comparison**: Evaluate stronger models beyond the current baseline and conduct thorough cross-validation.
-   **Top-K Recommendation Logic**: Implement and refine algorithms for generating top-K genre recommendations from predicted preference scores.
-   **Spotify Integration**: Integrate with the Spotify API to provide personalized recommendations and potentially map mood to Spotify audio features (e.g., `target_valence`, `energy`, `tempo`).
-   **Robustness**: Add fallback mechanisms for sparse user profiles or low model confidence.

## Contributing Personel

Koustubh: 23bc023
Krishna: 23bcs074
Saurav: 23bcs118
Nirbhay: 23bec032

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

-   Young People Survey dataset
-   Scikit-learn, LightGBM, Pandas, NumPy communities
