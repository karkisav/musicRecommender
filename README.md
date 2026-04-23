# PulsePersona (Personality -> Music + Spotify Playlist)

FastAPI app + web UI that:

- takes a 10-question personality quiz,
- predicts likely music genres using a trained model,
- optionally creates a Spotify playlist using your profile + Spotify top tracks/artists.

## Quick Start (macOS/Linux)

Run all commands from the repo root.

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model (required once before starting API):

```bash
python models/personality/train.py
```

4. Load Spotify env vars from `api/.env`:

```bash
set -a
source api/.env
set +a
```

5. Start the backend:

```bash
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000 --env-file /Users/sauravkarki/repos/music/api/.env  
```

6. Open:

```text
http://127.0.0.1:8000/
```

## Spotify Setup

In Spotify Developer Dashboard, add this redirect URI:

```text
http://127.0.0.1:8000/spotify/callback
```

Your `api/.env` should contain:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8000/spotify/callback
```

Important: this project reads Spotify credentials with `os.getenv(...)` and does not auto-load `.env` files. You must export them in your shell (step 4 above) before running `uvicorn`.

## Useful Endpoints

- `GET /health`
- `POST /predict/full`
- `GET /spotify/login-redirect`
- `POST /spotify/create-playlist`

## Common Issues

- `Model file not found`: run `python models/personality/train.py` first.
- `Missing SPOTIFY_CLIENT_ID`: env vars were not exported in your current shell.
- Spotify recommendation seed endpoint can return `404` for some apps; backend already falls back to mapped genres/top tracks.

## Dataset Note

`dataset/responses.csv` is expected for training. If it is missing, download the Young People Survey dataset and place `responses.csv` inside `dataset/`.


python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000 --env-file /Users/sauravkarki/repos/music/api/.env  