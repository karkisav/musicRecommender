import os
import secrets
import sys
from pathlib import Path
from typing import Dict

import requests
from fastapi import Cookie, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import PlaylistRequest, PredictRequest, PredictResponse
from api.spotify import (
    add_tracks_to_playlist,
    build_login_url,
    create_playlist,
    filter_recommendations_for_discovery,
    exchange_code_for_token,
    get_available_seed_genres,
    get_recommended_tracks,
    get_user_profile,
    get_user_top_artists,
    get_user_top_tracks,
    map_model_genres_to_spotify,
    search_tracks_by_genres,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "personality"

# Allow importing predict.py and config.py that currently use local imports.
sys.path.append(str(MODEL_DIR))
from predict import Predictor  # noqa: E402

app = FastAPI(title="PulsePersona API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory token store keyed by cookie session id.
SESSIONS: Dict[str, Dict] = {}
OAUTH_STATES: Dict[str, Dict] = {}


def get_model_path() -> Path:
    candidates = [
        ROOT / "personality_model.pkl",
        MODEL_DIR / "personality_model.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise RuntimeError("Model file not found. Train model first to create personality_model.pkl")


PREDICTOR = Predictor(str(get_model_path()))


def require_session(session_id: str | None) -> Dict:
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=401, detail="Spotify session not found. Login first.")
    session = SESSIONS[session_id]
    if not session.get("access_token"):
        raise HTTPException(status_code=401, detail="Spotify access token missing. Login again.")
    return session


def spotify_error_detail(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return "Spotify request failed."

    message = response.text
    try:
        payload = response.json()
        if isinstance(payload, dict):
            if isinstance(payload.get("error"), dict):
                message = payload["error"].get("message") or message
            else:
                message = payload.get("error_description") or payload.get("error") or message
    except ValueError:
        pass

    message = (message or "Spotify request failed").strip()
    return f"Spotify API error ({response.status_code}): {message}"


@app.get("/health")
def health():
    return {"status": "ok"}


WEB_DIR = ROOT / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.get("/")
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index_path))


@app.post("/predict/full", response_model=PredictResponse)
def predict_full(payload: PredictRequest):
    try:
        result = PREDICTOR.predict_with_scores(payload.answers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "trait_scores": result["trait_scores"],
        "genres": result["genres"],
    }


@app.get("/spotify/login")
def spotify_login(response: Response):
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/spotify/callback")
    login_url, state = build_login_url(redirect_uri)

    sid = secrets.token_urlsafe(24)
    OAUTH_STATES[state] = {"sid": sid, "frontend_return": ""}
    response.set_cookie("pp_session", sid, httponly=True, samesite="lax")

    return {"login_url": login_url}


@app.get("/spotify/login-redirect")
def spotify_login_redirect(
    response: Response,
    frontend_return: str = Query(default="http://127.0.0.1:8000/"),
):
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/spotify/callback")
    login_url, state = build_login_url(redirect_uri)

    sid = secrets.token_urlsafe(24)
    OAUTH_STATES[state] = {"sid": sid, "frontend_return": frontend_return}
    response = RedirectResponse(login_url)
    response.set_cookie("pp_session", sid, httponly=True, samesite="lax")
    return response


@app.get("/spotify/callback")
def spotify_callback(
    code: str = Query(...),
    state: str = Query(...),
    pp_session: str | None = Cookie(default=None),
):
    state_data = OAUTH_STATES.pop(state, None)
    if not state_data:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    expected_sid = state_data.get("sid")
    if not isinstance(expected_sid, str) or not expected_sid:
        raise HTTPException(status_code=400, detail="Invalid OAuth session state")
    frontend_return = state_data.get("frontend_return") or "http://127.0.0.1:8000/"

    sid = pp_session or expected_sid
    if sid != expected_sid:
        raise HTTPException(status_code=400, detail="Session mismatch")

    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/spotify/callback")
    token_data = exchange_code_for_token(code, redirect_uri)

    SESSIONS[sid] = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "token_type": token_data.get("token_type", "Bearer"),
        "expires_in": token_data.get("expires_in"),
    }

    profile = get_user_profile(SESSIONS[sid]["access_token"])
    SESSIONS[sid]["user_id"] = profile.get("id")

    redirect_target = f"{frontend_return}?spotify=ok"
    return RedirectResponse(redirect_target)


@app.get("/spotify/me")
def spotify_me(pp_session: str | None = Cookie(default=None)):
    session = require_session(pp_session)
    profile = get_user_profile(session["access_token"])
    return {
        "id": profile.get("id"),
        "display_name": profile.get("display_name"),
        "email": profile.get("email"),
    }


@app.post("/spotify/create-playlist")
def spotify_create_playlist(payload: PlaylistRequest, pp_session: str | None = Cookie(default=None)):
    session = require_session(pp_session)

    result = PREDICTOR.predict_with_scores(payload.answers)
    ranked = sorted(result["genres"], key=lambda g: g["probability"], reverse=True)
    model_top = [g["genre"] for g in ranked[: payload.top_n]]

    access_token = session["access_token"]
    try:
        available = get_available_seed_genres(access_token)
        genre_seeds = map_model_genres_to_spotify(model_top, available, max_count=3)

        # Pull enough profile data to filter familiar songs and artists out of recommendations.
        user_top_tracks = get_user_top_tracks(access_token, limit=max(payload.limit, 50))
        user_top_artists = get_user_top_artists(access_token, limit=50)
        top_tracks: list[str] = []
        top_artists: list[str] = []

        # Favor genre-only discovery; only add a tiny personal anchor when genre seeds are scarce.
        if len(genre_seeds) == 1 and user_top_artists:
            top_artists = user_top_artists[:1]
        elif not genre_seeds:
            top_tracks = user_top_tracks[:2]
            top_artists = user_top_artists[:2]

        # Keep total seeds <= 5 for Spotify recommendations.
        while len(genre_seeds) + len(top_tracks) + len(top_artists) > 5:
            if top_tracks:
                top_tracks.pop()
            elif top_artists:
                top_artists.pop()
            elif genre_seeds:
                genre_seeds.pop()   

        recommendation_source = "spotify_recommendations"
        tracks = []
        raw_recommendation_tracks = []
        try:
            raw_recommendation_tracks = get_recommended_tracks(
                access_token=access_token,
                seed_genres=genre_seeds,
                seed_tracks=top_tracks,
                seed_artists=top_artists,
                limit=max(payload.limit * 2, 40),
            )
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            # Spotify recommendations endpoints can fail for some apps/accounts.
            # Fall back to genre search rather than user's own top tracks.
            if status_code not in (400, 404):
                raise

        tracks = filter_recommendations_for_discovery(
            raw_recommendation_tracks,
            exclude_track_ids=user_top_tracks,
            exclude_artist_ids=user_top_artists,
            limit=payload.limit,
        )

        if not tracks and genre_seeds:
            recommendation_source = "genre_search_fallback"
            searched = search_tracks_by_genres(access_token, genre_seeds, limit=max(payload.limit * 2, 40))
            tracks = filter_recommendations_for_discovery(
                searched,
                exclude_track_ids=user_top_tracks,
                exclude_artist_ids=user_top_artists,
                limit=payload.limit,
            )

        if tracks:
            track_uris = [t["uri"] for t in tracks if t.get("uri")]
        else:
            track_uris = []

        if not track_uris:
            raise HTTPException(
                status_code=400,
                detail="Could not assemble playlist tracks from Spotify recommendations or genre search.",
            )

        user_id = session.get("user_id")
        if not user_id:
            profile = get_user_profile(access_token)
            user_id = profile.get("id")
            session["user_id"] = user_id
        if not user_id:
            raise HTTPException(status_code=400, detail="Spotify user ID not found")

        playlist = create_playlist(
            access_token=access_token,
            user_id=user_id,
            name=payload.playlist_name,
            description="Generated by PulsePersona using personality traits + Spotify profile",
            is_public=payload.is_public,
        )

        playlist_id = playlist.get("id")
        if not playlist_id:
            raise HTTPException(status_code=502, detail="Spotify did not return a playlist ID")
        add_tracks_to_playlist(access_token, playlist_id, track_uris)
    except requests.HTTPError as exc:
        raise HTTPException(status_code=502, detail=spotify_error_detail(exc)) from exc

    return {
        "playlist_id": playlist_id,
        "playlist_url": playlist.get("external_urls", {}).get("spotify"),
        "seed_genres_model": model_top,
        "seed_genres_spotify": genre_seeds,
        "seed_tracks": top_tracks,
        "seed_artists": top_artists,
        "recommendation_source": recommendation_source,
        "tracks_added": len(track_uris),
    }
