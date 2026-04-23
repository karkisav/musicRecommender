import base64
import os
import secrets
from typing import Dict, List, Tuple

import requests

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

SPOTIFY_SCOPE = "user-read-email user-read-private user-top-read playlist-modify-public playlist-modify-private"

# Map model output genre names to Spotify seed genres.
MODEL_TO_SPOTIFY_GENRE = {
    "Dance": "dance",
    "Folk": "folk",
    "Country": "country",
    "Classical music": "classical",
    "Musical": "show-tunes",
    "Pop": "pop",
    "Rock": "rock",
    "Metal or Hardrock": "metal",
    "Punk": "punk",
    "Hiphop, Rap": "hip-hop",
    "Reggae, Ska": "reggae",
    "Swing, Jazz": "jazz",
    "Rock n roll": "rock",
    "Alternative": "alt-rock",
    "Latino": "latino",
    "Techno, Trance": "edm",
    "Opera": "classical",
}


def build_login_url(redirect_uri: str) -> Tuple[str, str]:
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
    if not client_id:
        raise RuntimeError("Missing SPOTIFY_CLIENT_ID")

    state = secrets.token_urlsafe(24)
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SPOTIFY_SCOPE,
        "state": state,
        "show_dialog": "true",
    }
    req = requests.Request("GET", SPOTIFY_AUTH_URL, params=params).prepare()
    if not req.url:
        raise RuntimeError("Failed to build Spotify auth URL")
    return req.url, state


def exchange_code_for_token(code: str, redirect_uri: str) -> Dict:
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise RuntimeError("Missing SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET")

    basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    response = requests.post(SPOTIFY_TOKEN_URL, headers=headers, data=data, timeout=20)
    response.raise_for_status()
    return response.json()


def refresh_access_token(refresh_token: str) -> Dict:
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    response = requests.post(SPOTIFY_TOKEN_URL, headers=headers, data=data, timeout=20)
    response.raise_for_status()
    return response.json()


def spotify_get(access_token: str, path: str, params=None) -> Dict:
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(f"{SPOTIFY_API_BASE}{path}", headers=headers, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def spotify_post(access_token: str, path: str, payload: Dict) -> Dict:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    response = requests.post(f"{SPOTIFY_API_BASE}{path}", headers=headers, json=payload, timeout=20)
    response.raise_for_status()
    return response.json() if response.text else {}


def get_user_profile(access_token: str) -> Dict:
    return spotify_get(access_token, "/me")


def get_user_top_tracks(access_token: str, limit: int = 5) -> List[str]:
    data = spotify_get(access_token, "/me/top/tracks", params={"limit": limit, "time_range": "medium_term"})
    return [item["id"] for item in data.get("items", []) if item.get("id")]


def get_user_top_artists(access_token: str, limit: int = 5) -> List[str]:
    data = spotify_get(access_token, "/me/top/artists", params={"limit": limit, "time_range": "medium_term"})
    return [item["id"] for item in data.get("items", []) if item.get("id")]


def get_available_seed_genres(access_token: str) -> List[str]:
    try:
        data = spotify_get(access_token, "/recommendations/available-genre-seeds")
        return data.get("genres", [])
    except requests.HTTPError as exc:
        # Spotify marks this endpoint as deprecated and some apps receive 404.
        # Fall back to the known mapped seed genres so playlist flow can continue.
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 404:
            return sorted(set(MODEL_TO_SPOTIFY_GENRE.values()))
        raise


def map_model_genres_to_spotify(model_genres: List[str], available: List[str], max_count: int = 5) -> List[str]:
    seeds: List[str] = []
    for genre in model_genres:
        candidate = MODEL_TO_SPOTIFY_GENRE.get(genre)
        if candidate and candidate in available and candidate not in seeds:
            seeds.append(candidate)
        if len(seeds) >= max_count:
            break
    return seeds


def get_recommended_tracks(
    access_token: str,
    seed_genres: List[str],
    seed_tracks: List[str],
    seed_artists: List[str],
    limit: int = 30,
) -> List[Dict]:
    # Spotify allows up to 5 combined seeds among genres+tracks+artists.
    seeds_total = len(seed_genres) + len(seed_tracks) + len(seed_artists)
    if seeds_total == 0:
        return []

    params: Dict[str, str | int] = {
        "limit": limit,
    }
    if seed_genres:
        params["seed_genres"] = ",".join(seed_genres)
    if seed_tracks:
        params["seed_tracks"] = ",".join(seed_tracks)
    if seed_artists:
        params["seed_artists"] = ",".join(seed_artists)

    data = spotify_get(access_token, "/recommendations", params=params)
    return data.get("tracks", [])


def filter_recommendations_for_discovery(
    tracks: List[Dict],
    exclude_track_ids: List[str],
    exclude_artist_ids: List[str],
    limit: int,
) -> List[Dict]:
    excluded_tracks = set(exclude_track_ids)
    excluded_artists = set(exclude_artist_ids)

    filtered: List[Dict] = []
    fallback_without_artist_filter: List[Dict] = []
    seen_ids = set()

    for track in tracks:
        track_id = track.get("id")
        if not track_id or track_id in seen_ids:
            continue

        artist_ids = {
            artist.get("id")
            for artist in track.get("artists", [])
            if isinstance(artist, dict) and artist.get("id")
        }

        if track_id not in excluded_tracks:
            fallback_without_artist_filter.append(track)

        if track_id in excluded_tracks:
            continue
        if artist_ids and artist_ids.intersection(excluded_artists):
            continue

        filtered.append(track)
        seen_ids.add(track_id)
        if len(filtered) >= limit:
            return filtered

    if len(filtered) >= limit:
        return filtered

    for track in fallback_without_artist_filter:
        track_id = track.get("id")
        if not track_id or track_id in seen_ids:
            continue
        filtered.append(track)
        seen_ids.add(track_id)
        if len(filtered) >= limit:
            break

    return filtered


def search_tracks_by_genres(access_token: str, seed_genres: List[str], limit: int = 30) -> List[Dict]:
    if not seed_genres or limit <= 0:
        return []

    results: List[Dict] = []
    seen = set()
    per_genre = min(50, max(10, (limit // max(len(seed_genres), 1)) + 10))

    for genre in seed_genres:
        data = spotify_get(
            access_token,
            "/search",
            params={
                "q": f'genre:"{genre}"',
                "type": "track",
                "limit": per_genre,
                "market": "from_token",
            },
        )

        for item in data.get("tracks", {}).get("items", []):
            track_id = item.get("id")
            if not track_id or track_id in seen:
                continue
            results.append(item)
            seen.add(track_id)
            if len(results) >= limit:
                return results

    return results


def create_playlist(access_token: str, user_id: str, name: str, description: str, is_public: bool) -> Dict:
    payload = {
        "name": name,
        "description": description,
        "public": is_public,
    }
    return spotify_post(access_token, f"/users/{user_id}/playlists", payload)


def add_tracks_to_playlist(access_token: str, playlist_id: str, track_uris: List[str]) -> Dict:
    payload = {
        "uris": track_uris,
    }
    return spotify_post(access_token, f"/playlists/{playlist_id}/tracks", payload)
