from typing import Annotated, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    answers: List[Annotated[int, Field(ge=1, le=5)]] = Field(..., min_length=10, max_length=10)
    top_n: Annotated[int, Field(ge=1, le=10)] = 5


class GenreItem(BaseModel):
    genre: str
    probability: float
    predicted: Optional[int] = None


class PredictResponse(BaseModel):
    trait_scores: dict
    genres: List[GenreItem]


class PlaylistRequest(BaseModel):
    answers: List[Annotated[int, Field(ge=1, le=5)]] = Field(..., min_length=10, max_length=10)
    top_n: Annotated[int, Field(ge=1, le=10)] = 5
    playlist_name: str = "PulsePersona Mix"
    is_public: bool = False
    limit: Annotated[int, Field(ge=10, le=100)] = 30
