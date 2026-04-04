"""Pydantic schemas for API requests and responses.

Keeping schemas in one place makes contracts explicit and maintainable.
"""

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """Request payload containing the input text to analyze."""

    text: str = Field(..., min_length=1, description="Input text for NLP processing")


class TokensResponse(BaseModel):
    """Response model for tokenization output."""

    tokens: list[str]


class LemmasResponse(BaseModel):
    """Response model for lemmatization output."""

    lemmas: list[str]


class StemsResponse(BaseModel):
    """Response model for stemming output."""

    stems: list[str]


class PosTagItem(BaseModel):
    """Single POS tag record for one token."""

    token: str
    pos: str
    tag: str
    description: str


class PosTagsResponse(BaseModel):
    """Response model for POS tagging output."""

    pos_tags: list[PosTagItem]


class EntityItem(BaseModel):
    """Single named entity record."""

    text: str
    label: str
    description: str


class EntitiesResponse(BaseModel):
    """Response model for named entity recognition output."""

    entities: list[EntityItem]


class AnalyzeResponse(BaseModel):
    """Combined response model for all NLP tasks."""

    tokens: list[str]
    lemmas: list[str]
    stems: list[str]
    pos_tags: list[PosTagItem]
    entities: list[EntityItem]
