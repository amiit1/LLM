"""API routes for NLP preprocessing actions.

Routes are intentionally thin: they validate input/output and delegate work to
service functions. This keeps business logic centralized in services.
"""

from fastapi import APIRouter, HTTPException

from models.schemas import (
    AnalyzeResponse,
    EntitiesResponse,
    LemmasResponse,
    PosTagsResponse,
    StemsResponse,
    TextRequest,
    TokensResponse,
)
from services import nlp_service


router = APIRouter(prefix="/api/nlp", tags=["NLP"])


def _handle_service_errors(error: Exception) -> None:
    """Convert service-level runtime errors into API-friendly HTTP errors."""
    if isinstance(error, RuntimeError):
        raise HTTPException(status_code=500, detail=str(error)) from error
    raise error


@router.post("/tokenize", response_model=TokensResponse)
def tokenize_text(payload: TextRequest) -> TokensResponse:
    """Tokenize input text into a list of tokens."""
    try:
        return TokensResponse(tokens=nlp_service.tokenize(payload.text))
    except Exception as error:  # pragma: no cover - thin API error adapter
        _handle_service_errors(error)


@router.post("/lemmatize", response_model=LemmasResponse)
def lemmatize_text(payload: TextRequest) -> LemmasResponse:
    """Lemmatize input text into dictionary forms."""
    try:
        return LemmasResponse(lemmas=nlp_service.lemmatize(payload.text))
    except Exception as error:  # pragma: no cover
        _handle_service_errors(error)


@router.post("/stem", response_model=StemsResponse)
def stem_text(payload: TextRequest) -> StemsResponse:
    """Stem input text using Porter stemming."""
    try:
        return StemsResponse(stems=nlp_service.stem(payload.text))
    except Exception as error:  # pragma: no cover
        _handle_service_errors(error)


@router.post("/pos-tag", response_model=PosTagsResponse)
def pos_tag_text(payload: TextRequest) -> PosTagsResponse:
    """Assign POS tags to each token in the input."""
    try:
        return PosTagsResponse(pos_tags=nlp_service.pos_tag(payload.text))
    except Exception as error:  # pragma: no cover
        _handle_service_errors(error)


@router.post("/ner", response_model=EntitiesResponse)
def ner_text(payload: TextRequest) -> EntitiesResponse:
    """Extract named entities from the input text."""
    try:
        return EntitiesResponse(entities=nlp_service.ner(payload.text))
    except Exception as error:  # pragma: no cover
        _handle_service_errors(error)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(payload: TextRequest) -> AnalyzeResponse:
    """Run all NLP preprocessing tasks in one call."""
    try:
        return AnalyzeResponse(**nlp_service.analyze(payload.text))
    except Exception as error:  # pragma: no cover
        _handle_service_errors(error)
