"""API routes for word embedding exploration endpoints."""

from fastapi import APIRouter, HTTPException, Query

from models.schemas import (
    CorpusEmbeddingRequest,
    CorpusEmbeddingResponse,
    CorpusResponse,
    EmbeddingInfoResponse,
    EmbeddingProjectionResponse,
)
from services import embedding_service


router = APIRouter(prefix="/api/embeddings", tags=["Embeddings"])


def _handle_service_errors(error: Exception) -> None:
    """Convert embedding service errors to client-friendly HTTP responses."""
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=str(error)) from error
    if isinstance(error, RuntimeError):
        raise HTTPException(status_code=500, detail=str(error)) from error
    raise error


@router.get("/info", response_model=EmbeddingInfoResponse)
def embedding_info() -> EmbeddingInfoResponse:
    """Get summary details about the active embedding technique."""
    try:
        return EmbeddingInfoResponse(**embedding_service.get_embedding_info())
    except Exception as error:  # pragma: no cover - thin API error adapter
        _handle_service_errors(error)


@router.get("/corpus", response_model=CorpusResponse)
def corpus(limit: int = Query(default=80, ge=5, le=300)) -> CorpusResponse:
    """Return corpus entries from the custom corpus."""
    try:
        return CorpusResponse(documents=embedding_service.get_corpus(limit=limit))
    except Exception as error:  # pragma: no cover
        _handle_service_errors(error)


@router.post("/vector", response_model=CorpusEmbeddingResponse)
def vector(payload: CorpusEmbeddingRequest) -> CorpusEmbeddingResponse:
    """Return embedding values and nearest corpus neighbors for input text."""
    try:
        return CorpusEmbeddingResponse(
            **embedding_service.get_corpus_embedding(payload.text, top_k=payload.top_k)
        )
    except Exception as error:  # pragma: no cover
        _handle_service_errors(error)


@router.get("/visualize", response_model=EmbeddingProjectionResponse)
def visualize(
    method: str = Query(default="pca", pattern="^(pca|tsne)$"),
    limit: int = Query(default=14, ge=5, le=120),
) -> EmbeddingProjectionResponse:
    """Project corpus document embeddings into 2D for plotting."""
    try:
        return EmbeddingProjectionResponse(
            **embedding_service.project_embeddings(method=method, limit=limit)
        )
    except Exception as error:  # pragma: no cover
        _handle_service_errors(error)
