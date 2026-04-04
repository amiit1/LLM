"""Main FastAPI application entry point.

This file wires together middleware and route modules.
Keeping app startup logic in one place makes the project easier to scale.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.nlp_routes import router as nlp_router


app = FastAPI(
    title="NLP Preprocessing API",
    description="A beginner-friendly API for common NLP preprocessing tasks.",
    version="1.0.0",
)

# CORS is enabled so the frontend can call the API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check() -> dict:
    """Simple health check endpoint to confirm server availability."""
    return {"message": "NLP Preprocessing API is running."}


app.include_router(nlp_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
