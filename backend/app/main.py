from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import FRONTEND_ORIGIN
from .recommender import get_service
from .schemas import RecommendResponse

app = FastAPI(title="RecomPro API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_startup_error: str | None = None


@app.on_event("startup")
def _warm_up():
    """Build the catalog + embeddings once at startup so the first search isn't slow."""
    global _startup_error
    try:
        get_service()
    except Exception as exc:  # noqa: BLE001
        # Don't crash the whole app — surface the error on the endpoints instead,
        # so you can see it clearly (e.g. "reviews table is empty, run the loader script").
        _startup_error = str(exc)


@app.get("/api/health")
def health():
    if _startup_error:
        return {"status": "error", "detail": _startup_error}
    return {"status": "ok"}


@app.get("/api/recommend", response_model=RecommendResponse)
def recommend(
    q: str = Query(..., min_length=1, description="Natural-language product query, English/Bangla/Banglish"),
    top_k: int = Query(10, ge=1, le=50),
):
    if _startup_error:
        raise HTTPException(status_code=503, detail=_startup_error)
    service = get_service()
    try:
        return service.recommend(q, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
