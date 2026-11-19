from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"]
)
def health_check():

    return HealthResponse(
        status="healthy",
        service="hybrid-rag-api",
        version="1.0.0"
    )
