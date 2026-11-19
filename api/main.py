
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI

from api.routes import health, query
from src.graph import build_graph
from src.logging_config import get_logger


logger = get_logger("api")


def create_app(
    graph_factory: Callable[[], Any] | None = None
) -> FastAPI:
    """
    Create the FastAPI application.

    graph_factory allows tests to inject a fake graph
    instead of initializing the real RAG workflow.
    """

    factory = graph_factory or build_graph

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Initializing Hybrid RAG graph")

        app.state.graph = None

        try:
            app.state.graph = factory()

            logger.info(
                "Hybrid RAG graph initialized successfully"
            )

            yield

        finally:
            logger.info(
                "Hybrid RAG API application shutting down"
            )

            app.state.graph = None

    application = FastAPI(
        title="Hybrid RAG API",
        description=(
            "Production-oriented Hybrid Retrieval "
            "Augmented Generation API"
        ),
        version="1.0.0",
        lifespan=lifespan
    )

    application.include_router(
        health.router,
        prefix="/api/v1"
    )

    application.include_router(
        query.router,
        prefix="/api/v1"
    )

    @application.get("/")
    def root():
        return {
            "message": "Welcome to Hybrid RAG API",
            "status": "running"
        }

    return application


app = create_app()