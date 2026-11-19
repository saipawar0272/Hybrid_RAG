
from typing import Any

from fastapi import HTTPException, Request


def get_graph(request: Request) -> Any:
    """
    Retrieve the initialized LangGraph instance from application state.

    Raises:
        HTTPException: If the graph is unavailable.
    """
    graph = getattr(request.app.state, "graph", None)

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="RAG workflow is not available"
        )

    return graph