
import logging

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage

from api.dependencies import get_graph
from api.schemas import QueryRequest, QueryResponse


router = APIRouter()

logger = logging.getLogger("hybrid_rag.api.query")


@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"]
)
def execute_query(
    payload: QueryRequest,
    graph=Depends(get_graph)
):
    try:
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=payload.question)
                ]
            },
            config={
                "configurable": {
                    "thread_id": payload.thread_id
                }
            }
        )

        answer = result["messages"][-1].content

        if not isinstance(answer, str):
            answer = str(answer)

        logger.info(
            "Query completed successfully for thread_id=%s",
            payload.thread_id
        )

        return QueryResponse(
            question=payload.question,
            answer=answer,
            thread_id=payload.thread_id
        )

    except Exception:
        logger.exception(
            "Query execution failed for thread_id=%s",
            payload.thread_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred while "
                "processing the query"
            )
        )