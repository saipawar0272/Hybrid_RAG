
import pytest

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from api.main import create_app


class FakeGraph:
    """A lightweight substitute for the real LangGraph workflow."""

    def __init__(self, should_fail=False):
        self.calls = []
        self.should_fail = should_fail

    def invoke(self, state, config):
        self.calls.append({
            "state": state,
            "config": config
        })

        if self.should_fail:
            raise RuntimeError("Simulated graph execution failure")

        return {
            "messages": [
                AIMessage(content="Mock RAG answer")
            ]
        }


@pytest.fixture
def fake_graph():
    return FakeGraph()


@pytest.fixture
def api_client(fake_graph):
    app = create_app(
        graph_factory=lambda: fake_graph
    )

    with TestClient(app) as client:
        yield client


def test_health_endpoint(api_client):
    response = api_client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "hybrid-rag-api",
        "version": "1.0.0"
    }


def test_valid_query_returns_answer(api_client):
    payload = {
        "question": "How many annual leave days are available?",
        "thread_id": "test-thread-001"
    }

    response = api_client.post(
        "/api/v1/query",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == payload["question"]
    assert data["answer"] == "Mock RAG answer"
    assert data["thread_id"] == payload["thread_id"]


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   "
    ]
)
def test_invalid_question_returns_422(api_client, question):
    response = api_client.post(
        "/api/v1/query",
        json={
            "question": question,
            "thread_id": "invalid-input-test"
        }
    )

    assert response.status_code == 422


def test_missing_graph_returns_503():
    app = create_app(
        graph_factory=lambda: None
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query",
            json={
                "question": "Test question",
                "thread_id": "missing-graph-test"
            }
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "RAG workflow is not available"
    )


def test_graph_execution_failure_returns_500():
    failing_graph = FakeGraph(should_fail=True)

    app = create_app(
        graph_factory=lambda: failing_graph
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query",
            json={
                "question": "Test graph failure",
                "thread_id": "failure-test"
            }
        )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "An internal error occurred while processing the query"
    )


def test_thread_id_is_forwarded_to_graph(api_client, fake_graph):
    thread_id = "conversation-101"

    response = api_client.post(
        "/api/v1/query",
        json={
            "question": "What is the leave policy?",
            "thread_id": thread_id
        }
    )

    assert response.status_code == 200
    assert len(fake_graph.calls) == 1

    actual_thread_id = (
        fake_graph.calls[0]
        ["config"]["configurable"]["thread_id"]
    )

    assert actual_thread_id == thread_id


def test_different_thread_ids_are_forwarded_separately(
    api_client,
    fake_graph
):
    thread_ids = [
        "conversation-101",
        "conversation-202"
    ]

    for thread_id in thread_ids:
        response = api_client.post(
            "/api/v1/query",
            json={
                "question": "Test conversation",
                "thread_id": thread_id
            }
        )

        assert response.status_code == 200

    actual_thread_ids = [
        call["config"]["configurable"]["thread_id"]
        for call in fake_graph.calls
    ]

    assert actual_thread_ids == thread_ids
    assert len(set(actual_thread_ids)) == 2



def test_root_endpoint(api_client):
    response = api_client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to Hybrid RAG API",
        "status": "running"
    }


def test_query_converts_non_string_answer_to_string():
    class NonStringAnswerGraph:
        def invoke(self, state, config):
            return {
                "messages": [
                    AIMessage(content=[
                        {"type": "text", "text": "Mock answer"}
                    ])
                ]
            }

    graph = NonStringAnswerGraph()

    app = create_app(
        graph_factory=lambda: graph
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query",
            json={
                "question": "Test non-string answer",
                "thread_id": "non-string-test"
            }
        )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "[{'type': 'text', 'text': 'Mock answer'}]"
    )