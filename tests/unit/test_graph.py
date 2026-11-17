import pytest
from pydantic import ValidationError
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

import src.graph as graph_module



# --------------------------------------------------
# HELPER FUNCTION TESTS
# --------------------------------------------------

def test_last_user_text_returns_latest_human_message():
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi"),
        HumanMessage(content="What is the leave policy?"),
    ]

    result = graph_module._last_user_text(messages)

    assert result == "What is the leave policy?"


def test_last_user_text_returns_empty_string_without_human_message():
    messages = [
        AIMessage(content="Hello"),
    ]

    result = graph_module._last_user_text(messages)

    assert result == ""


# --------------------------------------------------
# ROUTE DECISION MODEL TESTS
# --------------------------------------------------

@pytest.mark.parametrize("route", ["rag", "tools", "chat"])
def test_route_decision_accepts_valid_routes(route):
    decision = graph_module.RouteDecision(route=route)

    assert decision.route == route


def test_route_decision_rejects_invalid_route():
    with pytest.raises(ValidationError):
        graph_module.RouteDecision(route="invalid")


# --------------------------------------------------
# GRAPH INITIALIZATION TESTS
# --------------------------------------------------

def test_build_graph_raises_error_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(graph_module, "GROQ_API_KEY", "")

    with pytest.raises(
        RuntimeError,
        match="GROQ_API_KEY is missing"
    ):
        graph_module.build_graph()


# --------------------------------------------------
# MOCK COMPONENTS FOR GRAPH EXECUTION
# --------------------------------------------------

class FakeRouterLLM:

    def __init__(self, route):
        self.route = route

    def invoke(self, messages):
        return graph_module.RouteDecision(route=self.route)


class FakeLLM:

    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, messages):
        if not self.responses:
            raise AssertionError("Unexpected LLM invocation")

        return AIMessage(content=self.responses.pop(0))

    def with_structured_output(self, schema):
        return FakeRouterLLM(self.route)

    def bind_tools(self, tools):
        return FakeToolLLM(self.tool_response)


class FakeToolLLM:

    def __init__(self, response):
        self.response = response

    def invoke(self, messages):
        return self.response


def setup_fake_graph(monkeypatch, route, responses=None, documents=None):
    """
    Build a graph using mocked LLM and retriever components.
    No real Groq API or embedding model is initialized.
    """

    fake_llm = FakeLLM(responses or [])

    fake_llm.route = route
    fake_llm.tool_response = AIMessage(
        content="Mock tool-agent response"
    )

    class FakeRetriever:

        def retrieve(self, query):
            return documents or []

    monkeypatch.setattr(
        graph_module,
        "GROQ_API_KEY",
        "test-api-key"
    )

    monkeypatch.setattr(
        graph_module,
        "ChatGroq",
        lambda **kwargs: fake_llm
    )

    monkeypatch.setattr(
        graph_module,
        "HybridRetriever",
        FakeRetriever
    )

    return graph_module.build_graph()


# --------------------------------------------------
# CHAT ROUTE TEST
# --------------------------------------------------

def test_graph_chat_route(monkeypatch):
    graph = setup_fake_graph(
        monkeypatch,
        route="chat",
        responses=["Hello! How can I help you?"]
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Hello")
            ]
        },
        config={
            "configurable": {
                "thread_id": "test-chat"
            }
        }
    )

    assert result["messages"][-1].content == (
        "Hello! How can I help you?"
    )


# --------------------------------------------------
# TOOLS ROUTE TEST
# --------------------------------------------------

def test_graph_tools_route_without_tool_call(monkeypatch):
    graph = setup_fake_graph(
        monkeypatch,
        route="tools"
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="What is 10 + 20?")
            ]
        },
        config={
            "configurable": {
                "thread_id": "test-tools"
            }
        }
    )

    assert result["messages"][-1].content == (
        "Mock tool-agent response"
    )


# --------------------------------------------------
# RAG ROUTE TEST
# --------------------------------------------------

def test_graph_rag_route(monkeypatch):
    documents = [
        Document(
            page_content="Employees receive 20 annual leave days.",
            metadata={
                "source": "company_policy.md",
                "page": 0
            }
        )
    ]

    graph = setup_fake_graph(
        monkeypatch,
        route="rag",
        responses=[
            "What is the annual leave allowance?",
            "[Source: company_policy.md, page 1] "
            "Employees receive 20 annual leave days.",
            "Employees receive 20 annual leave days. "
            "[company_policy.md]"
        ],
        documents=documents
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="How many annual leave days do employees receive?"
                )
            ]
        },
        config={
            "configurable": {
                "thread_id": "test-rag"
            }
        }
    )

    assert result["messages"][-1].content == (
        "Employees receive 20 annual leave days. "
        "[company_policy.md]"
    )




# --------------------------------------------------
# ACTUAL TOOL EXECUTION AND AGENT LOOP TEST
# --------------------------------------------------

class FakeToolCallingLLM:

    def __init__(self, responses):
        self.responses = list(responses)
        self.received_messages = []

    def invoke(self, messages):
        self.received_messages.append(messages)

        if not self.responses:
            raise AssertionError("Unexpected tool-agent invocation")

        return self.responses.pop(0)


class FakeChatModelForToolLoop:

    def __init__(self, tool_responses):
        self.tool_llm = FakeToolCallingLLM(tool_responses)

    def with_structured_output(self, schema):
        return FakeRouterLLM("tools")

    def bind_tools(self, tools):
        return self.tool_llm


def test_graph_executes_tool_and_loops_back_to_agent(monkeypatch):

    # Arrange: LLM requests the first calculation.
    first_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculate",
                "args": {"expression": "125 * 8"},
                "id": "call_1",
            }
        ],
    )

    # LLM receives the first tool result and requests another calculation.
    second_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculate",
                "args": {"expression": "1000 + 40"},
                "id": "call_2",
            }
        ],
    )

    # LLM receives the second tool result and produces a final answer.
    final_response = AIMessage(
        content="The final calculation result is 1040."
    )

    fake_llm = FakeChatModelForToolLoop(
        [
            first_tool_call,
            second_tool_call,
            final_response,
        ]
    )

    class FakeRetriever:

        def retrieve(self, query):
            return []

    monkeypatch.setattr(
        graph_module,
        "GROQ_API_KEY",
        "test-api-key"
    )

    monkeypatch.setattr(
        graph_module,
        "ChatGroq",
        lambda **kwargs: fake_llm
    )

    monkeypatch.setattr(
        graph_module,
        "HybridRetriever",
        FakeRetriever
    )

    graph = graph_module.build_graph()

    # Act: invoke the graph.
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Calculate 125 * 8, then add 40.")
            ]
        },
        config={
            "configurable": {
                "thread_id": "test-tool-loop"
            }
        }
    )

    # Assert 1: both actual tool executions occurred.
    tool_messages = [
        message
        for message in result["messages"]
        if message.type == "tool"
    ]

    assert len(tool_messages) == 2

    # Assert 2: calculator returned the correct results.
    assert [message.content for message in tool_messages] == [
        "1000",
        "1040",
    ]

    # Assert 3: agent was invoked three times.
    assert len(fake_llm.tool_llm.received_messages) == 3

    # Assert 4: final response came from the agent.
    assert result["messages"][-1].content == (
        "The final calculation result is 1040."
    )





# --------------------------------------------------
# CHECKPOINTING AND CONVERSATION MEMORY TESTS
# --------------------------------------------------

def test_graph_preserves_conversation_with_same_thread_id(monkeypatch):

    # Arrange
    graph = setup_fake_graph(
        monkeypatch,
        route="chat",
        responses=[
            "Hello! How can I help?",
            "You previously greeted me."
        ]
    )

    config = {
        "configurable": {
            "thread_id": "conversation-1"
        }
    }

    # Act: first conversation turn
    first_result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Hello")
            ]
        },
        config=config
    )

    # Act: second conversation turn using the same thread
    second_result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Do you remember my greeting?")
            ]
        },
        config=config
    )

    # Assert: previous conversation is retained
    message_contents = [
        message.content
        for message in second_result["messages"]
    ]

    assert "Hello" in message_contents
    assert "Hello! How can I help?" in message_contents
    assert "Do you remember my greeting?" in message_contents

    assert second_result["messages"][-1].content == (
        "You previously greeted me."
    )

    assert len(first_result["messages"]) == 2
    assert len(second_result["messages"]) == 4


def test_graph_isolates_different_thread_ids(monkeypatch):

    # Arrange
    graph = setup_fake_graph(
        monkeypatch,
        route="chat",
        responses=[
            "Response for user A",
            "Response for user B",
            "Follow-up response for user A"
        ]
    )

    config_a = {
        "configurable": {
            "thread_id": "user-A"
        }
    }

    config_b = {
        "configurable": {
            "thread_id": "user-B"
        }
    }

    # Act: user A starts a conversation
    graph.invoke(
        {
            "messages": [
                HumanMessage(content="My name is Rahul")
            ]
        },
        config=config_a
    )

    # Act: user B starts a separate conversation
    graph.invoke(
        {
            "messages": [
                HumanMessage(content="My name is Amit")
            ]
        },
        config=config_b
    )

    # Act: user A continues the original conversation
    result_a = graph.invoke(
        {
            "messages": [
                HumanMessage(content="What did I tell you?")
            ]
        },
        config=config_a
    )

    # Assert: user A's conversation contains its own history
    message_contents = [
        message.content
        for message in result_a["messages"]
    ]

    assert "My name is Rahul" in message_contents
    assert "What did I tell you?" in message_contents

    # Assert: user B's conversation is not present
    assert "My name is Amit" not in message_contents

    assert result_a["messages"][-1].content == (
        "Follow-up response for user A"
    )


