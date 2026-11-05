"""LangGraph orchestration: route -> RAG, tools, or general conversation."""
from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.hybrid_retriever import HybridRetriever
from src.tools import TOOLS


class RouteDecision(BaseModel):
    route: Literal["rag", "tools", "chat"] = Field(
        description="Choose rag for company knowledge, tools for calculations/weather, chat for general conversation."
    )


class AssistantState(MessagesState):
    route: str
    question: str
    search_query: str
    retrieved_context: str


def _last_user_text(messages):
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            return str(message.content)
    return ""


def build_graph():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY
    )
    router_llm = llm.with_structured_output(RouteDecision)
    tool_llm = llm.bind_tools(TOOLS)
    retriever = HybridRetriever()

    def route_request(state: AssistantState):
        history = state["messages"][-8:]
        router_prompt = [
            SystemMessage(content=(
                "Classify the latest user request. Return route='rag' for questions "
                "about company policies, internal IT support, engineering handbook, "
                "or facts likely to be in the indexed knowledge base. Return route='tools' "
                "for arithmetic/calculation or weather requests. Return route='chat' "
                "for greetings and general conversation. Use prior messages to resolve "
                "follow-up references such as 'what about sick leave?'."
            ))
        ] + history
        decision = router_llm.invoke(router_prompt)
        return {
            "route": decision.route,
            "question": _last_user_text(state["messages"])
        }

    def rewrite_query(state: AssistantState):
        # Turn follow-ups such as "what about the second one?" into standalone
        # search queries while retaining the original conversation messages.
        rewrite_prompt = [
            SystemMessage(content=(
                "Rewrite the latest user question as a standalone search query "
                "using conversation history only to resolve references. Preserve "
                "important names, identifiers and constraints. Return only the "
                "rewritten query, with no explanation. If the question is already "
                "standalone, return it unchanged."
            ))
        ] + state["messages"][-8:]
        rewritten = llm.invoke(rewrite_prompt)
        return {"search_query": str(rewritten.content).strip()}

    def retrieve_context(state: AssistantState):
        query = state.get("search_query") or state["question"]
        docs = retriever.retrieve(query)
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown source")
            page = doc.metadata.get("page")
            page_text = f", page {page + 1}" if isinstance(page, int) else ""
            formatted.append(
                f"[Source: {source}{page_text}]\n{doc.page_content}"
            )
        return {"retrieved_context": "\n\n---\n\n".join(formatted)}

    def compress_context(state: AssistantState):
        # LLM-based extractive-style evidence selection. Keep source labels.
        prompt = [
            SystemMessage(content=(
                "You are a retrieval-context compressor. Keep only statements that "
                "help answer the question. Preserve exact facts, numbers, conditions, "
                "and [Source: ...] labels. Do not add facts or answer the question. "
                "If nothing is relevant, return: NO_RELEVANT_CONTEXT."
            )),
            HumanMessage(content=(
                f"Question:\n{state['question']}\n\n"
                f"Retrieved context:\n{state.get('retrieved_context', '')}"
            ))
        ]
        compressed = llm.invoke(prompt)
        return {"retrieved_context": str(compressed.content)}

    def answer_from_context(state: AssistantState):
        system = SystemMessage(content=(
            "You are an enterprise knowledge assistant. Answer using only the "
            "retrieved context below. If the context does not contain the answer, "
            "say you could not find it in the available company documents. "
            "Do not invent policy details. Cite source filenames in square brackets. "
            f"\n\nRETRIEVED CONTEXT:\n{state.get('retrieved_context', '')}"
        ))
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    def tool_agent(state: AssistantState):
        response = tool_llm.invoke([
            SystemMessage(content=(
                "You are a helpful assistant. Use the available tools when needed. "
                "The weather tool returns mock demo data; clearly disclose that it "
                "is not live weather. Never claim a tool ran unless a tool result exists."
            ))
        ] + state["messages"])
        return {"messages": [response]}

    def general_chat(state: AssistantState):
        response = llm.invoke([
            SystemMessage(content="You are a helpful, concise assistant.")
        ] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AssistantState)
    builder.add_node("router", route_request)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve", retrieve_context)
    builder.add_node("compress", compress_context)
    builder.add_node("answer", answer_from_context)
    builder.add_node("tool_agent", tool_agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("chat", general_chat)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "rag": "rewrite_query",
            "tools": "tool_agent",
            "chat": "chat"
        }
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "compress")
    builder.add_edge("compress", "answer")
    builder.add_edge("answer", END)
    builder.add_edge("chat", END)

    # Tool-calling loop: model -> ToolNode -> model until no tool call remains.
    builder.add_conditional_edges(
        "tool_agent",
        tools_condition,
        {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "tool_agent")

    # InMemorySaver is for local learning. Use a DB-backed saver in production.
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
