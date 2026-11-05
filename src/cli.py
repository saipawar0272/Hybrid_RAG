from langchain_core.messages import HumanMessage
from src.graph import build_graph


def main():
    graph = build_graph()
    thread_id = input("Thread ID (e.g. demo-1): ").strip() or "demo-1"
    print("\nEnterprise AI Knowledge Assistant. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        print("\nAssistant:", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()
