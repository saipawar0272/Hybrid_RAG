import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph import build_graph

st.set_page_config(page_title="Enterprise AI Assistant", page_icon="🤖")
st.title("Enterprise AI Knowledge Assistant")
st.caption("Hybrid RAG · Conversational RAG · LangGraph · Function calling")

@st.cache_resource
def get_graph():
    return build_graph()

try:
    graph = get_graph()
except Exception as exc:
    st.error(str(exc))
    st.info("Complete setup in README.md, build the index, and configure your .env file.")
    st.stop()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

with st.sidebar:
    st.subheader("Conversation")
    st.code(st.session_state.thread_id)
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.display_messages = []
        st.rerun()
    st.caption("A new thread has separate LangGraph checkpoint memory.")
    st.warning("Weather tool uses mock demo data, not live weather.")

for item in st.session_state.display_messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

question = st.chat_input("Ask about company policies, IT support, or try a tool...")
if question:
    st.session_state.display_messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=question)]},
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    }
                )
                answer = result["messages"][-1].content
                st.markdown(answer)
                st.session_state.display_messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as exc:
                st.error(f"Request failed: {exc}")
