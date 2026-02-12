"""
Streamlit UI for the Chitown Custom Choppers RAG chatbot.
Orchestration logic lives in chat_orchestrator; this module only handles UI and caching.
"""
import sys
from pathlib import Path

# Bootstrap path so src can be imported
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from src import config

import streamlit as st

from src.RAG_chatbot.chat_orchestrator import (
    setup_rag_chain,
    load_knowledge_graph,
    build_chat_router_chain,
    build_graph_rag_chain,
    generate_chat_response_stream,
    configure_langsmith,
    get_langsmith_debug_info,
    test_langsmith_connection,
)


@st.cache_resource(show_spinner="Loading vector index and LLM…")
def _cached_setup_rag():
    return setup_rag_chain()


@st.cache_resource(show_spinner="Loading knowledge graph…")
def _cached_load_graph():
    return load_knowledge_graph()


def main():
    st.set_page_config(page_title="Chitown Custom Choppers RAG Bot", page_icon="🛠️")
    st.title("🛠️ Chitown Custom Choppers – Shop Knowledge Assistant")

    # LangSmith debug panel (sidebar)
    with st.sidebar:
        with st.expander("🔍 LangSmith tracing", expanded=False):
            configure_langsmith()
            info = get_langsmith_debug_info()
            st.write("**Tracing:**", info["LANGCHAIN_TRACING_V2"])
            st.write("**API key set:**", info["LANGCHAIN_API_KEY set"])
            st.write("**Key prefix:**", info["LANGCHAIN_API_KEY prefix"])
            st.write("**Project:**", info["LANGCHAIN_PROJECT"])
            if not info["LANGCHAIN_API_KEY set"] or info["LANGCHAIN_TRACING_V2"] == "False":
                st.warning(
                    "Traces may not appear. Set LANGCHAIN_API_KEY in .env; tracing on/off is in src/config.py."
                )
            else:
                st.caption("Traces should appear at smith.langchain.com")
            st.divider()
            st.caption("Connection test (auth + API reachability)")
            if st.button("Test LangSmith connection", key="langsmith_test_btn"):
                with st.spinner("Testing…"):
                    result = test_langsmith_connection()
                if result["ok"]:
                    st.success(result["message"])
                    if result.get("detail"):
                        st.caption(result["detail"])
                else:
                    st.error(result["message"])
                    if result.get("detail"):
                        st.code(result["detail"][:500], language=None)

    # Initialize RAG and GraphRAG resources (cached)
    try:
        rag_chain, vectorstore = _cached_setup_rag()
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        st.stop()

    try:
        knowledge_graph = _cached_load_graph()
    except Exception as e:
        st.error(f"Error loading knowledge graph: {e}")
        st.stop()

    # GraphRAG chain: 3 runnables (build context -> optional vector context -> answer) for clear LangSmith traces
    graphrag_chain = build_graph_rag_chain(knowledge_graph, vectorstore)

    chat_chain = build_chat_router_chain(rag_chain, graphrag_chain)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    prompt_input = st.chat_input(
        "Ask about HR policies, returns, builds, store operations, org structure, or Q3 sales…"
    )
    if not prompt_input:
        return

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate and show assistant response (streaming when the chain supports it)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_answer = ""
        sources = []
        try:
            for chunk in generate_chat_response_stream(prompt_input, chat_chain):
                if isinstance(chunk, dict):
                    full_answer = chunk.get("answer") or full_answer
                    if chunk.get("sources") is not None:
                        sources = chunk["sources"]
                # Show accumulated text with a cursor while streaming
                message_placeholder.markdown(full_answer + "▌")
            message_placeholder.markdown(full_answer)
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return

        # Show provenance block for vector-RAG answers
        if sources:
            st.markdown("---")
            st.markdown("**Sources used:**")
            for s in sources:
                st.markdown(
                    f"- Source {s['id']}: `{s['file_name']}` "
                    f"(type: {s['document_type']}, page: {s['page']}, "
                    f"distance={s['distance']:.4f}, conf≈{s['confidence']:.3f})"
                )

    # Store assistant answer in history (just the text, not the sources block)
    st.session_state.messages.append({"role": "assistant", "content": full_answer})


if __name__ == "__main__":
    main()
