import logging
from typing import List

import streamlit as st

from config.config import settings
from models.llm import generate_response
from utils.query_router import route_query
from utils.rag_pipeline import build_vector_store_from_uploads, compose_rag_prompt, retrieve_rag_context
from utils.research_agent import run_research_agent
from utils.web_search import perform_web_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AutoResearch Copilot", page_icon="AI", layout="wide")
st.title("AutoResearch Copilot - Autonomous AI Research Assistant")


def _format_sources(sources: List[str]) -> str:
    clean = sorted(set([src for src in sources if src]))
    if not clean:
        return ""
    formatted = "\n".join([f"- {src}" for src in clean])
    return f"Sources:\n{formatted}"


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "indexed_docs_count" not in st.session_state:
        st.session_state.indexed_docs_count = 0
    if "indexed_chunks_count" not in st.session_state:
        st.session_state.indexed_chunks_count = 0


_init_state()

with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents"):
        try:
            if not uploaded_files:
                st.warning("Upload at least one PDF or text file.")
            else:
                store, docs_count, chunk_count = build_vector_store_from_uploads(uploaded_files)
                st.session_state.vector_store = store
                st.session_state.indexed_docs_count = docs_count
                st.session_state.indexed_chunks_count = chunk_count
                st.success(f"Indexed {docs_count} document(s) into {chunk_count} chunks.")
        except Exception as exc:
            logger.exception("Document processing failed: %s", exc)
            st.error("Document processing failed. Check logs and file format.")

    response_mode = st.radio("Response Mode", ["concise", "detailed"], index=0)
    system_mode = st.radio("System Mode", ["chat", "research", "deep research"], index=0)

    provider_options = []
    if settings.groq_api_key:
        provider_options.append("groq")
    if settings.openai_api_key:
        provider_options.append("openai")
    if not provider_options:
        provider_options = ["groq", "openai"]

    preferred_provider = settings.llm_provider if settings.llm_provider in provider_options else "groq"
    default_provider_idx = provider_options.index(preferred_provider) if preferred_provider in provider_options else 0
    provider = st.selectbox("LLM Provider", provider_options, index=default_provider_idx)

    if not settings.groq_api_key and not settings.openai_api_key:
        st.warning("No LLM API key configured. Add GROQ_API_KEY or OPENAI_API_KEY in Streamlit secrets.")

    st.caption(
        f"Indexed documents: {st.session_state.indexed_docs_count} | "
        f"Indexed chunks: {st.session_state.indexed_chunks_count}"
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question about your docs or the web")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                has_documents = st.session_state.vector_store is not None
                selected_route = route_query(
                    query=user_query,
                    has_documents=has_documents,
                    research_mode=system_mode == "research",
                    deep_research_mode=system_mode == "deep research",
                )

                answer = ""
                sources: List[str] = []
                extra_sections: List[str] = []

                if selected_route in {"research", "deep_research"}:
                    research_output = run_research_agent(
                        query=user_query,
                        vector_store=st.session_state.vector_store,
                        response_mode=response_mode,
                        deep_research=selected_route == "deep_research",
                        provider=provider,
                    )
                    answer = research_output.get("answer", "")
                    sources.extend(research_output.get("sources", []))
                    plan = research_output.get("plan", [])
                    if plan:
                        plan_block = "\n".join([f"{idx + 1}. {step}" for idx, step in enumerate(plan)])
                        extra_sections.append(f"Research Plan:\n{plan_block}")

                elif selected_route == "hybrid":
                    rag_context, rag_sources, _ = retrieve_rag_context(
                        user_query, st.session_state.vector_store
                    )
                    web_context, web_sources, _ = perform_web_search(user_query)
                    hybrid_prompt = (
                        "Use both local document evidence and web evidence to answer the query.\n\n"
                        f"Question:\n{user_query}\n\n"
                        f"Document evidence:\n{rag_context or 'No document evidence found.'}\n\n"
                        f"Web evidence:\n{web_context or 'No web evidence found.'}"
                    )
                    answer = generate_response(hybrid_prompt, mode=response_mode, provider=provider)
                    sources.extend(rag_sources)
                    sources.extend(web_sources)

                elif selected_route == "rag":
                    rag_context, rag_sources, _ = retrieve_rag_context(
                        user_query, st.session_state.vector_store
                    )
                    rag_prompt = compose_rag_prompt(user_query, rag_context, mode=response_mode)
                    answer = generate_response(rag_prompt, mode=response_mode, provider=provider)
                    sources.extend(rag_sources)

                elif selected_route == "web":
                    web_context, web_sources, _ = perform_web_search(user_query)
                    web_prompt = (
                        "Answer using this web research summary and mention uncertainty where needed.\n\n"
                        f"Question:\n{user_query}\n\n"
                        f"Web summary:\n{web_context}"
                    )
                    answer = generate_response(web_prompt, mode=response_mode, provider=provider)
                    sources.extend(web_sources)

                else:
                    answer = generate_response(user_query, mode=response_mode, provider=provider)

                source_block = _format_sources(sources)
                full_response_parts = [answer]
                if extra_sections:
                    full_response_parts.extend(extra_sections)
                if source_block:
                    full_response_parts.append(source_block)

                final_response = "\n\n".join([part for part in full_response_parts if part])
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
            except Exception as exc:
                logger.exception("Failed to handle user query: %s", exc)
                error_message = f"Request failed: {exc}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
