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

st.set_page_config(
    page_title="NeuroResearch",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Visual theme ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Global ── */
.stApp {
    background: radial-gradient(ellipse 120% 80% at 20% 0%, #0f0d1e 0%, #09090f 55%, #060609 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #dde1f0;
}
.block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }
header[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0c0b1d 0%, #090910 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.12) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.8rem !important; }

.sidebar-brand {
    text-align: center;
    padding: 0 0.5rem 1.4rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.12);
    margin-bottom: 0.5rem;
}
.sidebar-brand-name {
    font-family: 'Cinzel', serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-brand-tag {
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3a3a5c;
    margin-top: 0.25rem;
}

.section-label {
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3d3d60;
    padding: 1.1rem 0 0.5rem;
    margin: 0;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 14px rgba(79, 70, 229, 0.35) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(79, 70, 229, 0.5) !important;
    background: linear-gradient(135deg, #5b52f0 0%, #8b46ff 100%) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs (text, password) ── */
.stTextInput > label, .stFileUploader > label {
    color: #4a4a72 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
.stTextInput input {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(99, 102, 241, 0.18) !important;
    border-radius: 8px !important;
    color: #c7d2fe !important;
    font-size: 0.82rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1) !important;
}
.stTextInput input::placeholder { color: #2d2d4a !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(99, 102, 241, 0.03) !important;
    border: 1px dashed rgba(99, 102, 241, 0.22) !important;
    border-radius: 10px !important;
    padding: 0.3rem !important;
}

/* ── Radio ── */
.stRadio > label {
    color: #4a4a72 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
.stRadio [data-testid="stMarkdownContainer"] p { color: #8892b0 !important; font-size: 0.82rem !important; }
[data-testid="stRadio"] label span { color: #8892b0 !important; }

/* ── Selectbox ── */
.stSelectbox > label {
    color: #4a4a72 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99, 102, 241, 0.18) !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
    font-size: 0.85rem !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.055) !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.75rem !important;
    transition: border-color 0.2s;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(99, 102, 241, 0.2) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(99, 102, 241, 0.22) !important;
    border-radius: 12px !important;
    color: #dde1f0 !important;
    font-size: 0.88rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(99, 102, 241, 0.55) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.08) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #2d2d4a !important; }

/* ── Alerts & status ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    border: none !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── Caption / stats ── */
.stCaption { color: #2d2d50 !important; font-size: 0.68rem !important; letter-spacing: 0.04em !important; }

/* ── Divider ── */
hr { border-color: rgba(99, 102, 241, 0.1) !important; margin: 0.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a42; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #4f46e5; }
</style>
""", unsafe_allow_html=True)


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
    if "runtime_openai_key" not in st.session_state:
        st.session_state.runtime_openai_key = ""
    if "runtime_groq_key" not in st.session_state:
        st.session_state.runtime_groq_key = ""
    if "runtime_tavily_key" not in st.session_state:
        st.session_state.runtime_tavily_key = ""


_init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-name">NeuroResearch</div>
        <div class="sidebar-brand-tag">Autonomous AI Research Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # Documents section
    st.markdown('<p class="section-label">Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Index Documents"):
        try:
            if not uploaded_files:
                st.warning("Upload at least one file first.")
            else:
                store, docs_count, chunk_count = build_vector_store_from_uploads(uploaded_files)
                st.session_state.vector_store = store
                st.session_state.indexed_docs_count = docs_count
                st.session_state.indexed_chunks_count = chunk_count
                st.success(f"{docs_count} document(s) indexed into {chunk_count} chunks.")
        except Exception as exc:
            logger.exception("Document processing failed: %s", exc)
            st.error("Document processing failed. Check file format.")

    if st.session_state.indexed_docs_count > 0:
        st.caption(
            f"{st.session_state.indexed_docs_count} doc(s)  |  "
            f"{st.session_state.indexed_chunks_count} chunks indexed"
        )

    # Mode section
    st.markdown('<p class="section-label">Mode</p>', unsafe_allow_html=True)
    response_mode = st.radio("Response depth", ["concise", "detailed"], index=0)
    system_mode = st.radio("Research mode", ["chat", "research", "deep research"], index=0)

    # Provider section
    st.markdown('<p class="section-label">Provider</p>', unsafe_allow_html=True)

    runtime_provider_keys = {
        "openai": st.session_state.runtime_openai_key.strip(),
        "groq": st.session_state.runtime_groq_key.strip(),
    }
    runtime_provider_keys = {k: v for k, v in runtime_provider_keys.items() if v}

    provider_options = []
    if settings.groq_api_key or runtime_provider_keys.get("groq"):
        provider_options.append("groq")
    if settings.openai_api_key or runtime_provider_keys.get("openai"):
        provider_options.append("openai")
    if not provider_options:
        provider_options = ["groq", "openai"]

    preferred_provider = settings.llm_provider if settings.llm_provider in provider_options else "groq"
    default_provider_idx = provider_options.index(preferred_provider) if preferred_provider in provider_options else 0
    provider = st.selectbox("LLM provider", provider_options, index=default_provider_idx)

    has_any_llm_key = bool(settings.groq_api_key or settings.openai_api_key or runtime_provider_keys)
    if not has_any_llm_key:
        st.warning("No LLM key found. Add a key below or configure Streamlit secrets.")

    # API Keys section
    st.markdown('<p class="section-label">API Keys</p>', unsafe_allow_html=True)
    st.session_state.runtime_openai_key = st.text_input(
        "OpenAI key",
        value=st.session_state.runtime_openai_key,
        type="password",
        placeholder="sk-...",
    )
    st.session_state.runtime_groq_key = st.text_input(
        "Groq key",
        value=st.session_state.runtime_groq_key,
        type="password",
        placeholder="gsk_...",
    )
    st.session_state.runtime_tavily_key = st.text_input(
        "Tavily key",
        value=st.session_state.runtime_tavily_key,
        type="password",
        placeholder="tvly-...",
    )

    runtime_tavily_key = st.session_state.runtime_tavily_key.strip()

# ── Main content ──────────────────────────────────────────────────────────────
_mode_labels = {
    "chat": "Chat",
    "research": "Research",
    "deep research": "Deep Research",
}
_depth_color = {"concise": "#4a4a72", "detailed": "#6366f1"}

st.markdown(f"""
<div style="text-align:center; padding: 2.2rem 0 1.6rem;">
    <div style="
        font-family: 'Cinzel', serif;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 45%, #67e8f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    ">NeuroResearch</div>
    <div style="
        font-size: 0.7rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: #333355;
        margin-top: 0.5rem;
        font-weight: 400;
    ">Autonomous AI Research Engine</div>
    <div style="
        display: inline-block;
        margin-top: 1rem;
        padding: 0.22rem 0.9rem;
        border-radius: 999px;
        border: 1px solid rgba(99,102,241,0.28);
        background: rgba(99,102,241,0.07);
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #818cf8;
    ">{_mode_labels.get(system_mode, system_mode)} &nbsp;&middot;&nbsp; {response_mode.capitalize()}</div>
    <div style="
        width: 60px;
        height: 1px;
        background: linear-gradient(90deg, transparent, #6366f1, transparent);
        margin: 1.2rem auto 0;
    "></div>
</div>
""", unsafe_allow_html=True)

# Welcome card when conversation is empty
if not st.session_state.messages:
    st.markdown("""
    <div style="
        max-width: 600px;
        margin: 1.5rem auto 2.5rem;
        padding: 2.2rem 2.4rem;
        background: rgba(255,255,255,0.018);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 48px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.03);
    ">
        <div style="
            font-size: 1rem;
            font-weight: 600;
            color: #c7d2fe;
            letter-spacing: 0.02em;
            margin-bottom: 0.7rem;
        ">What would you like to explore?</div>
        <div style="
            font-size: 0.82rem;
            color: #3d3d62;
            line-height: 1.75;
        ">
            Ask a question, upload your papers, or run a deep research workflow.
            I will route your request automatically based on the selected mode.
        </div>
        <div style="
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.65rem;
            margin-top: 1.6rem;
        ">
            <div style="padding:0.65rem 0.9rem; background:rgba(99,102,241,0.05); border:1px solid rgba(99,102,241,0.12); border-radius:9px; font-size:0.75rem; color:#64748b; text-align:left;">Document Q&amp;A from PDFs</div>
            <div style="padding:0.65rem 0.9rem; background:rgba(99,102,241,0.05); border:1px solid rgba(99,102,241,0.12); border-radius:9px; font-size:0.75rem; color:#64748b; text-align:left;">Real-time web search</div>
            <div style="padding:0.65rem 0.9rem; background:rgba(99,102,241,0.05); border:1px solid rgba(99,102,241,0.12); border-radius:9px; font-size:0.75rem; color:#64748b; text-align:left;">Multi-step research planning</div>
            <div style="padding:0.65rem 0.9rem; background:rgba(99,102,241,0.05); border:1px solid rgba(99,102,241,0.12); border-radius:9px; font-size:0.75rem; color:#64748b; text-align:left;">Deep synthesis with sources</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask anything — documents, web, or deep research")

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
                        provider_keys=runtime_provider_keys,
                        tavily_api_key=runtime_tavily_key or None,
                    )
                    answer = research_output.get("answer", "")
                    sources.extend(research_output.get("sources", []))
                    plan = research_output.get("plan", [])
                    research_errors = research_output.get("errors", [])
                    if research_errors:
                        st.warning("Research completed with partial fallbacks. Some sources may be unavailable.")
                    if plan:
                        plan_block = "\n".join([f"{idx + 1}. {step}" for idx, step in enumerate(plan)])
                        extra_sections.append(f"Research Plan:\n{plan_block}")

                elif selected_route == "hybrid":
                    rag_context, rag_sources, _ = retrieve_rag_context(
                        user_query, st.session_state.vector_store
                    )
                    web_context, web_sources, _ = perform_web_search(
                        user_query,
                        api_key=runtime_tavily_key or None,
                    )
                    hybrid_prompt = (
                        "Use both local document evidence and web evidence to answer the query.\n\n"
                        f"Question:\n{user_query}\n\n"
                        f"Document evidence:\n{rag_context or 'No document evidence found.'}\n\n"
                        f"Web evidence:\n{web_context or 'No web evidence found.'}"
                    )
                    answer = generate_response(
                        hybrid_prompt,
                        mode=response_mode,
                        provider=provider,
                        provider_keys=runtime_provider_keys,
                    )
                    sources.extend(rag_sources)
                    sources.extend(web_sources)

                elif selected_route == "rag":
                    rag_context, rag_sources, _ = retrieve_rag_context(
                        user_query, st.session_state.vector_store
                    )
                    rag_prompt = compose_rag_prompt(user_query, rag_context, mode=response_mode)
                    answer = generate_response(
                        rag_prompt,
                        mode=response_mode,
                        provider=provider,
                        provider_keys=runtime_provider_keys,
                    )
                    sources.extend(rag_sources)

                elif selected_route == "web":
                    web_context, web_sources, _ = perform_web_search(
                        user_query,
                        api_key=runtime_tavily_key or None,
                    )
                    web_prompt = (
                        "Answer using this web research summary and mention uncertainty where needed.\n\n"
                        f"Question:\n{user_query}\n\n"
                        f"Web summary:\n{web_context}"
                    )
                    answer = generate_response(
                        web_prompt,
                        mode=response_mode,
                        provider=provider,
                        provider_keys=runtime_provider_keys,
                    )
                    sources.extend(web_sources)

                else:
                    answer = generate_response(
                        user_query,
                        mode=response_mode,
                        provider=provider,
                        provider_keys=runtime_provider_keys,
                    )

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
