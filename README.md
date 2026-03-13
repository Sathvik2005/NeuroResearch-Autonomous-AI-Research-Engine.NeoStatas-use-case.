# NeuroResearch - Autonomous AI Research Engine

A production-oriented Streamlit research assistant that combines local document retrieval and live web intelligence to answer questions with grounded context.

## Overview

NeuroResearch is designed as a hybrid knowledge assistant:

- Local knowledge: retrieval-augmented generation (RAG) from uploaded documents
- Global knowledge: live web search via Tavily
- LLM reasoning: OpenAI and Groq support through a unified interface
- Research workflows: chat mode, research mode, and deep research mode with structured outputs

The application is modular and aligned with a clean separation of concerns for maintainability and deployment.

## Core Capabilities

- Upload PDF, TXT, and Markdown documents
- Extract and chunk document text
- Generate embeddings and run similarity retrieval
- Build and query a vector store (FAISS when available, NumPy fallback on Windows)
- Perform live web search and summarize evidence
- Route queries across direct LLM, RAG, web, and hybrid paths
- Produce concise or detailed responses
- Generate structured research reports in research modes
- Display source attribution for transparency

## Architecture

### UI Layer

- Streamlit app for chat interactions, file uploads, mode controls, and source display

### Model Layer

- LLM wrapper with provider abstraction for OpenAI and Groq
- Embedding wrapper with sentence-transformers and deterministic fallback

### Retrieval and Search Layer

- Document loading and parsing
- Text chunking pipeline
- Vector store creation and semantic retrieval
- Web search integration and summary extraction
- Query routing and research orchestration

## Project Structure

```text
project/
  config/
    config.py
  models/
    llm.py
    embeddings.py
  utils/
    document_loader.py
    text_splitter.py
    vector_store.py
    rag_pipeline.py
    web_search.py
    query_router.py
    research_planner.py
    research_agent.py
  app.py
  requirements.txt
```

## Prerequisites

- Python 3.11 recommended
- pip
- API keys for required providers

## Environment Configuration

Create and configure .env in the project root.

Required variables:

- OPENAI_API_KEY
- GROQ_API_KEY
- TAVILY_API_KEY

Optional variables:

- LLM_PROVIDER (default: openai)
- OPENAI_MODEL (default: gpt-4o-mini)
- GROQ_MODEL (default: llama-3.1-8b-instant)
- EMBEDDING_MODEL (default: sentence-transformers/all-MiniLM-L6-v2)
- DEFAULT_TOP_K (default: 4)
- MAX_SEARCH_RESULTS (default: 5)

## Installation

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

Default URL:

- http://localhost:8501

## Usage

1. Upload documents from the sidebar and click Process Documents.
2. Select response mode:
   - concise
   - detailed
3. Select system mode:
   - chat
   - research
   - deep research
4. Ask a question in the chat input.
5. Review response and attached sources.

## Deployment (Streamlit Cloud)

1. Push this project to a GitHub repository.
2. In Streamlit Cloud, create a new app from the repo.
3. Set the app entry point to app.py.
4. Configure environment variables in Streamlit Cloud secrets.
5. Deploy and validate provider connectivity.

## Operational Notes

- OpenAI may return insufficient_quota (HTTP 429) when billing or quota is unavailable.
- The app includes fallback behavior to Groq when OpenAI quota/rate-limit errors occur and Groq key is configured.
- On Windows, FAISS and sentence-transformers may be unavailable; built-in fallbacks preserve functional behavior.

## Security Guidance

- Do not commit .env or secrets.
- Rotate keys immediately if they are exposed.
- Use project-level API keys with scoped permissions and spend limits.

## License

Use and adapt according to your project or organization licensing policy.
