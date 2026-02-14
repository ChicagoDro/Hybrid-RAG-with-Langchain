# Hybrid RAG using LangChain

A **showcase of the LangChain ecosystem**: a production-style RAG chatbot that combines **vector search** and **GraphRAG** with **LCEL**, **streaming**, and **LangSmith** for observability. Built for demos, learning, and as a reference architecture.

![Chi-Town Custom Chopper](./chopper.png)

**Domain:** The app is powered by a synthetic knowledge base for **Chitown Custom Choppers**, a fictional bicycle shop in Rogers Park, Chicago. Documents cover HR, customer policies, products, marketing, and operations—ideal for demonstrating multi-document RAG, entity relationships, and hybrid retrieval without real sensitive data.

---

## Why LangChain?

This project is built to **show what the LangChain stack can do** in one place:

| Capability | How we use it |
|------------|----------------|
| **LCEL (LangChain Expression Language)** | All retrieval and answer paths are composed with `\|`, `RunnableBranch`, `RunnableLambda`, and `RunnableParallel`—no ad-hoc glue code. |
| **Runnable semantics** | Every step has a clear input/output and `run_name` so **LangSmith traces** tell the story (classify → retrieve → answer, or build graph context → optional vector context → answer). |
| **Streaming** | The chatbot streams the LLM response token-by-token via `chain.stream()` and a Streamlit placeholder. |
| **LangSmith** | Tracing is wired so you can open [smith.langchain.com](https://smith.langchain.com), run a query, and see the full pipeline: router → vector or graph chain → sub-steps. |
| **Multi-provider** | One codebase supports **OpenAI**, **Gemini**, **Claude**, and **Grok** for both chat and embeddings via env-based config. |
| **Structured + free-form** | Document classification uses **Pydantic** + `JsonOutputParser`; final answers are natural language with optional source citations. |

You get a single repo that demonstrates **routing**, **hybrid context** (graph + vector), **streaming**, and **observability**—all with LangChain-native patterns.

---

## High-Level Architecture

```
                    User query
                         │
                         ▼
              ┌──────────────────────┐
              │  Chat router (LCEL)  │  RunnableBranch
              │  is_graph_query()?   │
              └──────────┬───────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│  Vector RAG chain   │       │  Graph RAG chain    │
│  (document-based)   │       │  (graph + optional  │
│                     │       │   vector for sales) │
│ • classify → filter │       │                     │
│ • FAISS retrieval   │       │ • build context     │
│ • prompt | LLM      │       │ • optional vector   │
│ • stream answer     │       │ • prompt | LLM      │
└─────────────────────┘       └─────────────────────┘
         │                               │
         └───────────────┬───────────────┘
                         ▼
                   { answer, sources }
```

- **Vector path:** Query → document-type classification (LLM) → FAISS retrieval with metadata filter → RAG prompt → LLM → streamed answer + sources.
- **Graph path:** Query → build graph context (person/org/sales) → optionally add product/marketing vector context → single RAG prompt → LLM → answer + sources.
- **Router:** Heuristics (tokens + phrases) decide graph vs vector; implemented as `RunnableBranch` so the chosen branch appears clearly in LangSmith.

---

## LangChain & LangSmith in This Project

### LCEL chains

- **Vector RAG:** `RunnableParallel` (classify + passthrough) → `RunnableLambda` (retrieve with scores) → `RunnableParallel` (answer + sources). Each step has a `run_name` for tracing.
- **KG RAG (knowledge graph):** Three runnables in sequence—**kg_build_context** → **kg_augment_sales** → **kg_answer**—so the trace shows exactly where time is spent and what context was added.
- **Top level:** `RunnableBranch` between graph and vector chain; `generate_chat_response_stream()` calls `chain.stream()` for streaming.

### Observability

- **LangSmith:** Set `LANGCHAIN_API_KEY` and `LANGCHAIN_PROJECT` in `.env`; tracing on/off is in `src/config.py` (`LANGCHAIN_TRACING_V2`).
- **Run names:** Every runnable uses `.with_config(run_name="...")` so traces are readable (e.g. `vector_classify`, `vector_retrieve`, `kg_build_context`, `kg_answer`).
- **Streamlit sidebar:** Optional “Test LangSmith connection” and display of tracing/config state.

### Streaming

- The app uses `generate_chat_response_stream()`, which yields chunks from `chat_chain.stream(..., stream_mode="values")`. The UI updates a placeholder as tokens arrive (vector path streams; graph path typically emits one chunk).

---

## Repository Layout

```text
Hybrid-RAG-using-Langchain/          # or your repo name
├── README.md
├── requirements.txt
├── .env.example
├── Makefile                         # db, graph-db, index, app, clean, libs
├── src/
│   ├── config.py                    # Paths, index dir, LangSmith toggle
│   ├── setup/                       # Database management
│   │   ├── create_database.sql
│   │   ├── seed_data.sql
│   │   └── drop_database.sql
│   ├── RAG_build/                   # Offline indexing
│   │   ├── ingest_embed_index.py    # Chunk, embed, FAISS index
│   │   └── graph_kg_builder.py      # SQLite → graph_output.json
│   └── RAG_chatbot/
│       ├── streamlit_app.py         # Streamlit UI (calls orchestrator)
│       ├── chat_orchestrator.py     # Router, vector + graph chains, streaming
│       └── graph_retrieval.py       # Graph load, format_* helpers, sales data
├── data/
│   └── Chitown_Custom_Choppers/
│       ├── company_docs/           # PDFs
│       ├── document-metadata.json
│       ├── chitown_graph.db         # SQLite (created by make db)
│       └── graph/
│           └── graph_output.json   # Built by graph_kg_builder
└── indices/
    └── faiss_company_docs_index_{provider}/
```

- **config.py:** Single place for project root, data paths, index dir (provider-specific), and LangSmith settings.
- **chat_orchestrator.py:** All LangChain logic—router, vector RAG chain, graph RAG chain (3 steps), and streaming entrypoint.
- **streamlit_app.py:** Loads chains via orchestrator, uses `@st.cache_resource` for index and graph, and renders streamed answers + sources.

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd <repo-name>
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

### 2. Environment

```bash
cp .env.example .env
```

Edit `.env`: set **LLM_PROVIDER** (e.g. `openai`, `gemini`, `claude`, `grok`) and the matching API key(s). For LangSmith tracing, set **LANGCHAIN_API_KEY** and **LANGCHAIN_PROJECT** (tracing on/off is in `src/config.py`).

### 3. Data and indices

```bash
make db        # Create and seed SQLite graph DB
make graph-db  # Build graph from DB → graph_output.json
make index     # Ingest PDFs, build FAISS index (uses LLM_PROVIDER for embeddings)
```

Or in one go: `make all` (graph-db → index → app).

### 4. Run the app

```bash
make app
# or: streamlit run src/RAG_chatbot/streamlit_app.py
```

Open the URL (e.g. http://localhost:8501). Ask about policies, products, org structure, or Q3 sales; then open LangSmith to inspect the trace for that query.

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make libs` | Install/upgrade from requirements.txt |
| `make db` | Create and seed SQLite DB (setup scripts) |
| `make graph-db` | Build graph from DB → graph_output.json (depends on db) |
| `make index` | Build FAISS index from company_docs (depends on LLM_PROVIDER) |
| `make app` | Run Streamlit app (depends on index) |
| `make all` | graph-db → index → app |
| `make clean` | Remove indices and SQLite DB |

---

## What This Showcase Demonstrates

- **LangChain:** LCEL composition, runnables, prompts, parsers, FAISS integration, multi-provider LLM/embeddings.
- **LangSmith:** End-to-end tracing with named steps, connection check in the UI, and config-driven enable/disable.
- **Hybrid RAG:** Router chooses between vector-only and graph (+ optional vector) paths; both return `{ answer, sources }` and support streaming where applicable.
- **Production-style structure:** Config module, clear separation between UI (Streamlit) and orchestration (LangChain), and a single entrypoint for running and observing the pipeline.

---

## About the Author

**Pete Tamisin** – Technical GTM Leader • AI & Data Engineering Architect • Chicago, IL.

- 20+ years in data & AI platforms (Capital One, ex-Databricks, startups).
- Focus: modern data platforms, RAG systems, enterprise GenAI adoption, and teaching teams to ship real-world AI.

📧 [pete@tamisin.com](mailto:pete@tamisin.com) • 🔗 [LinkedIn](https://www.linkedin.com/in/peter-tamisin-50a3233a/)

---

*Hybrid RAG using LangChain* – where the LangChain ecosystem meets a full retrieval pipeline. Fork, extend, and use it as a reference for your own RAG and observability demos.
