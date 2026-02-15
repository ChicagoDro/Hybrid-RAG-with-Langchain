import os
from typing import Dict, Any, List, Literal, Optional, Tuple
from operator import itemgetter

import sys
from pathlib import Path

import re

# Ensure project root is on sys.path so `src` can be imported
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from src import config

# Load .env BEFORE any LangChain/LangSmith imports so tracing env vars are set
from dotenv import load_dotenv
load_dotenv(dotenv_path=config.ENV_PATH)

# --- LangChain Core ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
    RunnableParallel,
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.tools import BaseTool

from pydantic import BaseModel, Field
from langsmith import traceable
from langsmith import Client as LangSmithClient

# --- Vector Store & Embeddings ---
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_anthropic import ChatAnthropic



# --- GraphRAG utilities ---
from src.RAG_chatbot.graph_retrieval import (
    load_graph,
    list_people,
    format_org_summary,
    format_person_sales_summary,
    format_sales_overview,
    list_products,
    list_marketing_campaigns,
)

# ----------------------------- CONFIG ---------------------------------------------
# Paths and index dir from src.config


# ----------------------------- LANGSMITH CONFIGURATION -----------------------------


def configure_langsmith() -> None:
    """
    Ensure LangSmith env vars are set in os.environ so LangChain picks them up.
    Must run after load_dotenv() and before any chain invokes.
    """
    # Re-read from .env in case we're in a subprocess (e.g. Streamlit) with different cwd
    load_dotenv(dotenv_path=config.ENV_PATH)
    api_key = (os.getenv("LANGCHAIN_API_KEY") or "").strip()
    project = (os.getenv("LANGCHAIN_PROJECT") or "chitown-custom-choppers-chat").strip()
    # Force into process so LangChain's client sees them (tracing from config)
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.LANGCHAIN_TRACING_V2 else "false"
    if api_key and api_key != "your_langsmith_api_key_here":
        os.environ["LANGCHAIN_API_KEY"] = api_key
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project


def get_langsmith_debug_info() -> Dict[str, Any]:
    """Return current LangSmith-related env state (for sidebar debug)."""
    k = os.getenv("LANGCHAIN_API_KEY") or ""
    return {
        "LANGCHAIN_TRACING_V2": str(config.LANGCHAIN_TRACING_V2),
        "LANGCHAIN_API_KEY set": bool(k and k != "your_langsmith_api_key_here"),
        "LANGCHAIN_API_KEY prefix": k[:10] + "..." if len(k) > 10 else "(empty)",
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "(not set)"),
    }


def test_langsmith_connection() -> Dict[str, Any]:
    """
    Verify LangSmith API key and endpoint connectivity.
    Returns dict with: ok (bool), message (str), detail (str or None).
    """
    api_key = (os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if not api_key or api_key == "your_langsmith_api_key_here":
        return {
            "ok": False,
            "message": "API key not set",
            "detail": "Set LANGCHAIN_API_KEY in .env",
        }
    try:
        client = LangSmithClient(api_key=api_key)
        # Minimal API call to verify auth and that the endpoint responds
        projects = list(client.list_projects(limit=1))
        return {
            "ok": True,
            "message": "Authenticated; API responding",
            "detail": f"Workspace has {len(projects)}+ project(s). Endpoint: {getattr(client, 'api_url', 'default')}",
        }
    except Exception as e:
        err = str(e).strip()
        if "401" in err or "Unauthorized" in err or "invalid" in err.lower():
            return {"ok": False, "message": "Authentication failed", "detail": err}
        if "resolve" in err.lower() or "Connection" in err or "timeout" in err.lower():
            return {"ok": False, "message": "Endpoint unreachable", "detail": err}
        return {"ok": False, "message": "Connection test failed", "detail": err}


# Initialize LangSmith configuration
configure_langsmith()


# ----------------------------- LLM FACTORY ----------------------------------------


def get_llm():
    """
    Return an LLM configured for one of the supported providers:
      - OpenAI   (default)
      - Grok     (OpenAI-compatible endpoint)
      - Gemini   (Google Generative AI)
      - Claude   (Anthropic)

    Controlled via env var: LLM_PROVIDER in {"openai", "grok", "gemini", "claude"}.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # Env vars come in as strings — cast them
    temperature = float(os.getenv("LLM_TEMP", "0.2"))
    timeout = float(os.getenv("LLM_TIMEOUT", "60.0"))

    # ---------------------------
    # Claude (Anthropic)
    # ---------------------------
    if provider in {"claude", "anthropic"}:
        return ChatAnthropic(
            # Uses ANTHROPIC_API_KEY from the environment by default
            model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            temperature=temperature,
            timeout=timeout,
        )

    # ---------------------------
    # Gemini (Google GenAI)
    # ---------------------------
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            google_api_key=os.environ["GEMINI_API_KEY"],
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=temperature,
            timeout=timeout,
        )

    # ---------------------------
    # Grok (OpenAI-compatible)
    # ---------------------------
    if provider == "grok":
        return ChatOpenAI(
            api_key=os.environ["GROK_API_KEY"],
            model=os.getenv("GROK_MODEL", "grok-2-latest"),
            base_url=os.getenv("GROK_API_BASE", None),
            temperature=temperature,
            timeout=timeout,
        )

    # ---------------------------
    # OpenAI (default)
    # ---------------------------
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=temperature,
        timeout=timeout,
    )

def get_embedding_model():
    """
    Return an embeddings model compatible with the current provider.

    By default, we tie embeddings to LLM_PROVIDER:
      - LLM_PROVIDER=gemini  -> GoogleGenerativeAIEmbeddings
      - LLM_PROVIDER=openai  -> OpenAIEmbeddings
      - LLM_PROVIDER=grok    -> OpenAIEmbeddings (OpenAI-compatible)

    You can override with EMBEDDING_PROVIDER if needed.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "openai")).lower()

    # ---------------------------
    # Gemini embeddings
    # ---------------------------
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            google_api_key=os.environ["GEMINI_API_KEY"],
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
        )

    # ---------------------------
    # Default: OpenAI embeddings
    # ---------------------------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Either:\n"
            "- Set LLM_PROVIDER=gemini (and GEMINI_API_KEY) so embeddings use Gemini, or\n"
            "- Provide OPENAI_API_KEY for OpenAI/Grok embeddings."
        )

    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )


# ----------------------------- CLASSIFICATION MODEL ------------------------------


class DocumentCategory(BaseModel):
    category: str = Field(
        description=(
            "The most relevant document_type category for answering this query. "
            "Must be one of: 'HR Policy', 'Customer Policy', 'Customer Service', "
            " 'Operations Manual', 'Marketing', or 'Other'."
        )
    )


class RouteDecision(BaseModel):
    """LLM routing decision: which RAG path and (for KG) which context type. Editable in LangSmith Playground."""
    route: Literal["vector", "kg_org", "kg_sales"] = Field(
        description=(
            "Use 'vector' for questions answered by company documents: return policy, HR policy, "
            "customer service, products, marketing, operations, anything from PDFs.\n"
            "Use 'kg_org' for org structure only: org chart, who works here, who reports to whom, "
            "headcount, CEO/founder, departments—no sales numbers.\n"
            "Use 'kg_sales' for sales performance and numbers: who sold the most, top seller, "
            "best salesperson, revenue, Q3 sales, monthly sales, sold, sales by person/department."
        )
    )


class VectorStoreFilterRetriever(BaseRetriever):
    """
    Retriever over the vector store with optional metadata filter from config.
    Shows as run_type=retriever in LangSmith.
    """
    vectorstore: Any

    def _get_relevant_documents(
        self,
        query: str,
        *,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        search_kwargs: Dict[str, Any] = {"k": 3}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        docs_and_scores: List[Tuple[Document, float]] = (
            self.vectorstore.similarity_search_with_score(query, **search_kwargs)
        )
        return [doc for doc, _ in docs_and_scores]

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Override to pass metadata_filter from config into retrieval while keeping LangSmith callbacks."""
        config = ensure_config(config)
        metadata_filter = config.get("configurable", {}).get("metadata_filter")
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags"),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata") or {},
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=kwargs.pop("run_id", None),
        )
        try:
            result = self._get_relevant_documents(
                input, metadata_filter=metadata_filter
            )
            run_manager.on_retriever_end(result)
            return result
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise


# ----------------------------- RAG CHAIN SETUP -----------------------------------


def setup_rag_chain():
    # 1. Load vector store with LLM specific embeddings
    embedding_model = get_embedding_model()

    index_dir = config.get_index_dir()
    if not os.path.exists(index_dir):
        raise FileNotFoundError(
            f"Vector index directory '{index_dir}' not found. "
            f"Run ingest_embed_index.py first."
        )

    loaded_vectorstore = FAISS.load_local(
        index_dir,
        embedding_model,
        allow_dangerous_deserialization=True,  # ok for local/dev
    )

    llm = get_llm()

    # ---------------- Classification prompt & chain ----------------

    format_instructions = JsonOutputParser(
        pydantic_object=DocumentCategory
    ).get_format_instructions()

    CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a routing assistant for Chitown Custom Choppers, a custom chopper "
                    "bicycle shop in Rogers Park, Chicago.\n\n"
                    "Your job is to select the single best internal document category for a query.\n"
                    "Choose from:\n"
                    "  - HR Policy\n"
                    "  - Customer Policy\n"
                    "  - Customer Service\n"
                    "  - Operations Manual\n"
                    "  - Marketing\n"
                    "  - Other\n\n"
                    "Return ONLY a JSON object following these instructions:\n"
                    "{format_instructions}"
                ),
            ),
            ("human", "User query: {query}"),
        ]
    ).partial(format_instructions=format_instructions)

    parser = JsonOutputParser(pydantic_object=DocumentCategory)
    classification_chain = (
        CLASSIFICATION_PROMPT | llm | parser
    ).with_config(run_name="vector_classify_document_type")

    valid_categories = {
        "HR Policy",
        "Customer Policy",
        "Customer Service",
        "Operations Manual",
        "Marketing",
    }

    # ---------------- Helper: build filter from classification ----------------

    def build_metadata_filter(classifier_output: Any) -> Dict[str, Any] | None:
        """Convert classifier output into a FAISS metadata filter."""
        if isinstance(classifier_output, DocumentCategory):
            category = classifier_output.category
        elif isinstance(classifier_output, dict):
            category = classifier_output.get("category")
        else:
            category = None

        print(f"\n[DEBUG] Classification raw output: {classifier_output}")
        print(f"[DEBUG] Chosen category: {category}")

        if category in valid_categories:
            return {"document_type": category}
        else:
            print("[DEBUG] Unknown or missing category; using unfiltered search.")
            return None

    # ---------------- Retriever (run_type=retriever) + format step --------------

    # Retriever (run_type=retriever in LangSmith); filter passed via config
    vector_retriever = VectorStoreFilterRetriever(
        vectorstore=loaded_vectorstore
    ).with_config(run_name="vector_retrieve")

    def _invoke_retriever_and_passthrough_query(inputs: Dict[str, Any]) -> Dict[str, Any]:
        metadata_filter = build_metadata_filter(inputs["classification"])
        run_config: Dict[str, Any] = {
            "configurable": {"metadata_filter": metadata_filter}
        }
        docs = vector_retriever.invoke(inputs["query"], config=run_config)
        out: Dict[str, Any] = {"query": inputs["query"], "docs": docs}
        if "language" in inputs:
            out["language"] = inputs["language"]
        return out

    def _format_docs_to_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Build context string and sources from retrieved documents."""
        query = inputs["query"]
        docs: List[Document] = inputs["docs"]
        context_chunks = []
        sources = []
        for i, doc in enumerate(docs):
            file_name = doc.metadata.get("file_name", "Unknown")
            doc_type = doc.metadata.get("document_type", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            context_chunks.append(
                f"[Source {i+1} | {file_name} | type={doc_type} | page={page}]\n"
                f"{doc.page_content}"
            )
            sources.append({
                "id": i + 1,
                "file_name": file_name,
                "document_type": doc_type,
                "page": page,
            })
        context = (
            "\n\n---\n\n".join(context_chunks)
            if context_chunks
            else "No relevant documents found."
        )
        out: Dict[str, Any] = {"query": query, "context": context, "sources": sources}
        if "language" in inputs:
            out["language"] = inputs["language"]
        return out

    retriever_step = RunnableLambda(_invoke_retriever_and_passthrough_query).with_config(
        run_name="vector_retrieve_step"
    )
    format_docs_step = RunnableLambda(_format_docs_to_context).with_config(
        run_name="vector_format_context"
    )

    # ---------------- Answer prompt (RAG with context) ---------------------

    ANSWER_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are the in-house assistant for Chitown Custom Choppers, a custom "
                    "chopper bicycle shop in Rogers Park, Chicago.\n\n"
                    "Answer ONLY using the information in the 'Context' below. "
                    "If the answer is not clearly supported by the context, say that you "
                    "cannot answer from the shop documents.\n\n"
                    "Keep your tone friendly, clear, and grounded in the provided text.\n\n"
                    #"Response language: {language}"
                ),
            ),
            (
                "human",
                "Context:\n{context}\n\n"
                "Customer Question: {query}\n\n"
                "Answer in a helpful, concise way."
            ),
        ]
    )
    #client = LangSmithClient(api_key=os.getenv("LANGCHAIN_API_KEY"))
    #ANSWER_PROMPT = client.pull_prompt(
    #    "chitown-chatbot-vector"
    #)
    # ---------------- Normalize input (dict with query + language) and classify ----------------------

    def _classify_and_passthrough(inp: Any) -> Dict[str, Any]:
        if isinstance(inp, dict):
            query = inp.get("query", "")
            language = inp.get("language", "English")
        else:
            query = inp
            language = "English"
        classification = classification_chain.invoke({"query": query})
        return {"query": query, "classification": classification, "language": language}

    classify_and_passthrough = RunnableLambda(_classify_and_passthrough).with_config(
        run_name="vector_classify"
    )

    answer_and_sources = RunnableParallel(
        answer=(ANSWER_PROMPT | llm | StrOutputParser()).with_config(
            run_name="vector_generate"
        ),
        sources=itemgetter("sources"),
    ).with_config(run_name="vector_answer")

    rag_chain = (
        classify_and_passthrough
        | retriever_step
        | format_docs_step
        | answer_and_sources
    ).with_config(run_name="vector_chain")

    return rag_chain, loaded_vectorstore


# ----------------------------- VECTOR RETRIEVAL FOR SALES HYBRID -------------------


def retrieve_product_and_marketing_context(
    vectorstore: Any,
    query: str,
    extra_terms: Optional[List[str]] = None,
    k_per_doc_type: int = 3,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve chunks from Marketing and Operations Manual docs, optionally
    expanding the query with product/campaign names. Used to augment graph
    sales data with product and campaign narrative from PDFs.
    Returns (context_string, sources_list).
    """
    search_query = query
    if extra_terms:
        search_query = f"{query} {' '.join(extra_terms)}"

    all_docs: List[Tuple[Document, float]] = []
    for document_type in ("Marketing", "Operations Manual"):
        try:
            docs_scores = vectorstore.similarity_search_with_score(
                search_query,
                k=k_per_doc_type,
                filter={"document_type": document_type},
            )
            all_docs.extend(docs_scores)
        except Exception:
            # Filter may not be supported or no docs of that type
            pass

    # Sort by score (distance) and take best overall
    all_docs.sort(key=lambda x: x[1])
    kept = all_docs[: 2 * k_per_doc_type]

    context_chunks = []
    sources = []
    for i, (doc, score) in enumerate(kept):
        file_name = doc.metadata.get("file_name", "Unknown")
        doc_type = doc.metadata.get("document_type", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        distance = float(score)
        confidence = 1.0 / (1.0 + distance)
        context_chunks.append(
            f"[Source {i+1} | {file_name} | type={doc_type} | page={page}]\n{doc.page_content}"
        )
        sources.append({
            "id": i + 1,
            "file_name": file_name,
            "document_type": doc_type,
            "page": page,
            "distance": distance,
            "confidence": confidence,
        })

    context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""
    return context, sources


# ----------------------------- GRAPH RAG SETUP ------------------------------------


def load_knowledge_graph():
    """Load the knowledge graph from data/graph/graph_output.json."""
    return load_graph()


GRAPH_KEYWORDS = [
    "org chart",
    "organization",
    "organisational structure",
    "org structure",
    "who works there",
    "who works at",
    "who reports to",
    "reports to",
    "manager",
    "direct reports",
    "team",
    "department",
    "headcount",
    "q3 sales",
    "quarter 3 sales",
    "monthly sales",
    "sales by employee",
    "sales by person",
]


def _normalize_for_routing(text: str) -> str:
    """
    Lowercase, remove apostrophes/punctuation, collapse spaces.
    This makes 'who's the CEO' and 'who is the ceo?' look similar.
    """
    text = text.lower()
    # remove apostrophes like who's → whos
    text = re.sub(r"[’']", "", text)
    # replace non-alphanumeric with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


ORG_TOKENS = {
    "org", "organization", "structure", "department", "team", "orgchart",
}
PEOPLE_TOKENS = {
    "who", "employee", "employees", "people", "staff", "manager", "managers",
    "report", "reports", "reporting", "headcount", "ceo", "founder", "owner",
}
SALES_TOKENS = {
    "sales", "revenue", "q3", "quarter", "monthly", "sold", "seller", "selling",
}
# "Who sold the most", "top seller", "best sales" — need graph sales data
SALES_PERFORMANCE_TOKENS = {"sold", "most", "top", "best", "highest", "leader", "leading"}


@traceable(name="main_route_graph_vs_vector")
def is_graph_query(user_input: str) -> bool:
    """
    Fuzzy-ish router for graph questions.

    Heuristics:
    - If the question is about org structure / people / roles / headcount → graph
    - If the question is about sales (including "who sold the most", top seller) → graph
    """
    raw = user_input
    text = _normalize_for_routing(user_input)
    tokens = set(text.split())

    # Debug so you can see what the router is doing
    print(f"[DEBUG][ROUTER] normalized='{text}', tokens={tokens}")

    # 1) Direct “org-ish” intent: structure / departments / teams / headcount
    if ("org" in tokens and "chart" in tokens) or ("structure" in tokens and "org" in tokens):
        print(f"[DEBUG][ROUTER] GraphRAG (org+chart/structure) for: {raw!r}")
        return True

    if tokens & ORG_TOKENS and tokens & PEOPLE_TOKENS:
        # e.g. "organization structure", "team structure", "department staff"
        print(f"[DEBUG][ROUTER] GraphRAG (org+people tokens) for: {raw!r}")
        return True

    # 2) “Who” questions about people or roles
    if "who" in tokens and (tokens & {"works", "work", "reports", "manages", "manager", "managers", "ceo", "founder", "owner"}):
        # catches: "who is the ceo", "who's the ceo", "who works here", "who reports to rosa"
        print(f"[DEBUG][ROUTER] GraphRAG (who+people) for: {raw!r}")
        return True

    if "ceo" in tokens or "founder" in tokens or "owner" in tokens:
        print(f"[DEBUG][ROUTER] GraphRAG (role=ceo/founder/owner) for: {raw!r}")
        return True

    # 3) Headcount / how many people / employees
    if "how" in tokens and ("many" in tokens or "total" in tokens) and (tokens & {"people", "employees", "staff", "headcount"}):
        print(f"[DEBUG][ROUTER] GraphRAG (headcount) for: {raw!r}")
        return True

    # 4) Sales performance: "who sold the most", "top seller", "best sales", "who had the highest sales"
    if "who" in tokens and (tokens & SALES_PERFORMANCE_TOKENS):
        print(f"[DEBUG][ROUTER] GraphRAG (who+sales performance) for: {raw!r}")
        return True
    if tokens & SALES_PERFORMANCE_TOKENS and (tokens & SALES_TOKENS or "seller" in tokens):
        print(f"[DEBUG][ROUTER] GraphRAG (sales performance) for: {raw!r}")
        return True

    # 5) Sales questions (Q3 / monthly / by employee / sold / revenue)
    if tokens & SALES_TOKENS:
        # e.g. "q3 sales", "monthly sales", "sales by employee", "who sold"
        print(f"[DEBUG][ROUTER] GraphRAG (sales) for: {raw!r}")
        return True

    print(f"[DEBUG][ROUTER] NOT routing to GraphRAG for: {raw!r}")
    return False

def _is_sales_context(q: str, used_person_summary: bool, used_sales_overview: bool) -> bool:
    """True if we built a sales-related graph context (person sales or company sales overview)."""
    return used_person_summary or used_sales_overview


# ----------------------------- GRAPH RAG AS LCEL STEPS (for clear LangSmith traces) -------------------


def _graph_build_context(
    query: str,
    G: Any,
    kg_context: Optional[Literal["kg_org", "kg_sales"]] = None,
    language: str = "English",
) -> Dict[str, Any]:
    """
    Step 1: Build graph context (person summary, org summary, or sales overview).
    When kg_context is set by the router LLM, use it; otherwise fall back to keyword logic.
    Returns state dict for the next runnable.
    """
    q = query.lower()
    used_person_summary = False
    used_sales_overview = False

    # Router LLM decided kg_org vs kg_sales: use it (no brittle keyword list)
    if kg_context == "kg_sales":
        graph_context = format_sales_overview(G)
        used_sales_overview = True
    elif kg_context == "kg_org":
        graph_context = format_org_summary(G)
    else:
        # Fallback when chain is invoked without kg_context (e.g. tests)
        person_id = None
        for person in list_people(G):
            if person["name"].lower() in q:
                person_id = person["id"]
                break

        if person_id:
            graph_context = format_person_sales_summary(G, person_id)
            used_person_summary = True
        elif any(
            phrase in q
            for phrase in (
                "org chart",
                "org structure",
                "organization",
                "who works there",
                "headcount",
                "who works at",
            )
        ):
            graph_context = format_org_summary(G)
        elif any(
            phrase in q
            for phrase in (
                "sales",
                "revenue",
                "figures",
                "performance",
                "numbers",
                "q3",
                "total sales",
                "sold",
                "most",
                "top",
                "best",
            )
        ):
            graph_context = format_sales_overview(G)
            used_sales_overview = True
        else:
            graph_context = format_org_summary(G)

    return {
        "user_query": query,
        "q": q,
        "graph_context": graph_context,
        "used_person_summary": used_person_summary,
        "used_sales_overview": used_sales_overview,
        "language": language,
    }


def _graph_add_vector_context(state: Dict[str, Any], vectorstore: Any) -> Dict[str, Any]:
    """
    Step 2: For sales context, optionally add product/marketing vector retrieval.
    Merges vector_context and sources into state.
    """
    vector_context = ""
    sources: List[Dict[str, Any]] = []
    if vectorstore and _is_sales_context(
        state["q"], state["used_person_summary"], state["used_sales_overview"]
    ):
        extra_terms = [p["name"] for p in list_products()] + [
            c["name"] for c in list_marketing_campaigns()
        ]
        vector_context, sources = retrieve_product_and_marketing_context(
            vectorstore,
            state["user_query"],
            extra_terms=extra_terms or None,
            k_per_doc_type=3,
        )
    return {**state, "vector_context": vector_context, "sources": sources}


def _graph_answer_llm(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: Combine context, run LLM, return {answer, sources}.
    """
    graph_context = state["graph_context"]
    vector_context = state.get("vector_context") or ""
    sources = state.get("sources") or []

    if vector_context:
        combined = (
            "--- Sales data (from database) ---\n"
            f"{graph_context}\n\n"
            "--- Product and marketing docs (from company documents) ---\n"
            f"{vector_context}"
        )
        system_extra = (
            " You also have excerpts from product and marketing documents; use them to add relevant detail (e.g. product names, campaign descriptions) when answering."
        )
    else:
        combined = graph_context
        system_extra = ""

    language = state.get("language", "English")
    graph_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an internal assistant for Chitown Custom Choppers. "
                    "You are given structured information from the company's "
                    "knowledge graph (org chart, departments, Q3 2024 sales)."
                    + system_extra
                    + "\n\n"
                    "Use the information below when answering. If the context "
                    "does not clearly support an answer, say so.\n\n"
                    #"Response language: {language}\n\n"
                ),
            ),
            (
                "human",
                "User question:\n{query}\n\n"
                "Context:\n{graph_context}\n\n"
                "Provide a clear, concise answer grounded in the context."
            ),
        ]
    )

    llm = get_llm()
    chain = (graph_prompt | llm | StrOutputParser()).with_config(
        run_name="kg_generate"
    )
    answer_text = chain.invoke(
        {"query": state["user_query"], "graph_context": combined, "language": language}
    )
    return {"answer": answer_text, "sources": sources}


class GraphBuildContextTool(BaseTool):
    """Build graph context for a query. Shows as run_type=tool in LangSmith."""
    name: str = "kg_build_context"
    description: str = "Build context from the knowledge graph for the given query."
    G: Any = None

    def _run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        inp = args[0] if args else kwargs.get("query", kwargs)
        if isinstance(inp, dict):
            query = inp.get("query", "")
            kg_context = inp.get("kg_context")  # kg_org | kg_sales from router
            language = inp.get("language", "English")
            return _graph_build_context(query, self.G, kg_context=kg_context, language=language)
        return _graph_build_context(inp, self.G, kg_context=None)


class GraphAugmentSalesTool(BaseTool):
    """Augment graph state with vector/sales context. Shows as run_type=tool in LangSmith."""
    name: str = "kg_augment_sales"
    description: str = "Augment graph context with product/marketing docs when relevant."
    vectorstore: Any = None

    def _run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        state = args[0] if args else kwargs.get("state", kwargs)
        if not isinstance(state, dict):
            state = kwargs
        return _graph_add_vector_context(state, self.vectorstore)


def build_kg_rag_chain(G: Any, vectorstore: Any = None):
    """
    Build the knowledge-graph RAG chain as 3 runnables so LangSmith shows distinct steps:
      kg_build_context (tool) -> kg_augment_sales (tool) -> kg_answer (chain)
    """
    step1_r = GraphBuildContextTool(G=G).with_config(run_name="kg_build_context")
    step2_r = GraphAugmentSalesTool(vectorstore=vectorstore).with_config(
        run_name="kg_augment_sales"
    )
    step3_r = RunnableLambda(_graph_answer_llm).with_config(run_name="kg_answer")

    return (step1_r | step2_r | step3_r).with_config(run_name="kg_chain")


def build_chat_router_chain(vector_rag_chain, kg_rag_chain):
    """
    LLM-based router: one prompt + LLM + parser decides kg vs vector.
    The routing run appears as main_route_llm in the trace so you can open it in
    LangSmith Playground, fix the prompt when the path is wrong, and save.
    """
    llm = get_llm()
    route_parser = JsonOutputParser(pydantic_object=RouteDecision)
    route_format = route_parser.get_format_instructions()

    ROUTER_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a router for Chitown Custom Choppers, a bicycle shop.\n\n"
                "Choose the single best backend for this user query. Use semantic understanding "
                "(e.g. 'sold' and 'sales' are related; 'who sold the most' needs sales data).\n\n"
                "- **vector**: Policies, documents, returns, HR, customer service, products, "
                "marketing, operations—anything answered from company documents or PDFs.\n\n"
                "- **kg_org**: Org structure only—org chart, who works here, who reports to whom, "
                "headcount, CEO/founder, departments. No sales figures.\n\n"
                "- **kg_sales**: Sales performance and numbers—who sold the most, top seller, "
                "revenue, Q3 sales, monthly sales, sold, sales by person or department.\n\n"
                "Return ONLY valid JSON per these instructions:\n"
                "{format_instructions}"
            ),
            ("human", "User query: {query}"),
        ]
    ).partial(format_instructions=route_format)

    router_chain = (
        ROUTER_PROMPT | llm | route_parser
    ).with_config(run_name="main_route_llm")

    def router_step(inp: Any) -> Dict[str, Any]:
        if isinstance(inp, dict):
            query = inp.get("query", "")
            language = inp.get("language", "English")
        else:
            query = inp
            language = "English"
        decision = router_chain.invoke({"query": query})
        route = (
            decision.route
            if isinstance(decision, RouteDecision)
            else decision.get("route", "vector")
        )
        return {"query": query, "route": route, "language": language}

    def run_chosen_path(inp: Dict[str, Any]) -> Dict[str, Any]:
        route = inp.get("route", "vector")
        language = inp.get("language", "English")
        if route == "vector":
            return vector_rag_chain.invoke({"query": inp["query"], "language": language})
        # kg_org or kg_sales: pass kg_context so the KG chain uses the right context type
        return kg_rag_chain.invoke({
            "query": inp["query"],
            "kg_context": route,
            "language": language,
        })

    return (
        RunnableLambda(router_step).with_config(run_name="main_route_router")
        | RunnableLambda(run_chosen_path).with_config(run_name="main_route_execute")
    ).with_config(run_name="main_route")


def _normalize_chat_input(user_input: Any) -> Dict[str, Any]:
    """Accept string or dict; return dict with query and language."""
    if isinstance(user_input, dict):
        return {
            "query": user_input.get("query", ""),
            "language": user_input.get("language", "English"),
        }
    return {"query": user_input, "language": "English"}


@traceable(name="chitown_chat_orchestrator")
def generate_chat_response(user_input: Any, chat_chain):
    """
    Top-level orchestrator for a single user query (non-streaming).
    user_input can be a string (query only) or dict with "query" and optional "language".
    """
    return chat_chain.invoke(_normalize_chat_input(user_input))


def generate_chat_response_stream(user_input: Any, chat_chain):
    """
    Stream the chat response chunk by chunk. Uses stream_mode="updates" so the LLM
    streams token deltas; yields dicts with "answer" (delta or full) and/or "sources".
    user_input can be a string (query only) or dict with "query" and optional "language".
    """
    inp = _normalize_chat_input(user_input)
    for chunk in chat_chain.stream(
        inp,
        config={"stream_mode": "updates"},
    ):
        if isinstance(chunk, dict) and ("answer" in chunk or "sources" in chunk):
            yield chunk


# ---------------------------------------------------------------------------
# LangSmith run names (hierarchy = parent -> child). run_name -> description.
# ---------------------------------------------------------------------------
# main_route                     Top-level: LLM router then execute chosen path.
#   main_route_router            Runs router step (query -> route decision).
#     main_route_llm             Prompt | LLM | parser for route (vector | kg_org | kg_sales). Edit in Playground.
#   main_route_execute           Invokes kg_chain or vector_chain from route.
#     kg_chain                   Full KG RAG pipeline (when route=kg).
#       kg_build_context         Tool: build graph context for query (person/org/sales).
#       kg_augment_sales         Tool: add product/marketing vector context when sales.
#       kg_answer                Chain: combine context, run LLM, return answer + sources.
#         kg_generate            Prompt | LLM | parser for final KG answer.
#     vector_chain               Full vector RAG pipeline (when route=vector).
#       vector_classify          Parallel: passthrough query + run doc-type classification.
#         vector_classify_document_type   Classify query to document category (LLM + parser).
#       vector_retrieve_step     Invoke retriever with filter from classification; return query + docs.
#         vector_retrieve        Retriever: fetch docs from vectorstore (optional metadata filter).
#       vector_format_context    Build context string and sources from retrieved docs.
#       vector_answer            Parallel: generate answer + passthrough sources.
#         vector_generate        Prompt | LLM | parser for RAG answer.
