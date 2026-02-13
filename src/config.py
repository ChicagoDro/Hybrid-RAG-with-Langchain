"""
Central configuration: paths, directory names, and shared settings.
Import this module so all components use the same paths and env-derived values.
"""
import os
from pathlib import Path

# ------------------------------------------------------------------------------
# Project layout
# ------------------------------------------------------------------------------

# Repo root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Environment
ENV_PATH = PROJECT_ROOT / ".env"

# ------------------------------------------------------------------------------
# Data directories
# ------------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "Chitown_Custom_Choppers"
COMPANY_DOCS_DIR = DATA_DIR / "company_docs"
GRAPH_DIR = DATA_DIR / "graph"

# Document metadata JSON (used by ingestion)
DOCUMENT_METADATA_PATH = DATA_DIR / "document-metadata.json"

# Graph outputs
GRAPH_OUTPUT_PATH = GRAPH_DIR / "graph_output.json"

# SQLite graph DB (created by setup SQL scripts, read by graph_kg_builder + graph_retrieval)
SQLITE_DB_PATH = DATA_DIR / "chitown_graph.db"

# ------------------------------------------------------------------------------
# Indices (vector store)
# ------------------------------------------------------------------------------

INDICES_DIR = PROJECT_ROOT / "indices"
INDEX_NAME_PREFIX = "faiss_company_docs_index"


def get_llm_provider() -> str:
    """LLM_PROVIDER env var, default 'openai'. Used for index path and embedding model."""
    return os.getenv("LLM_PROVIDER", "openai").lower()


def get_index_dir() -> Path:
    """Directory where the FAISS index for the current provider is stored."""
    return INDICES_DIR / f"{INDEX_NAME_PREFIX}_{get_llm_provider()}"


# ------------------------------------------------------------------------------
# Setup (database management SQL files)
# ------------------------------------------------------------------------------

SETUP_DIR = PROJECT_ROOT / "src" / "setup"
CREATE_DB_SQL_PATH = SETUP_DIR / "create_database.sql"
SEED_DATA_SQL_PATH = SETUP_DIR / "seed_data.sql"
DROP_DB_SQL_PATH = SETUP_DIR / "drop_database.sql"

# ------------------------------------------------------------------------------
# Sales reporting (default period for summaries)
# ------------------------------------------------------------------------------

SALES_DEFAULT_YEAR = 2024
SALES_DEFAULT_QUARTER = 3  # 1=Jan–Mar, 2=Apr–Jun, 3=Jul–Sep, 4=Oct–Dec

# ------------------------------------------------------------------------------
# LangSmith tracing
# ------------------------------------------------------------------------------

# Enable LangSmith tracing (set in code; API key and project stay in .env)
LANGCHAIN_TRACING_V2 = True
