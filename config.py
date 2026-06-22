"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()


def _clean_openai_key(value: str | None) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if value.endswith("...") or value == "sk-..." or "[your_key]" in value.lower():
        return ""
    return value

# --- API Keys ---
OPENAI_API_KEY = _clean_openai_key(os.getenv("OPENAI_API_KEY", ""))

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
