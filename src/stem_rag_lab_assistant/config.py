import os
from pathlib import Path

# From src/configs/config.py -> project root is 3 levels up.
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "dataset"
LOADED_DATA_DIR = ROOT_DIR / "loaded_data"


def to_relative_path(path: Path) -> str:
    """Return ``path`` relative to the project root when possible.

    Artifacts (chunks.json / images.json) store portable paths (e.g.
    ``data/Lab1.pdf``) so they are not tied to a machine-specific absolute
    location. Falls back to the raw path if it is outside the project root.
    """
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path)

# Chunking placeholders
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# Model placeholders
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
ANSWER_MODEL_NAME = os.getenv("ANSWER_MODEL_NAME", "openai/gpt-oss-20b")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "openai/gpt-oss-120b")

# Retrieval placeholders
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "6"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "6"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))

# Cross-encoder reranker
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

# Graph retrieval (P6)
GRAPH_NEIGHBOUR_CAP = int(os.getenv("GRAPH_NEIGHBOUR_CAP", "2"))
GRAPH_EXPANSION_DEPTH = int(os.getenv("GRAPH_EXPANSION_DEPTH", "1"))
GRAPH_SEED_ENTITIES_CAP = int(os.getenv("GRAPH_SEED_ENTITIES_CAP", "5"))
GRAPH_MAX_EXPANDED_CHUNKS = int(os.getenv("GRAPH_MAX_EXPANDED_CHUNKS", "10"))

# LightRAG graph path (used by lightrag_hybrid method)
LIGHTRAG_WORKING_DIR = ROOT_DIR / "lightrag_data"

# Graph extraction (P3)
EXTRACTION_LLM_MODEL = os.getenv("EXTRACTION_LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
EXTRACTION_MAX_CONCURRENT = int(os.getenv("EXTRACTION_MAX_CONCURRENT", "2"))
EXTRACTION_MAX_ENTITIES_PER_CHUNK = int(os.getenv("EXTRACTION_MAX_ENTITIES_PER_CHUNK", "30"))
EXTRACTION_MAX_TOTAL_PER_CHUNK = int(os.getenv("EXTRACTION_MAX_TOTAL_PER_CHUNK", "50"))
EXTRACTION_MAX_TOKENS = int(os.getenv("EXTRACTION_MAX_TOKENS", "4096"))
EXTRACTION_RETRY_MAX = int(os.getenv("EXTRACTION_RETRY_MAX", "2"))
