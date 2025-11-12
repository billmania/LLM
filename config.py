"""Provide configuration details."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
VECTOR_DB_DIR = PROJECT_ROOT / 'vector_db'

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
PDF_MIN_WORDS = 50  # Skip PDFs with fewer words (likely scans)

EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
EMBEDDING_BATCH_SIZE = 32  # Adjust for GTX 1060 VRAM
LLM_MODEL = 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF'
LLM_MODEL_FILE = 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'

QDRANT_COLLECTION = 'documents'
TOP_K_RESULTS = 5
