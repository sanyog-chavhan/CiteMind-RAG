from pathlib import Path

# Source document (downloaded automatically if missing)
PAPER_URL = "https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf"
PAPER_FILENAME = "cacm12.pdf"

# Paths
BASE_DIR = Path(__file__).resolve().parent
PAPERS_DIR = BASE_DIR / "papers"
CHROMA_PATH = BASE_DIR / "chroma_db"

# Vector Store Configuration
COLLECTION_NAME = "research_papers"

# Chunking Configuration (used by ingest.py)
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Number of characters to overlap between chunks

# Retrieval Configuration (used by rag.py)
TOP_K = 4  # Number of top documents to retrieve for a query

# Embedding Model Configuration (for ingestion and querying)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Claude Configuration (for querying)
CLAUDE_MODEL = "claude-sonnet-5-5"  # Model for generating answers
MAX_OUTPUT_TOKENS = 1024  # Maximum tokens for the generated answer
TEMPERATURE = 0.0  # Temperature for response generation (controls randomness)

