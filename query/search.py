"""Search through the vectors."""
from pathlib import Path

from qdrant_client import QdrantClient

from sentence_transformers import SentenceTransformer


class VectorSearcher:
    """Search for a vector."""

    def __init__(self, model_name: str, db_path: Path, collection_name: str):
        """Initialize the attributes."""
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(path=str(db_path))
        self.collection_name = collection_name

    def search(self, query: str, top_k: int = 5):
        """Search for relevant chunks."""
        query_embedding = self.model.encode(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )

        return results
