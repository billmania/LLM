"""Embed the text."""
import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from sentence_transformers import SentenceTransformer

import torch

from tqdm import tqdm


class EmbeddingIndexer:
    """Create an index for the embedding."""

    def __init__(self, model_name: str, db_path: Path, collection_name: str):
        """Initialize the attributes."""
        print(f'Loading embedding model: {model_name}')
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f'Using device: {self.device}')

        # Initialize Qdrant
        self.client = QdrantClient(path=str(db_path))
        self.collection_name = collection_name

        # Get embedding dimension
        test_embedding = self.model.encode('test')
        self.embedding_dim = len(test_embedding)

        # Create collection
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f'Created collection: {collection_name}')
        except Exception as e:
            print(f'Collection may already exist: {e}')

    def embed_and_index(self, chunks_file: Path, batch_size: int = 32):
        """Generate embeddings and index all chunks."""
        chunks = []
        with open(chunks_file, 'r') as f:
            for line in f:
                chunks.append(json.loads(line))

        print(f'Indexing {len(chunks)} chunks...')

        # Process in batches
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i + batch_size]
            texts = [chunk['text'] for chunk in batch]

            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                device=self.device
            )

            # Create points for Qdrant
            points = []
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                point_id = i + j
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        'text': chunk['text'],
                        'metadata': chunk['metadata']
                    }
                ))

            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        print(f'Indexed {len(chunks)} chunks successfully')
