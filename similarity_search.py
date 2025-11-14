#!/usr/bin/env python
"""Test for similarity."""
from pathlib import Path

from config import EMBEDDING_MODEL, VECTOR_DB_DIR

from qdrant_client import QdrantClient

from sentence_transformers import SentenceTransformer


db_path = Path(VECTOR_DB_DIR)
client = QdrantClient(path=str(db_path))
model = SentenceTransformer(EMBEDDING_MODEL)

print('Collections:', client.get_collections())

collection_name = 'documents'

query = 'Python programming language'
query_vector = model.encode(query)

print(f'\nQuery: {query}')
print(f'Query vector (first 10): {query_vector[:10]}')

# Search
results = client.search(
    collection_name=collection_name,
    query_vector=query_vector.tolist(),
    limit=3
)

print('\n=== Top 3 Results ===')
for i, result in enumerate(results):
    print(f'\n{i+1}. Score: {result.score:.4f}')
    print(f"   Text: {result.payload['text'][:200]}")
    print(f"   Source: {result.payload['metadata']['source']}")
