#!/usr/bin/env python
"""Inspect a few vectors from the collection."""
from pathlib import Path

from config import VECTOR_DB_DIR

from qdrant_client import QdrantClient

db_path = Path(VECTOR_DB_DIR)
client = QdrantClient(path=str(db_path))

print('Collections:', client.get_collections())

collection_name = 'documents'

points = client.scroll(
    collection_name=collection_name,
    limit=5,
    with_payload=True,
    with_vectors=True
)

for i, point in enumerate(points[0]):
    print(f'\n=== Point {i} (ID: {point.id}) ===')
    print(f"Text: {point.payload['text'][:200]}...")
    print(f"Metadata: {point.payload['metadata']}")
    print(f'Vector (first 10 dims): {point.vector[:10]}')
    print(
        f'Vector stats: min={min(point.vector):.4f}'
        f', max={max(point.vector):.4f}'
        f', mean={sum(point.vector)/len(point.vector):.4f}'
    )
