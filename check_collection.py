#!/usr/bin/env python
"""Get the details of the vectors in the collection."""
from pathlib import Path

from config import VECTOR_DB_DIR

from qdrant_client import QdrantClient

db_path = Path(VECTOR_DB_DIR)
client = QdrantClient(path=str(db_path))

print('Collections:', client.get_collections())

collection_name = 'documents'
info = client.get_collection(collection_name)
print(f'\nCollection: {collection_name}')
print(f'Vector count: {info.points_count}')
print(f'Vector dimension: {info.config.params.vectors.size}')
print(f'Distance metric: {info.config.params.vectors.distance}')
