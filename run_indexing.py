#!/usr/bin/env python3
"""Run on the desktop."""
import sys
from pathlib import Path

from config import (
    CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL,
    PROCESSED_DIR, QDRANT_COLLECTION, RAW_DIR, VECTOR_DB_DIR
)

from indexing.chunk import TextChunker
from indexing.embed import EmbeddingIndexer
from indexing.extract import DocumentExtractor

sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Run the logic."""
    print('=== Document Indexing Pipeline ===')

    print('\n[1/3] Extracting text from documents...')
    extractor = DocumentExtractor(PROCESSED_DIR)
    num_docs = extractor.process_all(RAW_DIR)

    print('\n[2/3] Chunking documents...')
    chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    num_chunks = chunker.process_documents(
        PROCESSED_DIR / 'extracted_documents.jsonl',
        PROCESSED_DIR / 'chunks.jsonl'
    )

    print('\n[3/3] Generating embeddings and indexing...')
    indexer = EmbeddingIndexer(
        EMBEDDING_MODEL,
        VECTOR_DB_DIR,
        QDRANT_COLLECTION
    )
    indexer.embed_and_index(
        PROCESSED_DIR / 'chunks.jsonl',
        batch_size=EMBEDDING_BATCH_SIZE
    )

    print('\n=== Indexing Complete ===')
    print(f'Documents processed: {num_docs}')
    print(f'Chunks created: {num_chunks}')
    print(f'Vector DB location: {VECTOR_DB_DIR}')
    print("\nNext: Copy {VECTOR_DB_DIR} to your Orin's USB SSD")


if __name__ == '__main__':
    main()
