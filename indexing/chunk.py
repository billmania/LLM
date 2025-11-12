"""Break the text into chunks."""
import json
from pathlib import Path
from typing import Dict, List


class TextChunker:
    """Break the text into chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """Initialize the attributes."""
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, doc_metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunks.append({
                'text': chunk_text,
                'metadata': doc_metadata
            })

        return chunks

    def process_documents(self, input_file: Path, output_file: Path):
        """Chunk all documents and save."""
        all_chunks = []

        with open(input_file, 'r') as f:
            for line in f:
                doc = json.loads(line)

                metadata = {
                    'source': doc['source'],
                    'type': doc['type'],
                    **doc.get('metadata', {})
                }

                chunks = self.chunk_text(doc['text'], metadata)
                all_chunks.extend(chunks)

        with open(output_file, 'w') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk) + '\n')

        print(f'Created {len(all_chunks)} chunks -> {output_file}')
        return len(all_chunks)
