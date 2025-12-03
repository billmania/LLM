"""Break the text into chunks."""
import json
from pathlib import Path
from typing import Dict, List

from langchain_text_splitters import MarkdownTextSplitter


class TextChunker:
    """Break the text into chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """Initialize the attributes."""
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._chunks = 0

    def chunk_text(self, text: str, doc_metadata: Dict) -> List[Dict]:
        """Split plain text into overlapping chunks."""
        if type(text) is not str:
            print(
                f'chunk_test: WARNING: Not str type: {text}'
            )
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            # TODO: make email message metadata available to the generator
            chunks.append({
                'text': chunk_text,
                'metadata': doc_metadata
            })
            self._chunks += 1

        return chunks

    def chunk_markdown(self, text: str, source: str) -> List[Dict]:
        """Split Markdown text into overlapping chunks."""
        if type(text) is not str:
            raise TypeError(
                f'chunk_markdown() called with {type(text)} but must be str'
            )

        chunks = []

        md_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap
        )

        for chunk in md_splitter.create_documents([text]):
            chunks.append(
                {
                    'text': chunk.page_content,
                    'metadata': {
                        'source': source,
                        'type': 'markdown',
                        'metadata': {'why': 'filler'}
                    }
                }
            )
            self._chunks += 1

        return chunks

    def chunk_dict(self, doc: dict) -> List[Dict]:
        """Break a dict into chunks."""
        try:
            metadata = {
                'source': doc['source'],
                'type': doc['type'],
                **doc.get('metadata', {})
            }

        except Exception as e:
            print(
                'Exception in chunk_dict()'
                f', {e}'
            )

        chunks = self.chunk_text(doc['text'], metadata)

        return chunks

    def chunk_list(self, docs: list) -> List[Dict]:
        """Break a list of dicts into chunks."""
        list_chunks = []

        for doc in docs:
            list_chunks += self.chunk_dict(doc)

        return list_chunks

    def process_documents(self, input_file: Path, output_file: Path) -> int:
        """Chunk all documents and save.

        input_file is expected to contain the extracted documents as JSON
        objects, most of which are dicts and one of which is a list of dicts.
        output_file is where the chunks will be written.
        """
        all_chunks = []

        print(
            f'Processing extracted documents from {input_file}'
        )
        with open(input_file, 'r') as f:
            for line in f:
                doc = json.loads(line)
                if type(doc) is dict:
                    if doc['type'] != 'pdf':
                        all_chunks += self.chunk_dict(doc)
                    else:
                        try:
                            all_chunks += self.chunk_markdown(
                                doc['text'],
                                doc['source']
                            )

                        except TypeError as e:
                            print(
                                f'chunk_markdown excepted: {e}'
                            )
                elif type(doc) is list:
                    all_chunks += self.chunk_list(doc)
                else:
                    print("Don't know what to do with doc")
                    continue

        with open(output_file, 'w') as f:
            for chunk in all_chunks:
                try:
                    f.write(json.dumps(chunk) + '\n')
                except TypeError as e:
                    print(
                        f'Failed to JSONify because of {e}'
                        f'\n{chunk}'
                    )

        print(f'Created {self._chunks} chunks in {output_file}')
        return len(all_chunks)
