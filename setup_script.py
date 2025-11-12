# config.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

# Processing settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
PDF_MIN_WORDS = 50  # Skip PDFs with fewer words (likely scans)

# Model settings
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_BATCH_SIZE = 32  # Adjust for GTX 1060 VRAM
LLM_MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
LLM_MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Vector DB settings
QDRANT_COLLECTION = "documents"
TOP_K_RESULTS = 5

# ============================================================================
# indexing/extract.py - Text extraction from various formats
# ============================================================================

import mailbox
import email
from email import policy
import pymupdf  # PyMuPDF
from docx import Document
from odf import text as odf_text
from odf.opendocument import load as odf_load
import json
from pathlib import Path
from typing import Dict, List
import re

class DocumentExtractor:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_pdf(self, pdf_path: Path) -> Dict:
        """Extract text from PDF, skip if appears to be scanned"""
        doc = pymupdf.open(pdf_path)
        
        # Check first page for text content
        first_page_text = doc[0].get_text()
        word_count = len(first_page_text.split())
        
        if word_count < PDF_MIN_WORDS:
            doc.close()
            return None  # Skip scanned documents
        
        # Extract all text
        text = ""
        for page in doc:
            text += page.get_text()
        
        doc.close()
        
        return {
            "source": str(pdf_path),
            "type": "pdf",
            "text": text.strip(),
            "metadata": {"pages": len(doc)}
        }
    
    def extract_text_file(self, txt_path: Path) -> Dict:
        """Extract text from plain text file"""
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        return {
            "source": str(txt_path),
            "type": "text",
            "text": text.strip(),
            "metadata": {}
        }
    
    def extract_docx(self, docx_path: Path) -> Dict:
        """Extract text from DOCX"""
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        
        return {
            "source": str(docx_path),
            "type": "docx",
            "text": text.strip(),
            "metadata": {}
        }
    
    def extract_odt(self, odt_path: Path) -> Dict:
        """Extract text from ODT"""
        doc = odf_load(str(odt_path))
        paragraphs = doc.getElementsByType(odf_text.P)
        text = "\n".join([str(p) for p in paragraphs])
        
        return {
            "source": str(odt_path),
            "type": "odt",
            "text": text.strip(),
            "metadata": {}
        }
    
    def clean_email_body(self, body: str) -> str:
        """Remove quoted replies and excess whitespace"""
        lines = body.split('\n')
        cleaned = []
        
        for line in lines:
            # Skip common quote markers
            if line.startswith('>') or line.startswith('On ') and 'wrote:' in line:
                break
            cleaned.append(line)
        
        return '\n'.join(cleaned).strip()
    
    def extract_mbox(self, mbox_path: Path) -> List[Dict]:
        """Extract messages from mbox file"""
        mbox = mailbox.mbox(str(mbox_path))
        messages = []
        
        for idx, message in enumerate(mbox):
            try:
                subject = message.get('Subject', 'No Subject')
                sender = message.get('From', 'Unknown')
                date = message.get('Date', '')
                
                # Extract body (prefer plain text)
                body = ""
                if message.is_multipart():
                    for part in message.walk():
                        if part.get_content_type() == 'text/plain':
                            body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            break
                else:
                    body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
                
                body = self.clean_email_body(body)
                
                if body:  # Only include messages with content
                    messages.append({
                        "source": f"{mbox_path.name}::{idx}",
                        "type": "email",
                        "text": body,
                        "metadata": {
                            "subject": subject,
                            "from": sender,
                            "date": date
                        }
                    })
            except Exception as e:
                print(f"Error processing message {idx} in {mbox_path}: {e}")
                continue
        
        return messages
    
    def process_all(self, raw_dir: Path):
        """Process all documents and save to processed directory"""
        all_docs = []
        
        # Process PDFs
        print("Processing PDFs...")
        for pdf_path in (raw_dir / "pdfs").glob("*.pdf"):
            doc = self.extract_pdf(pdf_path)
            if doc:
                all_docs.append(doc)
        
        # Process text files
        print("Processing text files...")
        for txt_path in (raw_dir / "docs").glob("*.txt"):
            all_docs.append(self.extract_text_file(txt_path))
        
        # Process DOCX
        print("Processing DOCX files...")
        for docx_path in (raw_dir / "docs").glob("*.docx"):
            try:
                all_docs.append(self.extract_docx(docx_path))
            except Exception as e:
                print(f"Error processing {docx_path}: {e}")
        
        # Process ODT
        print("Processing ODT files...")
        for odt_path in (raw_dir / "docs").glob("*.odt"):
            try:
                all_docs.append(self.extract_odt(odt_path))
            except Exception as e:
                print(f"Error processing {odt_path}: {e}")
        
        # Process mbox files
        print("Processing mbox files...")
        for mbox_path in (raw_dir / "mbox").glob("*.mbox"):
            messages = self.extract_mbox(mbox_path)
            all_docs.extend(messages)
        
        # Save all processed documents
        output_file = self.processed_dir / "extracted_documents.jsonl"
        with open(output_file, 'w') as f:
            for doc in all_docs:
                f.write(json.dumps(doc) + '\n')
        
        print(f"Processed {len(all_docs)} documents -> {output_file}")
        return len(all_docs)

# ============================================================================
# indexing/chunk.py - Text chunking with metadata preservation
# ============================================================================

from typing import List, Dict
import json

class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, doc_metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "metadata": doc_metadata
            })
        
        return chunks
    
    def process_documents(self, input_file: Path, output_file: Path):
        """Chunk all documents and save"""
        all_chunks = []
        
        with open(input_file, 'r') as f:
            for line in f:
                doc = json.loads(line)
                
                # Create metadata for this document
                metadata = {
                    "source": doc["source"],
                    "type": doc["type"],
                    **doc.get("metadata", {})
                }
                
                # Chunk the document
                chunks = self.chunk_text(doc["text"], metadata)
                all_chunks.extend(chunks)
        
        # Save chunks
        with open(output_file, 'w') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk) + '\n')
        
        print(f"Created {len(all_chunks)} chunks -> {output_file}")
        return len(all_chunks)

# ============================================================================
# indexing/embed.py - Generate embeddings and populate vector DB
# ============================================================================

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import torch
from tqdm import tqdm

class EmbeddingIndexer:
    def __init__(self, model_name: str, db_path: Path, collection_name: str):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Initialize Qdrant
        self.client = QdrantClient(path=str(db_path))
        self.collection_name = collection_name
        
        # Get embedding dimension
        test_embedding = self.model.encode("test")
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
            print(f"Created collection: {collection_name}")
        except Exception as e:
            print(f"Collection may already exist: {e}")
    
    def embed_and_index(self, chunks_file: Path, batch_size: int = 32):
        """Generate embeddings and index all chunks"""
        chunks = []
        with open(chunks_file, 'r') as f:
            for line in f:
                chunks.append(json.loads(line))
        
        print(f"Indexing {len(chunks)} chunks...")
        
        # Process in batches
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]
            
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
                        "text": chunk["text"],
                        "metadata": chunk["metadata"]
                    }
                ))
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        print(f"Indexed {len(chunks)} chunks successfully")

# ============================================================================
# query/search.py - Vector search
# ============================================================================

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class VectorSearcher:
    def __init__(self, model_name: str, db_path: Path, collection_name: str):
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(path=str(db_path))
        self.collection_name = collection_name
    
    def search(self, query: str, top_k: int = 5):
        """Search for relevant chunks"""
        # Embed query
        query_embedding = self.model.encode(query)
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        return results

# ============================================================================
# query/generate.py - LLM response generation
# ============================================================================

from llama_cpp import Llama

class ResponseGenerator:
    def __init__(self, model_path: str):
        print(f"Loading LLM from {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # Offload all to GPU
            verbose=False
        )
    
    def generate(self, query: str, context_chunks: List[str]) -> str:
        """Generate response using retrieved context"""
        # Build context
        context = "\n\n".join([f"Document {i+1}:\n{chunk}" 
                              for i, chunk in enumerate(context_chunks)])
        
        # Build prompt
        prompt = f"""Based on the following documents, answer the question. If the answer cannot be found in the documents, say so.

Documents:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        response = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["Question:", "\n\n"]
        )
        
        return response['choices'][0]['text'].strip()

# ============================================================================
# query/server.py - Simple query interface
# ============================================================================

from flask import Flask, request, jsonify, render_template_string
import sys

app = Flask(__name__)

# Initialize components (set these in main)
searcher = None
generator = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Base Query</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        input[type="text"] { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; }
        .result { margin-top: 20px; padding: 20px; background: #f5f5f5; border-radius: 5px; }
        .sources { margin-top: 20px; font-size: 14px; color: #666; }
    </style>
</head>
<body>
    <h1>Knowledge Base Query</h1>
    <form id="queryForm">
        <input type="text" id="query" placeholder="Enter your question..." />
        <button type="submit">Search</button>
    </form>
    <div id="result"></div>
    
    <script>
        document.getElementById('queryForm').onsubmit = async (e) => {
            e.preventDefault();
            const query = document.getElementById('query').value;
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = 'Searching...';
            
            const response = await fetch('/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            });
            
            const data = await response.json();
            
            let html = `<div class="result"><strong>Answer:</strong><br>${data.answer}</div>`;
            html += '<div class="sources"><strong>Sources:</strong><ul>';
            data.sources.forEach(s => {
                html += `<li>${s.source} (score: ${s.score.toFixed(3)})</li>`;
            });
            html += '</ul></div>';
            
            resultDiv.innerHTML = html;
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({'error': 'No query provided'}), 400
    
    # Search for relevant chunks
    results = searcher.search(query_text, top_k=TOP_K_RESULTS)
    
    # Extract text and sources
    context_chunks = [r.payload['text'] for r in results]
    sources = [{'source': r.payload['metadata']['source'], 
                'score': r.score} for r in results]
    
    # Generate answer
    answer = generator.generate(query_text, context_chunks)
    
    return jsonify({
        'answer': answer,
        'sources': sources
    })

def run_server(model_path: str, db_path: Path):
    global searcher, generator
    
    searcher = VectorSearcher(EMBEDDING_MODEL, db_path, QDRANT_COLLECTION)
    generator = ResponseGenerator(model_path)
    
    print("Starting server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)

# ============================================================================
# REQUIREMENTS.txt
# ============================================================================
"""
pymupdf>=1.23.0
python-docx>=1.0.0
odfpy>=1.4.1
sentence-transformers>=2.2.2
qdrant-client>=1.7.0
llama-cpp-python>=0.2.0
torch>=2.0.0
tqdm>=4.65.0
flask>=3.0.0
"""

# ============================================================================
# run_indexing.py - Run on desktop with GTX 1060
# ============================================================================
"""
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from indexing.extract import DocumentExtractor
from indexing.chunk import TextChunker
from indexing.embed import EmbeddingIndexer

def main():
    print("=== Document Indexing Pipeline ===")
    
    # Step 1: Extract text from documents
    print("\n[1/3] Extracting text from documents...")
    extractor = DocumentExtractor(PROCESSED_DIR)
    num_docs = extractor.process_all(RAW_DIR)
    
    # Step 2: Chunk documents
    print("\n[2/3] Chunking documents...")
    chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    num_chunks = chunker.process_documents(
        PROCESSED_DIR / "extracted_documents.jsonl",
        PROCESSED_DIR / "chunks.jsonl"
    )
    
    # Step 3: Generate embeddings and index
    print("\n[3/3] Generating embeddings and indexing...")
    indexer = EmbeddingIndexer(
        EMBEDDING_MODEL,
        VECTOR_DB_DIR,
        QDRANT_COLLECTION
    )
    indexer.embed_and_index(
        PROCESSED_DIR / "chunks.jsonl",
        batch_size=EMBEDDING_BATCH_SIZE
    )
    
    print(f"\n=== Indexing Complete ===")
    print(f"Documents processed: {num_docs}")
    print(f"Chunks created: {num_chunks}")
    print(f"Vector DB location: {VECTOR_DB_DIR}")
    print(f"\nNext: Copy {VECTOR_DB_DIR} to your Orin's USB SSD")

if __name__ == "__main__":
    main()
"""

# ============================================================================
# run_query_server.py - Run on Orin
# ============================================================================
"""
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from query.server import run_server

def main():
    # Download LLM if needed
    llm_path = PROJECT_ROOT / "models" / LLM_MODEL_FILE
    
    if not llm_path.exists():
        print(f"LLM not found at {llm_path}")
        print(f"Download from: https://huggingface.co/{LLM_MODEL}")
        print(f"Place the .gguf file in: {llm_path.parent}")
        sys.exit(1)
    
    print("Starting query server...")
    print(f"LLM: {llm_path}")
    print(f"Vector DB: {VECTOR_DB_DIR}")
    
    run_server(str(llm_path), VECTOR_DB_DIR)

if __name__ == "__main__":
    main()
"""
