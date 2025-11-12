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

