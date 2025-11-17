"""Run the server."""
from pathlib import Path

from config import (
    CLEAR_CONTEXT,
    EMBEDDING_MODEL,
    NO_RAG,
    QDRANT_COLLECTION,
    RELEVANCE_THRESHOLD,
    TOP_K_RESULTS
)

from flask import Flask, jsonify, render_template_string, request

from query.generate import ResponseGenerator
from query.search import VectorSearcher

app = Flask(__name__)

searcher = None
generator = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Base Query</title>
    <style>
        body { font-family: Arial; max-width: 800px;
               margin: 50px auto; padding: 20px; }
        input[type="text"] { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; }
        .result { margin-top: 20px; padding: 20px;
                  background: #f5f5f5; border-radius: 5px; }
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

            let html = `<div class="result"><strong>Answer:</strong>`;
            html += `<br>${data.answer}</div>`;
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
    """Create the index.html file."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/query', methods=['POST'])
def query():
    """Process the query.

    If the query begins with the NO_RAG sequence, don't
    add any local context to it before generating an answer.

    If the query begins with the CLEAR_CONTEXT sequence,
    clear the LLM's context before generating an answer.

    Otherwise, retrieve local context and send it with the query
    to the LLM.
    """
    data = request.json
    query_text = data.get('query', '')

    if not query_text:
        return jsonify({'error': 'No query provided'}), 400

    try:
        clear_context = True if query_text.find(CLEAR_CONTEXT) == 0 else False
        if clear_context:
            query_text = query_text[len(CLEAR_CONTEXT):]

    except Exception as e:
        print(
            f'Exception while checking for CLEAR_CONTEXT: {e}'
        )
        return jsonify({
            'answer': '',
            'sources': []
        })

    try:
        if query_text.find(NO_RAG) == 0:
            context_chunks = []
            sources = []
            query_text = query_text[len(NO_RAG):]
        else:
            results = searcher.search(query_text, top_k=TOP_K_RESULTS)
            relevant_results = [
                result for result in results
                if result.score > RELEVANCE_THRESHOLD
            ]

            if not relevant_results:
                print(
                    'No relevant results'
                    f' (best score: {results[0].score if results else 0})'
                )
                answer = generator.generate(query_text, [], clear_context)
                return jsonify({
                    'answer': answer,
                    'sources': [],
                    'note': 'Answer generated without document context'
                })

            context_chunks = [
                result.payload['text'] for result in relevant_results
            ]
            sources = [{'source': result.payload['metadata']['source'],
                        'score': result.score} for result in relevant_results]

    except Exception as e:
        print(
            f'Exception while checking for NO_RAG: {e}'
        )
        return jsonify({
            'answer': '',
            'sources': []
        })

    answer = generator.generate(query_text, context_chunks, clear_context)
    print(f'Answer: {answer}')

    return jsonify({
        'answer': answer,
        'sources': sources
    })


def run_server(model_path: str, db_path: Path):
    """Run the server."""
    global searcher, generator

    searcher = VectorSearcher(EMBEDDING_MODEL, db_path, QDRANT_COLLECTION)
    generator = ResponseGenerator(model_path)

    print('Starting server on http://localhost:5000')
    app.run(host='0.0.0.0', port=5000)
