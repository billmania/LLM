"""Generate responses."""
from typing import List

from llama_cpp import Llama


class ResponseGenerator:
    """Generate a response."""

    def __init__(self, model_path: str):
        """Initialize the attributes."""
        print(f'Loading LLM from {model_path}')
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # Offload all to GPU
            verbose=False
        )

    def generate(self, query: str, context_chunks: List[str]) -> str:
        """Generate response using retrieved context."""
        context = '\n\n'.join([f'Document {i+1}:\n{chunk}'
                              for i, chunk in enumerate(context_chunks)])

        prompt = (
            f"""[INST] Based on the following documents, answer the question
briefly.

{context}

Question: {query} [/INST]"""
        )

        print('=== FULL CONTEXT BEING SENT ===')
        print(context)
        print('=== END CONTEXT ===')
        print('"\nTotal prompt length: {len(prompt)} characters')

        response = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=['[/INST]', '</s>']
        )

        return response['choices'][0]['text'].strip()
