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

    def generate(
        self,
        query: str,
        context_chunks: List[str],
        clear_model_context: bool = False
    ) -> str:
        """Generate response using retrieved context."""
        context = '\n\n'.join([f'Document {i+1}:\n{chunk}'
                              for i, chunk in enumerate(context_chunks)])

        prompt = (
            f"""[INST] Based on the following documents, answer the question
briefly.

{context}

Question: {query} [/INST]"""
        )

        if clear_model_context:
            print('Resetting the model')
            self.llm.reset()

        print(
            f'Context size: {self.llm.n_ctx()}\n'
            f'Embed size: {self.llm.n_embd()}\n'
            f'Vocabulary size: {self.llm.n_vocab()}\n'
        )

        response = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=['[/INST]', '</s>']
        )

        return response['choices'][0]['text'].strip()
