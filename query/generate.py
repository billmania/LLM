"""Generate responses."""
from typing import List

from config import MAX_MODEL_CONTEXT

from llama_cpp import Llama


class ResponseGenerator:
    """Generate a response."""

    def __init__(self, model_path: str):
        """Initialize the attributes."""
        print(f'Instantiating LLM from {model_path}')
        self.llm = Llama(
            model_path=model_path,
            n_ctx=MAX_MODEL_CONTEXT,
            n_gpu_layers=-1,
            verbose=False
        )
        print(
            f'Model vocabulary: {self.llm.n_vocab()}'
            f', Context: {self.llm.n_ctx()}'
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

        # TODO: Determine if the "duplicate <s>" is a bug
        # context_begin = '<s>' if clear_model_context else ''
        context_begin = ''

        prompt = (
            f'{context_begin}'
            f"""[INST] Based on the following documents, answer the question
briefly. If an answer does not exist, clearly state that.
{context}

Question:
{query}[/INST]"""
        )

        print(
            'Prompt:\n'
            f'{prompt}'
        )
        if clear_model_context:
            print('Resetting the model')
            # TODO: Determine if this really clears the context from the LLM
            self.llm.reset()

        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.7,
            stop=['[/INST]']
        )

        if response['choices'][0]['text'].strip().find('##########') == 0:
            print(
                "Model didn't find an answer"
                f', Metadata {self.llm.metadata}'
            )

            print('response members')
            for key in response:
                print(f'{key}: {response[key]}\n')
            return 'No answer generated'

        return response['choices'][0]['text'].strip()
