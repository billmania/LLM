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

