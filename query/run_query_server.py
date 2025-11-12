#!/usr/bin/env python3
"""Run on the Orin."""
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
