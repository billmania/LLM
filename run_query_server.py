#!/usr/bin/env python3
"""Run on the Orin."""
import sys
from pathlib import Path

from config import (
    LLM_MODEL,
    LLM_MODEL_FILE,
    MODEL_DIR,
    VECTOR_DB_DIR
)

from query.server import run_server

sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Do it."""
    llm_path = MODEL_DIR / 'models' / LLM_MODEL_FILE

    if not llm_path.exists():
        print(f'LLM not found at {llm_path}')
        print(f'Download from: https://huggingface.co/{LLM_MODEL}')
        print(f'Place the .gguf file in: {MODEL_DIR}')
        sys.exit(1)

    print('Starting query server')
    print(f'LLM: {llm_path}')
    print(f'Vector DB: {VECTOR_DB_DIR}')

    run_server(str(llm_path), VECTOR_DB_DIR)


if __name__ == '__main__':
    main()
