#!/usr/bin/env bash

export ORIN_BASE_DIR="orin:/home/bill/projects/LLM"

cd /home/bill/projects/LLM || exit 1

rsync -rvc config.py requirements_orin.txt run_query_server.py  ${ORIN_BASE_DIR}
rsync -rvc --delete query ${ORIN_BASE_DIR}
