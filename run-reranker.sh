#!/bin/bash
# Реранкер для локального стенда RAG-lab. llama.cpp, порт 8081.
# Отдельный процесс: эмбеддер и реранкер в одном llama-server не совмещаются.
set -euo pipefail

STAND="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$STAND/.env"; set +a
export LLAMA_CACHE="$STAND/models"

# bge-reranker-v2-m3: мультиязычный, Apache 2.0, отлаженный путь запуска.
# Обязательны все три флага: --embedding, --pooling rank, --reranking.
exec llama-server \
  -hf gpustack/bge-reranker-v2-m3-GGUF:Q8_0 \
  --embedding --pooling rank --reranking \
  --host 127.0.0.1 --port 8081 \
  --api-key "$LLAMA_API_KEY" \
  --cors-origins localhost
