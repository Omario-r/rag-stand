#!/bin/bash
# multilingual-e5-large — модель партнёра, поднимается рядом со штатным
# эмбеддером ради сравнительных замеров. В пайплайне НЕ используется.
#
# Использование: ./run-embedder-e5.sh [пулинг] [порт]
#   ./run-embedder-e5.sh mean 8082    (по умолчанию)
#   ./run-embedder-e5.sh cls  8083
#
# Почему именно сборка Zenabius (замер 08.08.2026, RAG/research/reference/local-stand.md):
#   Официального GGUF у e5 не существует — intfloat публикует только
#   safetensors. Из любительских конвертаций годятся объявляющие
#   архитектуру `bert`; свежая сборка cstr объявляет `xlmr`, которую
#   llama.cpp b10280 не знает, и не загружается вовсе.
#   Проверять архитектуру ДО скачивания: ./gguf-arch.sh <repo> <файл.gguf>
#
# -c 512 — архитектурный предел e5 (XLM-R), а не наш выбор. Ставим честно,
#   чтобы обрезание длинных текстов было видно, а не замаскировано.
#
# --pooling обязателен и передаётся явно: e5 обучена с mean-пулингом,
#   а cls на нашем золотом наборе даёт recall@5 0,741 против 0,815.
#   Это самый большой разрыв из всех измеренных — молча полагаться
#   на метаданные GGUF нельзя.
#
# ВНИМАНИЕ: не вставлять пустых строк между строками с обратным слешем —
# команда молча оборвётся, а лишние аргументы потеряются. Это уже случалось.
set -euo pipefail

POOLING="${1:-mean}"
PORT="${2:-8082}"

STAND="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$STAND/.env"; set +a
export LLAMA_CACHE="$STAND/models"      # модели рядом со стендом, не в ~/Library

echo "e5, пулинг $POOLING, порт $PORT"

exec llama-server \
  -hf Zenabius/multilingual-e5-large-Q8_0-GGUF:q8_0 \
  --embeddings \
  --host 127.0.0.1 --port "$PORT" \
  --api-key "$LLAMA_API_KEY" \
  --cors-origins localhost \
  --pooling "$POOLING" \
  --parallel 1 \
  -c 512
