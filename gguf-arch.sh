#!/bin/bash
# Архитектура GGUF-сборки без скачивания файла целиком.
#
# Зачем: у моделей без официального GGUF приходится брать любительские
# конвертации, и часть из них объявляет архитектуру, которой наша сборка
# llama.cpp не знает. Проверка стоит 16 КБ и секунду, скачивание вслепую —
# сотни мегабайт и тупик. Разбор: RAG/08-local-stand.md, раздел
# «Архитектуру GGUF видно до скачивания».
#
# Использование:
#   ./gguf-arch.sh Zenabius/multilingual-e5-large-Q8_0-GGUF multilingual-e5-large-q8_0.gguf
#   -> general.architecture = bert      (годится)
#   ./gguf-arch.sh cstr/multilingual-e5-large-GGUF multilingual-e5-large-q8_0.gguf
#   -> general.architecture = xlmr      (llama.cpp b10280 не загрузит)
#
# -L обязателен: resolve/main отвечает редиректом на CDN, без него
# приходит HTML вместо файла и проверка падает на «не GGUF».
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "использование: $0 <repo> <файл.gguf>" >&2
  exit 2
fi

curl -sL -r 0-16383 "https://huggingface.co/$1/resolve/main/$2" | python3 -c '
import struct, sys
b = sys.stdin.buffer.read()
assert b[:4] == b"GGUF", "не GGUF (проверьте имя репозитория и файла)"
o = 24                                  # магия, версия, счётчики тензоров и KV
n, = struct.unpack_from("<Q", b, o); o += 8
k = b[o:o+n].decode(); o += n
t, = struct.unpack_from("<I", b, o); o += 4
l, = struct.unpack_from("<Q", b, o); o += 8
print(k, "=", b[o:o+l].decode())'
