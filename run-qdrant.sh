#!/bin/bash
# Qdrant для локального стенда RAG-lab. Нативный бинарник, без Docker и без root.
# Документация стенда: ~/Development/Home/RAG/research/reference/local-stand.md
set -euo pipefail

STAND="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$STAND/.env"; set +a

# Qdrant создаёт служебный файл .qdrant-initialized в ТЕКУЩЕМ каталоге.
# Без этого перехода он мусорит там, откуда запустили скрипт (например в волте).
cd "$STAND"

# --- Ключ доступа --------------------------------------------------------
# Даже на localhost: без ключа любая открытая вкладка браузера может слать
# запросы на 127.0.0.1:6333 и, например, удалить коллекцию. Ключ в .env.
export QDRANT__SERVICE__API_KEY="$QDRANT_API_KEY"

# --- Безопасность: слушаем только петлю ---------------------------------
# Дефолт Qdrant — 0.0.0.0, то есть база была бы видна всей сети (кафе, коворкинг,
# офисный wi-fi) без пароля. Для проекта, чьё обещание — «данные не покидают
# контур», это недопустимо. 127.0.0.1 = доступ только с этой машины.
export QDRANT__SERVICE__HOST=127.0.0.1
export QDRANT__SERVICE__HTTP_PORT=6333
export QDRANT__SERVICE__GRPC_PORT=6334

# --- Телеметрия выключена ------------------------------------------------
# По умолчанию Qdrant шлёт анонимную статистику наружу. Приучаемся сразу:
# в контуре клиента исходящих соединений быть не должно.
export QDRANT__TELEMETRY_DISABLED=true

# --- Данные вне iCloud ---------------------------------------------------
# ~/Development не синхронизируется (в отличие от Desktop и Documents).
export QDRANT__STORAGE__STORAGE_PATH="$STAND/storage"
export QDRANT__STORAGE__SNAPSHOTS_PATH="$STAND/snapshots"

# --- Веб-интерфейс -------------------------------------------------------
# Нативный бинарник не содержит статику дашборда (в отличие от docker-образа).
# Взята отдельным релизом qdrant-web-ui, лежит в static/.
export QDRANT__SERVICE__STATIC_CONTENT_DIR="$STAND/static"

exec "$STAND/bin/qdrant"
