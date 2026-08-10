#!/bin/bash
# Остановить локальный стенд. Данные в storage/ остаются на месте.
STAND="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pkill -f "$STAND/bin/qdrant"           && echo "qdrant остановлен"   || echo "qdrant не запущен"
pkill -f "llama-server.*port 8080"     && echo "эмбеддер остановлен" || echo "эмбеддер не запущен"
pkill -f "llama-server.*port 8081"     && echo "реранкер остановлен" || echo "реранкер не запущен"
pkill -f "llama-server.*port 8084"     && echo "модель остановлена"  || echo "модель не запущена"
exit 0
