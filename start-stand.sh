#!/bin/bash
# Поднять весь локальный стенд: Qdrant + эмбеддер + реранкер.
# Логи — в logs/, останов — ./stop-stand.sh
set -euo pipefail

STAND="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$STAND/logs"

start() {  # $1 = имя, $2 = скрипт, $3 = порт
  if lsof -nP -iTCP:"$3" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  $1: уже слушает :$3, пропускаю"
    return
  fi
  nohup "$STAND/$2" > "$STAND/logs/$1.log" 2>&1 &
  echo "  $1: запущен (pid $!), лог logs/$1.log"
}

echo "Поднимаю стенд:"
start qdrant   run-qdrant.sh   6333
start embedder run-embedder.sh 8080
start reranker run-reranker.sh 8081

echo
echo "Жду готовности (модели грузятся в память ~10–30 с)..."
for i in $(seq 1 60); do
  q=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:6333/healthz" || echo 000)
  e=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8080/health"  || echo 000)
  r=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8081/health"  || echo 000)
  [ "$q$e$r" = "200200200" ] && { echo "Готово: Qdrant + эмбеддер + реранкер отвечают."; exit 0; }
  sleep 2
done

echo "Не все сервисы поднялись за 2 минуты. Статус: qdrant=$q embedder=$e reranker=$r"
echo "Смотреть логи: tail -30 $STAND/logs/*.log"
exit 1
