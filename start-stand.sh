#!/bin/bash
# Поднять весь локальный стенд: Qdrant + эмбеддер + реранкер + модель.
# Логи — в logs/, останов — ./stop-stand.sh
#
# Генеративная модель (8084) поднимается последней и ждётся дольше прочих:
# она весит 6,26 ГБ против 0,6 у эмбеддера, и с холодного диска читается
# заметно дольше. На ПЕРВОМ запуске её здесь не будет вовсе — llama-server
# сначала скачает файл, а это часы на медленном канале; поэтому скачивание
# делается отдельно, руками: ./run-llm.sh
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
start llm      run-llm.sh      8084

echo
echo "Жду готовности (модели грузятся в память ~10–60 с)..."
for i in $(seq 1 90); do
  q=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:6333/healthz" || echo 000)
  e=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8080/health"  || echo 000)
  r=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8081/health"  || echo 000)
  l=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8084/health"  || echo 000)
  [ "$q$e$r$l" = "200200200200" ] && { echo "Готово: Qdrant + эмбеддер + реранкер + модель отвечают."; exit 0; }
  sleep 2
done

echo "Не все сервисы поднялись за 3 минуты. Статус: qdrant=$q embedder=$e reranker=$r llm=$l"
echo "Если llm=000 на первом запуске — модель ещё качается: tail -f $STAND/logs/llm.log"
echo "Смотреть логи: tail -30 $STAND/logs/*.log"
exit 1
