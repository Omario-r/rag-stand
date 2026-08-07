#!/bin/bash
# Проверка стенда до всякого кода. Пять минут, экономят часы отладки.
# Порядок проверок — из ~/Development/Home/RAG/08-local-stand.md
set -uo pipefail

STAND="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$STAND/.env"; set +a

echo "──────────────────────────────────────────────────────────"
echo " 1. Сервисы отвечают"
echo "──────────────────────────────────────────────────────────"
printf "  Qdrant   %s\n" "$(curl -s -o /dev/null -w '%{http_code}' -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/healthz")"
printf "  эмбеддер %s\n" "$(curl -s -o /dev/null -w '%{http_code}' "$EMBEDDER_URL/health")"
printf "  реранкер %s\n" "$(curl -s -o /dev/null -w '%{http_code}' "$RERANKER_URL/health")"

echo
echo "──────────────────────────────────────────────────────────"
echo " 2. Безопасность: без ключа доступа быть не должно"
echo "──────────────────────────────────────────────────────────"
q=$(curl -s -o /dev/null -w '%{http_code}' "$QDRANT_URL/collections")
e=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$EMBEDDER_URL/v1/embeddings" \
      -H 'Content-Type: application/json' -d '{"model":"x","input":["t"]}')
[ "$q" = "401" ] && echo "  Qdrant без ключа:   401 ✅" || echo "  Qdrant без ключа:   $q ⚠️ ОТКРЫТ"
[ "$e" = "401" ] && echo "  эмбеддер без ключа: 401 ✅" || echo "  эмбеддер без ключа: $e ⚠️ ОТКРЫТ"

ip=$(ipconfig getifaddr en0 2>/dev/null)
if [ -n "$ip" ]; then
  curl -s --max-time 3 -o /dev/null "http://$ip:6333/healthz" \
    && echo "  из локальной сети:  ДОСТУПЕН ⚠️" \
    || echo "  из локальной сети:  недоступен ✅ (проверено с $ip)"
fi

echo
echo "──────────────────────────────────────────────────────────"
echo " 3. Эмбеддер: размерность и смысл"
echo "──────────────────────────────────────────────────────────"
python3 - <<'PY'
import json, os, urllib.request

URL = os.environ["EMBEDDER_URL"] + "/v1/embeddings"
KEY = os.environ["LLAMA_API_KEY"]

def embed(texts):
    req = urllib.request.Request(
        URL,
        data=json.dumps({"model": "x", "input": texts}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)["data"]
    return [x["embedding"] for x in sorted(d, key=lambda x: x["index"])]

def cos(a, b):
    return sum(x*y for x, y in zip(a, b)) / ((sum(x*x for x in a)**.5) * (sum(y*y for y in b)**.5))

# Два текста про одно и то же РАЗНЫМИ словами + один про другое.
# Разные слова принципиальны: одинаковые превратили бы проверку смысла
# в проверку совпадения подстрок.
a = "Перевозчик уплачивает неустойку 0,1% от стоимости перевозки за каждый день просрочки доставки груза."
b = "За задержку доставки с исполнителя взыскивается пеня в размере одной десятой процента в сутки."
c = "Арендатор обязан поддерживать помещение в надлежащем санитарном состоянии."

va, vb, vc = embed([a, b, c])
print(f"  размерность: {len(va)}")
print(f"  норма:       {sum(x*x for x in va)**.5:.4f}  (1.0 = нормализован)")
print()
print(f"  похожие по смыслу (неустойка/пеня):  {cos(va, vb):.4f}")
print(f"  разные по смыслу  (неустойка/аренда): {cos(va, vc):.4f}")
d = cos(va, vb) - cos(va, vc)
print(f"  разрыв: {d:.4f}", "✅" if d > 0.1 else "⚠️ подозрительно мал — проверить pooling и префиксы")
PY

echo
echo "──────────────────────────────────────────────────────────"
echo " 4. Реранкер: различает релевантное и нет"
echo "──────────────────────────────────────────────────────────"
python3 - <<'PY'
import json, os, urllib.request

URL = os.environ["RERANKER_URL"] + "/v1/rerank"
KEY = os.environ["LLAMA_API_KEY"]

docs = [
    "5.3. Сверхнормативный простой транспортного средства под погрузкой оплачивается Заказчиком по ставке 1 500 рублей за час.",
    "9.1. Все споры по настоящему договору разрешаются в Арбитражном суде города Москвы.",
]
req = urllib.request.Request(
    URL,
    data=json.dumps({"model": "r", "query": "кто платит за простой машины под погрузкой",
                     "documents": docs}).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
)
with urllib.request.urlopen(req, timeout=120) as r:
    res = json.load(r)["results"]

for x in sorted(res, key=lambda x: -x["relevance_score"]):
    print(f"  {x['relevance_score']:+.4f}  {docs[x['index']][:62]}…")

s = sorted(x["relevance_score"] for x in res)
print()
if abs(s[-1]) < 1e-6 and abs(s[0]) < 1e-6:
    print("  ⚠️ оба скора ничтожны — битая конвертация GGUF (см. 05-retrieval-rerank.md)")
elif s[-1] > s[0]:
    print(f"  разрыв: {s[-1]-s[0]:.4f} ✅ релевантный документ выше")
else:
    print("  ⚠️ реранкер не различает документы")
PY

echo
echo "──────────────────────────────────────────────────────────"
echo " 5. Предельная длина текста"
echo "──────────────────────────────────────────────────────────"
python3 - <<'PY'
import json, os, urllib.request, urllib.error

BASE = os.environ["EMBEDDER_URL"]
KEY  = os.environ["LLAMA_API_KEY"]
AUTH = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

def get(path):
    try:
        with urllib.request.urlopen(urllib.request.Request(BASE + path, headers=AUTH), timeout=30) as r:
            return json.load(r)
    except Exception:
        return {}

# Что сервер говорит о себе сам. Единственный честный источник: скрипт запуска
# может расходиться с реальностью (см. 08-local-stand.md, раздел про грабли).
p = get("/props")
if p:
    print(f"  модель:   {p.get('model_alias', '?')} ({p.get('model_ftype', '?')})")
    print(f"  сборка:   {p.get('build_info', '?')}")
    print(f"  n_ctx:    {p.get('default_generation_settings', {}).get('n_ctx', '?')}"
          f"   слотов: {p.get('total_slots', '?')}")
    print()

# Предел длины одного текста задаёт n_ctx (проверено 06.08.2026: n_ubatch на это
# не влияет). Проверяем не «сколько объявлено», а сколько проходит на самом деле:
# контекстуализированные чанки и родители small-to-big длиннее обычных.
clause = ("Договор перевозки груза № ТР-114/2026. Раздел 4. Ответственность сторон. "
          "За просрочку доставки груза Перевозчик уплачивает неустойку 0,1 процента "
          "от стоимости перевозки за каждый день просрочки. ")

def probe(mult):
    """Вернуть (ok, число_токенов)."""
    body = json.dumps({"model": "x", "input": [clause * mult]}).encode()
    req = urllib.request.Request(BASE + "/v1/embeddings", data=body, headers=AUTH)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return True, json.load(r).get("usage", {}).get("prompt_tokens", 0)
    except urllib.error.HTTPError as e:
        try:
            return False, json.loads(e.read())["error"].get("n_prompt_tokens", 0)
        except Exception:
            return False, 0

MIN_USABLE = 2000     # ниже этого родительские разделы для small-to-big не влезут

last_ok = 0
for mult in (7, 21, 60, 110):
    ok, tokens = probe(mult)
    print(f"  {tokens:>6} токенов → {'✅ обработан' if ok else '❌ отвергнут'}")
    if ok:
        last_ok = tokens
    else:
        break

print()
if last_ok == 0:
    print("  ⚠️ не прошёл даже короткий текст — смотреть логи эмбеддера")
elif last_ok < MIN_USABLE:
    print(f"  ⚠️ предел около {last_ok} токенов — маловато.")
    print("     Контекстуализированные чанки пройдут, а родительские разделы")
    print("     для small-to-big могут не влезть. Поднять -c в run-embedder.sh")
    print("     (цена вопроса — таблица памяти там же в комментарии).")
else:
    print(f"  ✅ предел не ниже {last_ok} токенов — хватает и на чанки с шапкой,")
    print("     и на родительские разделы для small-to-big")
PY
echo
