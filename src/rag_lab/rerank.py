"""Реранкер: пересортировка кандидатов кросс-энкодером. Этап 6.

Разбор — RAG/05-retrieval-rerank.md, лаба — RAG/lab-06-rerank.md.

Отличие от эмбеддера принципиальное, и от него всё остальное. Эмбеддер
считает вектор документа ОТДЕЛЬНО от запроса, заранее, один раз на весь
корпус. Реранкер читает пару «запрос + документ» ЦЕЛИКОМ и выдаёт одно
число — насколько этот документ отвечает на этот вопрос.

Отсюда два следствия:

  * качество выше: модель видит слова запроса рядом со словами документа
    и может сопоставить их напрямую, а не через одно общее сжатое
    представление;
  * стоимость выше на порядки: N документов = N прогонов модели, и ничего
    нельзя посчитать заранее. Поэтому реранкер стоит вторым каскадом,
    на 50 кандидатах, а не на всём корпусе.

Сервер — тот же llama-server, отдельный процесс с флагами
`--embedding --pooling rank --reranking` (см. run-reranker.sh).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

RETRIES = 3
TIMEOUT = 180


class Reranker:
    def __init__(self, url: str | None = None, api_key: str | None = None):
        self.url = (url or os.environ["RERANKER_URL"]).rstrip("/")
        self.api_key = api_key or os.environ.get("LLAMA_API_KEY", "")

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        last = None
        for attempt in range(RETRIES):
            try:
                req = urllib.request.Request(self.url + path, data, headers)
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    raise RuntimeError(f"{e.code}: {e.read().decode()[:400]}") from None
                last = e
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
            time.sleep(2**attempt)
        raise RuntimeError(f"{RETRIES} попыток не удались: {last}")

    def rank(self, query: str, documents: list[str], top_n: int | None = None):
        """(запрос, документы) -> [(индекс документа, оценка)] по убыванию.

        Возвращаются ИНДЕКСЫ входного списка, а не тексты: так вызывающий
        код сам решает, что делать с payload, и ничего не теряет по дороге.

        Порядок в ответе сервера не гарантирован — сортируем сами.
        """
        if not documents:
            return []
        body = {"model": "reranker", "query": query, "documents": documents}
        if top_n:
            body["top_n"] = top_n
        resp = self._post("/v1/rerank", body)
        pairs = [(r["index"], r["relevance_score"]) for r in resp["results"]]
        pairs.sort(key=lambda p: -p[1])
        return pairs[: top_n or len(pairs)]

    def healthy(self) -> bool:
        """Санитарная проверка: не битая ли конвертация модели.

        Признак битой сборки из RAG/05-retrieval-rerank.md — все документы
        получают одинаково ничтожную оценку порядка 1e-23 (потерянные веса
        `cls.output.weight` при конвертации в GGUF).

        Проверка отвечает на вопрос «модель жива», а НЕ «конфигурация верна»
        ([[lab-style#Правило 10. Санитарная проверка не заменяет метрику]]).
        """
        pairs = self.rank(
            "во сколько обойдётся простой машины под погрузкой",
            [
                "За каждый час простоя транспортного средства сверх нормативного "
                "времени Заказчик уплачивает 1 500 рублей за час.",
                "Хранитель обеспечивает температурный режим от плюс 5 до плюс 25 "
                "градусов Цельсия.",
            ],
        )
        best, worst = pairs[0], pairs[-1]
        return best[0] == 0 and abs(best[1] - worst[1]) > 0.1
