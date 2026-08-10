"""Вызов генеративной модели со структурированным выводом. Этап 7.

Разбор — RAG/07-extraction.md, выбор модели — RAG/10-models.md,
лаба — RAG/lab-07-extraction.md.

Парный модуль к `extract.py`: тот делает всё ДО и ПОСЛЕ модели (схема,
проверка цитат, нормализация) и намеренно не знает про сервер. Этот знает
только про сервер и ничего не знает про договоры.

Три решения, ради которых модуль существует:

1. ДЕТЕРМИНИЗМ ПО УМОЛЧАНИЮ: `temperature 0` и фиксированный `seed`.
   Не ради «качества» — ради того же, ради чего кэш векторов в `embed.py`:
   без него сдвиг метрики между прогонами не отличить от настоящего
   изменения. Извлечение, дающее два разных ответа на один договор,
   нельзя ни сравнивать с эталоном, ни отлаживать.

2. СХЕМА ИДЁТ ОТДЕЛЬНО ОТ ПРОМПТА. `response_format` ограничивает вывод
   грамматикой, но в промпт не подставляется: модель схему не видит.
   Описание полей словами делает `extract.prompt_for` — здесь только
   передача. Разделение намеренное: перепутать легко, а симптом — валидный
   JSON, заполненный чем попало.

3. ОТВЕТ ВОЗВРАЩАЕТСЯ ВМЕСТЕ С ЗАМЕРАМИ (`Reply`). Задержка и число токенов
   нужны в журнале экспериментов на каждом прогоне, а не «когда понадобится»:
   добавить их задним числом означает переснять все прогоны.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

RETRIES = 3
TIMEOUT = 600          # 18 полей с цитатами — это сотни токенов вывода
SEED = 1
MAX_TOKENS = 2048


@dataclass
class Reply:
    """Ответ модели вместе с тем, во что он обошёлся."""

    data: dict                      # разобранный JSON по схеме
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    raw: str = field(repr=False, default="")

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds else 0.0


class Llm:
    def __init__(self, url=None, api_key=None, seed=SEED, temperature=0.0):
        self.url = (url or os.environ["LLM_URL"]).rstrip("/")
        self.api_key = api_key or os.environ.get("LLAMA_API_KEY", "")
        self.seed = seed
        self.temperature = temperature

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
                # 4xx — наша ошибка: кривая схема или промпт. Повтор не поможет
                if e.code < 500:
                    raise RuntimeError(f"{e.code}: {e.read().decode()[:400]}") from None
                last = e
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
            time.sleep(2**attempt)
        raise RuntimeError(f"{RETRIES} попыток не удались: {last}")

    def tokens(self, text: str) -> int:
        """Длина текста в токенах ЭТОЙ модели.

        Не берётся у эмбеддера и не считается по коэффициенту: на одном и том
        же корпусе T-lite даёт ~6 символов на токен, Qwen3-Embedding — 1,4-2,5.
        Коэффициент — свойство пары «модель + текст» (RAG/lab-03-chunking.md).
        """
        return len(self._post("/tokenize", {"content": text})["tokens"])

    def fill(self, prompt: str, schema: dict, name: str = "extraction") -> Reply:
        """Заполнить схему по промпту. Возвращает разобранный JSON и замеры.

        Грамматику строит сервер по JSON Schema, поэтому невалидный JSON
        физически невозможен — но валидный НЕ означает верный: пустая схема
        заполнится выдумкой так же охотно, как настоящая. Проверка значений —
        не здесь, а в `extract.quote_ok` и дальше.
        """
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": MAX_TOKENS,
        }
        started = time.perf_counter()
        out = self._post("/v1/chat/completions", body)
        seconds = time.perf_counter() - started

        text = out["choices"][0]["message"]["content"]
        usage = out.get("usage", {})
        finish = out["choices"][0].get("finish_reason")
        if finish == "length":
            raise RuntimeError(
                f"вывод обрезан на {MAX_TOKENS} токенах — JSON неполон. "
                "Увеличить MAX_TOKENS или сузить схему"
            )
        return Reply(
            data=json.loads(text),
            seconds=seconds,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            raw=text,
        )
