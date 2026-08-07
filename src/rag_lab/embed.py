"""Векторизация текста через llama.cpp-сервер.

Этап 2 роадмапа. Разбор решений — RAG/03-embeddings.md, лаба — RAG/lab-02-embeddings.md.

Три вещи, ради которых этот модуль существует:

1. ЕДИНАЯ функция формирования текста для запроса и документа (`text_for`).
   Забытый префикс не даёт ошибки — он даёт тихо деградировавший поиск.
   Пока префиксы не нужны (замер 07.08.2026 не показал эффекта), но место
   для них одно, и оно здесь.

2. КЭШ по хэшу текста. Обоснование не только скорость: один и тот же текст
   даёт llama.cpp два разных вектора в зависимости от состояния кэша промптов
   сервера (косинус 0,99999929). С кэшем повторная сборка корпуса даёт те же
   векторы, без него — соседние, и сдвиг метрики не отличить от настоящего.

3. РЕТРАИ. Сервер локальный, но пайплайн переедет на чужой.

Батчинг здесь есть, но ускорения он не даёт — замерено, 07.08.2026: 100 текстов
векторизуются за 3,5-4,2 с в любой комбинации. Батчим ради меньшего числа
точек отказа, а не ради скорости.
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

BATCH = 32
RETRIES = 3
TIMEOUT = 120


def text_for(kind: str, text: str) -> str:
    """Единственное место, где текст готовится к векторизации.

    kind: "query" или "document".

    Qwen3-Embedding поддерживает инструкции для запросов, но на нашем корпусе
    они не дали ничего (recall@5 0,790 без префикса против 0,778 и 0,741
    с двумя вариантами инструкции — шум). Поэтому обе ветки пока одинаковы.
    Менять модель — менять эту функцию, и только её.
    """
    if kind not in ("query", "document"):
        raise ValueError(f"kind должен быть query или document, не {kind!r}")
    return text.strip()


class Embedder:
    def __init__(self, url=None, api_key=None, cache_path=None, model="x"):
        self.url = (url or os.environ["EMBEDDER_URL"]).rstrip("/")
        self.api_key = api_key or os.environ.get("LLAMA_API_KEY", "")
        self.model = model
        self._dim = None
        path = cache_path or Path(__file__).resolve().parents[2] / "cache" / "embeddings.sqlite"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "  key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
        )
        self.db.commit()
        self.hits = self.misses = 0

    # --- сервер ---

    def _post(self, path, body):
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
                # 4xx — наша ошибка, повторять бессмысленно
                if e.code < 500:
                    raise RuntimeError(f"{e.code}: {e.read().decode()[:400]}") from None
                last = e
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
            time.sleep(2**attempt)
        raise RuntimeError(f"{RETRIES} попыток не удались: {last}")

    @property
    def dim(self) -> int:
        """Размерность вектора. Читается у сервера, не хардкодится."""
        if self._dim is None:
            req = urllib.request.Request(
                self.url + "/v1/models",
                headers={"Authorization": "Bearer " + self.api_key} if self.api_key else {},
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                meta = json.load(r)["data"][0]["meta"]
            self._dim = meta["n_embd"]
        return self._dim

    # --- кэш ---

    def _key(self, kind, text):
        raw = f"{self.model}\x00{kind}\x00{text}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _cached(self, keys):
        out = {}
        for i in range(0, len(keys), 500):
            part = keys[i : i + 500]
            q = ",".join("?" * len(part))
            for key, dim, blob in self.db.execute(
                f"SELECT key, dim, vec FROM vectors WHERE key IN ({q})", part
            ):
                a = array.array("f")
                a.frombytes(blob)
                out[key] = list(a)
        return out

    def _store(self, pairs):
        self.db.executemany(
            "INSERT OR REPLACE INTO vectors (key, dim, vec) VALUES (?, ?, ?)",
            [(k, len(v), array.array("f", v).tobytes()) for k, v in pairs],
        )
        self.db.commit()

    # --- главное ---

    def embed(self, texts, kind="document", use_cache=True):
        """Список строк -> список векторов. Порядок сохраняется."""
        prepared = [text_for(kind, t) for t in texts]
        keys = [self._key(kind, t) for t in prepared]

        have = self._cached(keys) if use_cache else {}
        self.hits += sum(1 for k in keys if k in have)

        todo = []
        seen = set()
        for k, t in zip(keys, prepared):
            if k not in have and k not in seen:
                seen.add(k)
                todo.append((k, t))
        self.misses += len(todo)

        fresh = []
        for i in range(0, len(todo), BATCH):
            part = todo[i : i + BATCH]
            resp = self._post("/v1/embeddings", {"model": self.model, "input": [t for _, t in part]})
            # опираемся на index, а не на порядок в data
            by_index = {d["index"]: d["embedding"] for d in resp["data"]}
            for j, (k, _) in enumerate(part):
                fresh.append((k, by_index[j]))

        if fresh:
            if use_cache:
                self._store(fresh)
            have.update(dict(fresh))

        return [have[k] for k in keys]

    def embed_one(self, text, kind="document", use_cache=True):
        return self.embed([text], kind=kind, use_cache=use_cache)[0]

    def stats(self):
        (n,) = self.db.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return {"в кэше": n, "попаданий": self.hits, "промахов": self.misses}
