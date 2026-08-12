"""Векторизация текста через llama.cpp-сервер.

Этап 2 роадмапа. Разбор решений — RAG/research/reference/embeddings.md, лаба — RAG-labs/lab-02-embeddings.md.

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
import http.client
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

# Потолок текстов в одном запросе. На контуре партнёра берётся из
# EMBED_BATCH и ставится низким: его эмбеддер надо кормить помаленьку.
BATCH = int(os.environ.get("EMBED_BATCH", 32))
# Ретраев больше трёх и пауза длиннее не из-за сети, а из-за перезагрузок
# эмбеддера партнёра: 1+2+4+8+16 с покрывает его 4-секундный подъём
# с запасом даже при нескольких падениях подряд.
RETRIES = 5
TIMEOUT = 120

# Сколько ждать подъёма сервера после падения, секунд.
WAIT_READY = 180

# Пауза после того, как /health ответил «ok», секунд.
SETTLE = 3

# Самое консервативное «символов на токен» из замеренных: у Qwen3 медиана
# 1,99, минимум 1,46 (замер 09.08.2026, scripts/measure-tokens.py). Берём
# ниже минимума, потому что ошибка в эту сторону дробит батч на мелкие,
# а батчинг всё равно не ускоряет (замер 07.08.2026) — то есть стоит нам
# ничего. Ошибка в другую сторону роняет сервер.
CHARS_PER_TOKEN = 1.4

# Доля контекста сервера, которую разрешаем занять одним запросом.
BATCH_BUDGET = 0.7


# Префиксы — свойство модели, а не наше решение. Qwen3 обучена без них,
# e5 — с ними. Пока контур был один, разницы не было видно; с двумя контурами
# сразу одна глобальная функция обслужить обе модели не может.
PREFIX_SCHEMES = {
    "none": ("", ""),                    # Qwen3-Embedding и всё, что без префиксов
    "e5": ("query: ", "passage: "),      # multilingual-e5-*, модель партнёра
}


def scheme_for_model(model_id: str) -> str:
    """Какая схема префиксов у модели с таким идентификатором.

    Определяется по имени, потому что через API префиксы не объявляются никак.
    Ошибка здесь не даёт исключения — она даёт тихо деградировавший поиск,
    поэтому схему всегда можно задать явно через EMBEDDER_PREFIXES.
    """
    return "e5" if "e5" in model_id.lower() else "none"


def text_for(kind: str, text: str, scheme: str = "none") -> str:
    """Единственное место, где текст готовится к векторизации.

    kind: "query" или "document".

    Qwen3-Embedding поддерживает инструкции для запросов, но на нашем корпусе
    они не дали ничего (recall@5 0,790 без префикса против 0,778 и 0,741
    с двумя вариантами инструкции — шум). Поэтому для неё обе ветки одинаковы.

    Префикс входит в текст ДО хеширования, поэтому в ключ кэша он попадает сам
    и отдельно учитывать его не нужно.
    """
    if kind not in ("query", "document"):
        raise ValueError(f"kind должен быть query или document, не {kind!r}")
    if scheme not in PREFIX_SCHEMES:
        raise ValueError(f"неизвестная схема префиксов {scheme!r}")
    qp, dp = PREFIX_SCHEMES[scheme]
    return (qp if kind == "query" else dp) + text.strip()


CANARY = "проверочная строка для отпечатка сервера"


class Embedder:
    """Клиент эмбеддера. Один экземпляр — один контур.

    ВАЖНО ПРО КЭШ. Ключ кэша включает отпечаток сервера, и это не украшение.
    Qwen3 на локальном стенде и e5 у партнёра обе дают 1024 измерения — то
    есть перепутанные контуры НЕ вызовут ошибки, только тихо неверные векторы
    и бессмысленные метрики. Раньше в ключ входила строка `model`, которую
    надо было не забыть передать; из восьми мест вызова её передавало одно.
    Теперь отпечаток снимается с самого сервера и забыть его нельзя.
    """

    def __init__(self, url=None, api_key=None, cache_path=None, model=None,
                 prefixes=None):
        self.url = (url or os.environ["EMBEDDER_URL"]).rstrip("/")
        self.api_key = api_key or os.environ.get("LLAMA_API_KEY", "")
        # model задаётся явно только там, где отпечатка недостаточно:
        # compare-embedders.py поднимает одну модель с разным пулингом и
        # ходит к разным портам, различая варианты сам.
        self.model = model
        self._explicit_model = model is not None
        self._ident = None
        self._scheme = prefixes or os.environ.get("EMBEDDER_PREFIXES") or None
        self._dim = None
        self._budget = None
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

    def _ready(self) -> bool:
        """Отвечает ли сервер готовностью. 503 «Loading model» — это не ошибка,
        а «ещё гружусь»."""
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return json.load(r).get("status") == "ok"
        except Exception:
            return False

    def _wait_ready(self, limit=WAIT_READY) -> bool:
        """Ждёт подъёма сервера, опрашивая /health.

        Ждать по /health, а не спать фиксированную паузу, — принципиально.
        Эмбеддер партнёра после падения под нагрузкой возвращается за 4 с,
        но иногда дольше, и слепые паузы 1+2+4+8+16 с однажды исчерпались
        раньше, чем он догрузил модель (замер 10.08.2026).
        """
        deadline = time.time() + limit
        while time.time() < deadline:
            if self._ready():
                # Пауза на осадку. /health отвечает «ok» раньше, чем сервер
                # готов держать нагрузку: без неё следующий же запрос
                # укладывал его снова, и пять ретраев уходили в пустоту.
                time.sleep(SETTLE)
                return True
            time.sleep(2)
        return False

    def _post(self, path, body, retries=None):
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        last = None
        for attempt in range(retries or RETRIES):
            try:
                req = urllib.request.Request(self.url + path, data, headers)
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                # 503 у llama.cpp — «модель ещё грузится», ждём подъёма
                if e.code == 503:
                    last = e
                    self._wait_ready()
                    continue
                # прочие 4xx — наша ошибка, повторять бессмысленно
                if e.code < 500:
                    raise RuntimeError(f"{e.code}: {e.read().decode()[:400]}") from None
                last = e
            except (ConnectionError, http.client.HTTPException) as e:
                # Обрыв соединения без ответа. На контуре партнёра это
                # не сбой сети, а штатное поведение: его эмбеддер (сборка
                # b7176) умирает после ~6000 суммарно векторизованных токенов
                # и поднимается супервизором. Замер 10.08.2026: падает
                # на 64-72 чанках нашего корпуса независимо от размера порции
                # и порядка текстов. Пока он не обновит сборку, ретрай
                # с ожиданием — единственный способ загрузить корпус целиком.
                last = e
                self._wait_ready()
                continue
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
            time.sleep(2**attempt)
        raise RuntimeError(f"{RETRIES} попыток не удались: {last}")

    def _get(self, path):
        req = urllib.request.Request(
            self.url + path,
            headers={"Authorization": "Bearer " + self.api_key} if self.api_key else {},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.load(r)

    @property
    def dim(self) -> int:
        """Размерность вектора. Читается у сервера, не хардкодится."""
        if self._dim is None:
            self._dim = self._get("/v1/models")["data"][0]["meta"]["n_embd"]
        return self._dim

    @property
    def model_id(self) -> str:
        """Что сервер сам про себя говорит. У llama.cpp это путь к файлу."""
        return self._get("/v1/models")["data"][0]["id"]

    @property
    def budget(self) -> int:
        """Сколько токенов сервер готов принять за один запрос.

        Считается как n_ctx × число слотов с запасом. Нужно потому, что
        BATCH=32 — это про количество текстов, а сервер ограничен их суммарной
        длиной. На локальном стенде (n_ctx 8192) разницы не было видно;
        на контуре партнёра (n_ctx 512, 4 слота) батч из 32 чанков — это
        ~12 800 токенов против бюджета 2048, и запрос не отвергается,
        а обрывает соединение: 10.08.2026 его эмбеддер на этом ушёл
        в перезагрузку (вернулся сам за 4 секунды, он под супервизором).
        """
        if self._budget is None:
            try:
                p = self._get("/props")
                n_ctx = p["default_generation_settings"]["n_ctx"]
                slots = p.get("total_slots", 1)
                self._budget = max(1, int(n_ctx * slots * BATCH_BUDGET))
            except Exception:
                # /props недоступен — берём предел одного текста, он всегда верен
                self._budget = 512
        return self._budget

    @property
    def scheme(self) -> str:
        """Схема префиксов: явная, из EMBEDDER_PREFIXES или выведенная из имени."""
        if self._scheme is None:
            self._scheme = scheme_for_model(self.model_id)
        return self._scheme

    # --- отпечаток сервера ---

    def identity(self) -> str:
        """Строка, меняющаяся при любом изменении того, что сервер выдаёт.

        Собирается из трёх частей:

        1. Имя модели и версия сборки — дёшево и объясняет отпечаток человеку.
        2. Отпечаток КАНАРЕЙКИ — вектор фиксированной строки, округлённый
           до 3 знаков и хешированный. Ловит то, чего нет в метаданных:
           пулинг, квантизацию, подмену файла модели. Партнёру предстоит
           переключить пулинг с `cls` на `mean` — адрес и имя модели при этом
           не изменятся, а все векторы станут другими, и без канарейки кэш
           продолжил бы отдавать старые.
        3. Схема префиксов — на случай, если её задали явно вопреки имени.

        Округление до 3 знаков — не аккуратность, а осознанный допуск: один
        и тот же текст даёт llama.cpp векторы, отличающиеся в 7-м знаке
        (косинус 0,99999929, замер 07.08.2026). Более точный отпечаток
        менялся бы сам по себе.

        Режим отказа выбран намеренно: ложное несовпадение отпечатка стоит
        пересчёта корпуса (7 секунд на 144 чанка), ложное совпадение стоит
        неверных метрик, которые никто не заметит.
        """
        if self._ident is None:
            if self._explicit_model:
                self._ident = self.model
            else:
                try:
                    props = self._get("/props")
                    build = props.get("build_info", "?")
                except Exception:
                    build = "?"
                vec = self._post("/v1/embeddings", {"input": CANARY})["data"][0]["embedding"]
                rounded = ",".join(f"{x:.3f}" for x in vec)
                canary = hashlib.sha256(rounded.encode()).hexdigest()[:12]
                name = self.model_id.rsplit("/", 1)[-1]
                self._ident = f"{name}|{build}|{canary}|{self.scheme}"
        return self._ident

    # --- кэш ---

    def _key(self, kind, text):
        raw = f"{self.identity()}\x00{kind}\x00{text}".encode()
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

    def _batches(self, todo):
        """Режет список на запросы, влезающие в контекст сервера.

        Ограничений два и они разной природы: BATCH — про число текстов,
        budget — про их суммарную длину. Нарушение первого ничем не грозило,
        нарушение второго роняет сервер, поэтому режем по обоим.

        Текст длиннее бюджета уходит в запрос один: сервер ответит на него
        честной ошибкой (у партнёра HTTP 500 после 508 токенов), и она укажет
        на конкретный чанк, а не на батч из тридцати двух.
        """
        budget = self.budget
        cur, cur_tokens = [], 0
        for item in todo:
            est = len(item[1]) / CHARS_PER_TOKEN
            if cur and (len(cur) >= BATCH or cur_tokens + est > budget):
                yield cur
                cur, cur_tokens = [], 0
            cur.append(item)
            cur_tokens += est
        if cur:
            yield cur

    def embed(self, texts, kind="document", use_cache=True):
        """Список строк -> список векторов. Порядок сохраняется."""
        prepared = [text_for(kind, t, self.scheme) for t in texts]
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

        # Кэш пишется ПОСЛЕ КАЖДОЙ порции, а не в конце. На локальном стенде
        # разницы нет, на контуре партнёра это разница между «работает»
        # и «не работает»: его эмбеддер падает в середине корпуса, и при
        # записи в конце падение на 140-м чанке выбрасывало 139 успешных.
        # Теперь повторный запуск продолжает с места обрыва — ровно то,
        # что партнёр называет «совать помаленьку».
        for part in self._batches(todo):
            batch_fresh = self._embed_batch(part)
            if use_cache:
                self._store(batch_fresh)
            have.update(dict(batch_fresh))

        return [have[k] for k in keys]

    def _embed_batch(self, part):
        """Векторизует порцию, дробя её пополам при обрыве соединения.

        Зачем дробление, а не просто повтор. На контуре партнёра нашлись
        порции, которые роняют сервер ДЕТЕРМИНИРОВАННО: четвёрка чанков
        72-75 нашего корпуса убивает его при здоровом сервере до и после,
        тогда как та же тройка 72-74 и та же пара 74-75 проходят. Причину
        выделить не удалось — от суммы токенов она не зависит (порция
        из 8 чанков на 645 токенов проходит, эта из 4 на 478 нет), от порядка
        и от конкретного чанка тоже. Простой повтор здесь бесполезен: он
        повторяет ровно то, что не работает, и пять попыток уходят в пустоту.

        Дробление обходит проблему, не понимая её, и это осознанный размен:
        корпус грузится сегодня, а причина остаётся вопросом партнёру
        (сборка b7176 против нашей b10280, которая так не делает).
        Одиночный текст, если он всё-таки не проходит, отдаёт ошибку наверх
        с указанием на себя — её уже можно разбирать глазами.
        """
        # имя модели llama.cpp игнорирует (проверено на контуре партнёра:
        # заведомо чужое имя принимается), но поле шлём — оно обязательно
        # по спецификации OpenAI, а пайплайну предстоит ходить и к другим
        # реализациям
        body = {"model": self.model or "default", "input": [t for _, t in part]}
        # Порции больше одного текста повторяем дважды, а не пять раз: если
        # порция роняет сервер детерминированно, повторы её не спасут, а вот
        # дробление спасёт — и начать его надо быстро. Пять попыток остаются
        # одиночному тексту, где дробить уже нечего и повтор осмыслен.
        try:
            resp = self._post("/v1/embeddings", body,
                              retries=None if len(part) == 1 else 2)
        except RuntimeError:
            if len(part) == 1:
                raise
            self._wait_ready()
            mid = len(part) // 2
            return self._embed_batch(part[:mid]) + self._embed_batch(part[mid:])
        # опираемся на index, а не на порядок в data
        by_index = {d["index"]: d["embedding"] for d in resp["data"]}
        return [(k, by_index[j]) for j, (k, _) in enumerate(part)]

    def embed_one(self, text, kind="document", use_cache=True):
        return self.embed([text], kind=kind, use_cache=use_cache)[0]

    def stats(self):
        (n,) = self.db.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return {"в кэше": n, "попаданий": self.hits, "промахов": self.misses}
