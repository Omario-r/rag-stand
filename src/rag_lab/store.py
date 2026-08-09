"""Коллекция Qdrant: создание, загрузка чанков, доведение до индекса.

Этап 4 роадмапа. Разбор решений — RAG/04-qdrant.md, лаба — RAG/lab-04-loading.md.

Четыре вещи, ради которых этот модуль существует:

1. ИДЕМПОТЕНТНОСТЬ. Идентификатор пойнта — UUID5 от `<doc_id>#<номер пункта>`,
   а не счётчик. Повторная загрузка того же договора перезаписывает свои
   пойнты, а не плодит дубли. При отладке чанкера корпус перезаливается
   десятки раз, и без этого коллекция быстро перестаёт значить что-либо.

2. ПОРОГ ИНДЕКСАЦИИ ВОЗВРАЩАЕТСЯ. `indexing_threshold=0` на время загрузки —
   главный приём ускорения, и он же главная ловушка: забытый ноль оставляет
   коллекцию без HNSW, то есть на полном переборе. На 144 пойнтах разницы
   не видно, на корпусе клиента это отказ. Поэтому возврат порога не
   «шаг инструкции», а функция `finish_indexing`, которая ещё и дожидается
   зелёного статуса.

3. PAYLOAD-ИНДЕКСЫ СОЗДАЮТСЯ ВМЕСТЕ С КОЛЛЕКЦИЕЙ. Фильтр работает и без них,
   только медленно и с полным перебором payload. Создавать их отдельным
   шагом «когда понадобится» — значит не создать никогда.

4. РАЗМЕРНОСТЬ ЧИТАЕТСЯ У ЭМБЕДДЕРА. Не константа: сменится модель —
   сменится размерность, а коллекция с неверной размерностью не создастся
   вовсе (изменить её потом нельзя, только пересоздать).
"""

from __future__ import annotations

import os
import time
import uuid

from qdrant_client import QdrantClient, models

COLLECTION = "contracts_v1"

# Пространство имён для UUID5. Фиксированное и общее со сквозным прогоном
# лаб 2: идентификатор пойнта обязан зависеть только от адреса чанка.
NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# Рабочее значение порога индексации, килобайты. Сегменты меньше порога
# не индексируются — на малых объёмах полный перебор быстрее графа.
INDEXING_THRESHOLD = 20000

# Поля payload, по которым будем фильтровать. Тип не косметика: `keyword`
# ищет по точному совпадению значения целиком, `text` — по вхождению слова.
# Для doc_id нужен именно keyword (RAG/04-qdrant.md).
PAYLOAD_INDEXES = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "part": models.PayloadSchemaType.KEYWORD,
    "section_no": models.PayloadSchemaType.KEYWORD,
    "clause_no": models.PayloadSchemaType.KEYWORD,
    "has_table": models.PayloadSchemaType.BOOL,
}


def point_id(chunk_id: str) -> str:
    """`TR-114#4.2` -> устойчивый UUID.

    Qdrant принимает только целые и UUID, а адрес чанка — строка. UUID5
    детерминирован: один и тот же адрес всегда даёт один и тот же id.
    """
    return str(uuid.uuid5(NS, chunk_id))


def client(prefer_grpc: bool = True, timeout: int = 60) -> QdrantClient:
    """Клиент из переменных окружения. Переезд к партнёру — правка .env."""
    url = os.environ["QDRANT_URL"]
    return QdrantClient(
        url=url,
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        prefer_grpc=prefer_grpc,
        # gRPC живёт на своём порту, клиент сам его не выведет из URL
        grpc_port=int(os.environ.get("QDRANT_GRPC_PORT", 6334)),
        timeout=timeout,
    )


def create_collection(
    cl: QdrantClient,
    name: str,
    dim: int,
    recreate: bool = False,
    with_indexes: bool = True,
) -> None:
    """Коллекция под загрузку: индексация выключена, payload-индексы созданы.

    `indexing_threshold=0` ставится сразу при создании, а не отдельным
    вызовом перед загрузкой: так меньше шансов начать заливать в коллекцию,
    которая строит граф на каждый батч.
    """
    if recreate and cl.collection_exists(name):
        cl.delete_collection(name)
    if cl.collection_exists(name):
        return

    cl.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
            on_disk=True,
        ),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )
    if with_indexes:
        for field, schema in PAYLOAD_INDEXES.items():
            cl.create_payload_index(name, field_name=field, field_schema=schema)


def to_points(chunks, vectors) -> list[models.PointStruct]:
    """Чанки + векторы -> пойнты. Порядок обязан совпадать."""
    if len(chunks) != len(vectors):
        raise ValueError(f"чанков {len(chunks)}, векторов {len(vectors)}")
    return [
        models.PointStruct(id=point_id(ch.id), vector=v, payload={"chunk_id": ch.id, **ch.payload})
        for ch, v in zip(chunks, vectors)
    ]


def upload(
    cl: QdrantClient,
    name: str,
    points,
    batch_size: int = 128,
    parallel: int = 1,
    wait: bool = False,
) -> None:
    """Пакетная загрузка. Клиент сам режет на батчи и раскладывает по работникам.

    `wait=False` намеренно: при `wait=True` заливка сериализуется — отправили
    батч, стоим, пока он применится, только потом шлём следующий. Обратная
    сторона — сразу после возврата данных в поиске может ещё не быть,
    поэтому дальше обязателен `finish_indexing`.

    ВНИМАНИЕ, `parallel > 1`. Справочник (RAG/04-qdrant.md) советует 2–4
    «потока», но qdrant-client разводит работу ПРОЦЕССАМИ, а не потоками.
    На macOS процессы порождаются через spawn, то есть дочерний заново
    импортирует главный модуль. Скрипт без `if __name__ == "__main__"`
    при этом выполняется целиком в каждом работнике — на нашем стенде это
    дало три процесса, заново векторизующих корпус, вместо загрузки.

    Поэтому по умолчанию 1: значение, которое не может испортить чужой
    скрипт. Больше единицы ставить осознанно и только под защитой `__main__`.
    """
    cl.upload_points(
        collection_name=name,
        points=points,
        batch_size=batch_size,
        parallel=parallel,
        wait=wait,
    )


def upload_naive(cl: QdrantClient, name: str, points) -> None:
    """Как делать НЕ надо: по одному пойнту, с ожиданием каждого.

    Существует ради замера — на этой функции меряется цена приёмов
    из `upload`. В пайплайне не используется.
    """
    for p in points:
        cl.upsert(collection_name=name, points=[p], wait=True)


def finish_indexing(
    cl: QdrantClient,
    name: str,
    dim: int,
    threshold: int = INDEXING_THRESHOLD,
    timeout: float = 300,
    grace: float = 5.0,
) -> dict:
    """Вернуть порог индексации и дождаться, пока индекс действительно построится.

    Возвращает состояние коллекции и явный вердикт — построен HNSW или нет.

    ЗАЧЕМ НЕ ПРОСТО «ЖДАТЬ GREEN». Замер 09.08.2026 на 10 000 пойнтов:
    после возврата порога в 20000 КБ коллекция была зелёной **сразу**
    и оставалась зелёной 24 секунды с `indexed_vectors_count = 0`.
    Индекс не строился вовсе.

    Причина: `indexing_threshold` применяется К СЕГМЕНТУ, а не к коллекции.
    Qdrant сам разложил 10 000 пойнтов на 5 сегментов, в каждом вышло
    около 8 МБ векторов при пороге 20 МБ — ни один сегмент до порога
    не дотянул. Проверено обеими сторонами: с порогом 1000 КБ индекс
    построился за 2 с, и с одним сегментом вместо пяти — за 3 с.

    Отсюда: **зелёный статус означает «оптимизатору нечего делать»,
    а не «индекс построен»**. Ждать его как признак готовности нельзя —
    он выставлен с самого начала. Поэтому ждём здесь роста
    `indexed_vectors_count`, а если индексация так и не началась за `grace`
    секунд — говорим об этом прямо, а не выдаём молчание за успех.

    Непостроенный индекс на малой коллекции — не ошибка: полный перебор
    на тысячах векторов быстрее графа. Ошибка — не знать, что происходит,
    и потом объяснять медленный поиск чем-нибудь другим.
    """
    cl.update_collection(
        collection_name=name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=threshold),
    )
    started = time.perf_counter()
    while True:
        info = cl.get_collection(name)
        waited = time.perf_counter() - started
        done = info.indexed_vectors_count and info.indexed_vectors_count >= (info.points_count or 0)
        if done:
            break
        # индексация не началась и не начнётся: статус зелёный, значит
        # оптимизатор ничего не запланировал
        if waited > grace and info.status == models.CollectionStatus.GREEN:
            break
        if waited > timeout:
            raise TimeoutError(f"коллекция {name} не доиндексировалась за {timeout} с")
        time.sleep(0.5)

    segments = info.segments_count or 1
    per_segment_kb = (info.points_count or 0) * dim * 4 / segments / 1024
    indexed = info.indexed_vectors_count or 0
    if indexed >= (info.points_count or 0) and indexed:
        verdict = "HNSW построен"
    else:
        verdict = (
            f"HNSW НЕ строился: ~{per_segment_kb:.0f} КБ векторов на сегмент "
            f"при пороге {threshold} КБ. Поиск идёт полным перебором — "
            f"на этом объёме так и быстрее"
        )
    return {
        "статус": str(info.status),
        "пойнтов": info.points_count,
        "сегментов": segments,
        "проиндексировано векторов": indexed,
        "ждали, с": round(waited, 2),
        "вердикт": verdict,
    }
