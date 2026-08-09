"""Поиск: вопрос -> вектор -> кандидаты из Qdrant. Этап 5.

Тонкий слой над Qdrant. Существует ради трёх вещей, каждая из которых
уже успела где-нибудь стоить времени:

1. Запрос векторизуется ТОЙ ЖЕ функцией, что и документы (`text_for`
   из embed.py). Забытый префикс не даёт ошибки — он даёт тихо
   деградировавший поиск.

2. Возвращается `chunk_id`, а не идентификатор пойнта. UUID5 нужен
   Qdrant, а метрике нужен адрес `TR-114#4.2`.

3. Случайный поиск как точка отсчёта. Без него непонятно, много ли
   значит recall@50 = 1,0 на корпусе из 144 чанков.
"""

from __future__ import annotations

import random

from qdrant_client import models


def search(cl, emb, question: str, limit: int = 50, collection: str = None, flt=None):
    """Вопрос -> список (chunk_id, score, payload) по убыванию близости."""
    from . import store

    qv = emb.embed_one(question, kind="query")
    res = cl.query_points(
        collection_name=collection or store.COLLECTION,
        query=qv,
        limit=limit,
        with_payload=True,
        query_filter=flt,
    )
    return [(p.payload["chunk_id"], p.score, p.payload) for p in res.points]


def search_batch(cl, emb, questions: list[str], limit: int = 50, collection: str = None):
    """То же для списка вопросов. Векторы считаются одной пачкой.

    Не ради скорости — замер 07.08.2026 показал, что батчинг ничего
    не ускоряет, — а ради одного обращения к кэшу и меньшего числа
    точек отказа.
    """
    from . import store

    qvs = emb.embed(questions, kind="query")
    out = []
    for qv in qvs:
        res = cl.query_points(
            collection_name=collection or store.COLLECTION,
            query=qv,
            limit=limit,
            with_payload=True,
        )
        out.append([(p.payload["chunk_id"], p.score, p.payload) for p in res.points])
    return out


def random_order(chunk_ids: list[str], limit: int, seed: int) -> list[str]:
    """Случайная выдача — точка отсчёта, а не поиск.

    Смысл: на маленьком корпусе top-50 покрывает треть всех чанков,
    и высокий recall@50 может получиться сам собой. Сравнение с этой
    функцией показывает, сколько из результата дал поиск, а сколько —
    размер корпуса.
    """
    rnd = random.Random(seed)
    pool = list(chunk_ids)
    rnd.shuffle(pool)
    return pool[:limit]


def filter_by(**kwargs) -> models.Filter:
    """Фильтр по точным значениям payload: filter_by(doc_type='аренда_ТС')."""
    return models.Filter(
        must=[
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in kwargs.items()
        ]
    )
