"""Метрики качества поиска. Этап 5.

Разбор — RAG/research/reference/retrieval-rerank.md, лаба — RAG-labs/lab-05-search.md.

Все функции принимают ранжированный список найденных идентификаторов
и множество верных. Ничего не знают ни про Qdrant, ни про эмбеддер:
метрику должно быть можно посчитать на бумажке.

ДВЕ РАЗНЫЕ ВЕЩИ НАЗЫВАЮТСЯ СЛОВОМ RECALL, и путать их дорого:

  * `recall_at_k` — какая ДОЛЯ верных пунктов найдена. Вопрос с тремя
    верными ответами, из которых нашёлся один, даёт 0,33;
  * `hit_at_k`    — нашёлся ли ХОТЯ БЫ ОДИН верный. Тот же вопрос даёт 1.

Справочник определяет recall@50 через вторую («доля вопросов, где нужный
чанк попал в кандидаты»), а замер 08.08.2026 считал первую. Поэтому здесь
есть обе и печатаются обе — иначе цифры разных замеров нельзя сравнивать
между собой, а это ровно то, ради чего журнал экспериментов и заводится.
"""

from __future__ import annotations

import math


def recall_at_k(found: list[str], relevant: set[str], k: int) -> float:
    """Доля верных пунктов, попавших в top-k."""
    if not relevant:
        return 0.0
    return len(relevant & set(found[:k])) / len(relevant)


def hit_at_k(found: list[str], relevant: set[str], k: int) -> float:
    """1, если в top-k есть хоть один верный."""
    return 1.0 if relevant & set(found[:k]) else 0.0


def reciprocal_rank(found: list[str], relevant: set[str]) -> float:
    """1/позиция первого верного. Ноль, если верных не нашлось вовсе."""
    for pos, item in enumerate(found, 1):
        if item in relevant:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(found: list[str], relevant: set[str], k: int) -> float:
    """nDCG@k с бинарной полезностью.

    Учитывает и порядок, и число верных ответов: вопрос с тремя верными,
    у которого все три стоят первыми, даёт 1,0; тот же вопрос с одним
    верным на пятом месте — заметно меньше.

    Идеальный порядок считается по min(число верных, k) — иначе вопрос
    с четырьмя верными ответами не смог бы получить единицу при k = 3,
    и метрика наказывала бы за разметку, а не за поиск.
    """
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(found[:k], 1) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def summarize(rows: list[dict], ks=(1, 5, 50)) -> dict:
    """Средние по вопросам. `rows` — то, что вернул прогон по каждому вопросу."""
    n = len(rows)
    if not n:
        return {}
    out = {}
    for k in ks:
        out[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / n
        out[f"hit@{k}"] = sum(r[f"hit@{k}"] for r in rows) / n
    out["MRR"] = sum(r["MRR"] for r in rows) / n
    out["nDCG@5"] = sum(r["nDCG@5"] for r in rows) / n
    return out


def evaluate_one(found: list[str], relevant: set[str], ks=(1, 5, 50)) -> dict:
    """Все метрики по одному вопросу."""
    row = {"найдено верных": len(relevant & set(found)), "верных всего": len(relevant)}
    for k in ks:
        row[f"recall@{k}"] = recall_at_k(found, relevant, k)
        row[f"hit@{k}"] = hit_at_k(found, relevant, k)
    row["MRR"] = reciprocal_rank(found, relevant)
    row["nDCG@5"] = ndcg_at_k(found, relevant, 5)
    row["позиция первого верного"] = next(
        (i for i, x in enumerate(found, 1) if x in relevant), None
    )
    return row
