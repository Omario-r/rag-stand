#!/usr/bin/env python
"""Сравнение конфигураций эмбеддера на золотом наборе.

Замер 08.08.2026, разбор — RAG/research/reference/local-stand.md, раздел «Корпус первой
партии не различает конфигурации».

Повод: партнёр использует multilingual-e5-large, мы — Qwen3-Embedding-0.6B.
Надо было понять, есть ли основания требовать смены модели (не оказалось),
и заодно проверить два тихих параметра: префиксы и пулинг.

ВАЖНО, две оговорки:

1. Цифры здесь — НЕ базовая линия проекта. Разбор пунктов черновой
   (регулярка в одну строку), SH-45 не покрыт вовсе. Годятся только для
   сравнения вариантов между собой в одинаковых условиях.

2. Средние по 27 вопросам почти ничего не значат — корпус слишком мал.
   Поэтому вторая таблица важнее первой: она показывает, сколько вопросов
   ВООБЩЕ изменили ответ. Обычно это 1-3 из 27, то есть разницы нет.
   Не заменять этот счётчик доверительным интервалом среднего: при таком
   числе ничьих интервал упирается в ноль и выглядит значимым, не будучи им.

Требует поднятых серверов:
    ./run-embedder.sh        Qwen3, порт 8080  (штатный)
    ./run-embedder-e5.sh mean 8082
    ./run-embedder-e5.sh cls  8083

Запуск:
    set -a; source .env; set +a
    ./.venv/bin/python compare-embedders.py
"""

import glob
import json
import os
import re
import sys
import urllib.request

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from rag_lab.embed import Embedder  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get("LLAMA_API_KEY", "")

# Черновая регулярка разбора пунктов — та же, что в check-prefixes.py.
# Настоящий парсер — этап 3.
CLAUSE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.{40,})$")

# (метка, url, пулинг, префикс запроса, префикс документа)
ARMS = [
    ("Qwen3 без префикса (наша база)", "http://127.0.0.1:8080", "—", "", ""),
    ("e5 mean без префиксов (как у партнёра)", "http://127.0.0.1:8082", "mean", "", ""),
    ("e5 mean query:/passage:", "http://127.0.0.1:8082", "mean", "query: ", "passage: "),
    ("e5 cls без префиксов", "http://127.0.0.1:8083", "cls", "", ""),
    ("e5 cls query:/passage:", "http://127.0.0.1:8083", "cls", "query: ", "passage: "),
]

BASE = "e5 mean без префиксов (как у партнёра)"


def load_chunks():
    chunks = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "corpus", "*.md"))):
        base = os.path.basename(path)[:-3]
        if base == "README":
            continue
        doc = base.replace("-2026", "")
        for line in open(path, encoding="utf-8"):
            m = CLAUSE.match(line.strip())
            if m:
                chunks[f"{doc}#{m.group(1)}"] = m.group(2).strip()
    return chunks


def ntok(url, text):
    """Длина текста в токенах ЭТОЙ модели. Читаем у сервера, не считаем сами."""
    req = urllib.request.Request(
        url + "/tokenize",
        json.dumps({"content": text}).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
    )
    return len(json.load(urllib.request.urlopen(req, timeout=60))["tokens"])


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def recall_at_5_per_question(url, pool, qpref, dpref, chunks, keys, qs):
    """Список recall@5 по каждому вопросу — не среднее, оно считается снаружи."""
    e = Embedder(
        url=url,
        api_key=KEY,
        cache_path=os.path.join(ROOT, "cache", "compare-embedders.sqlite"),
        # пулинг и префиксы меняют вектор при том же тексте, поэтому входят в ключ кэша
        model=f"{url[-4:]}|{pool}|{'pref' if qpref else 'raw'}",
    )
    dv = e.embed([dpref + chunks[k] for k in keys], kind="document")
    qv = e.embed([qpref + q["q"] for q in qs], kind="query")
    out = []
    for q, v in zip(qs, qv):
        rel = {r for r in q["relevant"] if r in chunks}
        order = sorted(range(len(keys)), key=lambda i: -cos(v, dv[i]))
        top = [keys[i] for i in order]
        out.append(
            (
                len(rel & set(top[:5])) / len(rel),
                len(rel & set(top[:10])) / len(rel),
                next((1 / p for p, k in enumerate(top, 1) if k in rel), 0.0),
            )
        )
    return out


def main():
    chunks = load_chunks()
    keys = list(chunks)
    gs = yaml.safe_load(open(os.path.join(ROOT, "corpus", "golden-set.yaml"), encoding="utf-8"))
    all_q = gs["questions"]
    qs = [q for q in all_q if any(r in chunks for r in q["relevant"])]
    print(f"пунктов разобрано: {len(keys)}, вопросов пригодных: {len(qs)}/{len(all_q)}")
    print("(SH-45 не покрыт — нумерация Word, в тексте цифр нет)")

    # Упираются ли наши пункты в предел e5 в 512 токенов
    try:
        lens = [ntok("http://127.0.0.1:8082", t) for t in chunks.values()]
        srt = sorted(lens)
        print(
            f"длина пунктов в токенах e5: {srt[0]}–{srt[-1]}, медиана {srt[len(srt)//2]}, "
            f"за пределом 512: {sum(1 for n in lens if n > 512)}"
        )
    except OSError:
        print("сервер e5 на 8082 не отвечает — пропускаю подсчёт токенов")

    per = {}
    print()
    print(f"{'конфигурация':40} {'recall@5':>9} {'recall@10':>10} {'MRR':>7}")
    print("-" * 70)
    for label, url, pool, qp, dp in ARMS:
        rows = recall_at_5_per_question(url, pool, qp, dp, chunks, keys, qs)
        per[label] = [r[0] for r in rows]
        n = len(rows)
        print(
            f"{label:40} {sum(r[0] for r in rows)/n:9.3f} "
            f"{sum(r[1] for r in rows)/n:10.3f} {sum(r[2] for r in rows)/n:7.3f}"
        )

    print()
    print(f"Попарно с «{BASE}», recall@5.")
    print("Главная колонка — последняя: сколько вопросов вообще изменили ответ.")
    print()
    print(f"{'конфигурация':40} {'Δ среднего':>11} {'различается':>12}")
    print("-" * 66)
    for label in per:
        if label == BASE:
            continue
        d = [a - b for a, b in zip(per[label], per[BASE])]
        diff = sum(1 for x in d if x != 0)
        print(f"{label:40} {sum(d)/len(d):+11.3f} {f'{diff} из {len(d)}':>12}")

    print()
    print("Если в последней колонке единицы — конфигурации неразличимы,")
    print("и разница в средних является шумом, а не результатом.")


if __name__ == "__main__":
    main()
