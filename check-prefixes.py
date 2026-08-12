#!/usr/bin/env python
"""Нужны ли префиксы-инструкции для запросов. Шаг 14 из RAG-labs/lab-02-embeddings.md.

A/B на золотом наборе: одни и те же вопросы и пункты, отличается только
подготовка текста запроса.

ВАЖНО: цифры здесь — НЕ базовая линия проекта. Разбор пунктов черновой
(регулярка в одну строку), SH-45 не покрыт вовсе. Годятся только для
сравнения вариантов между собой в одинаковых условиях.
"""

import glob
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from rag_lab.embed import Embedder  # noqa: E402

CLAUSE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.{40,})$")

ARMS = {
    "без префикса": lambda q: q,
    "Instruct общий": lambda q: (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery: " + q
    ),
    "Instruct предметный": lambda q: (
        "Instruct: Найди пункт договора, отвечающий на вопрос\nQuery: " + q
    ),
}


def load_chunks(root):
    chunks = {}
    for path in sorted(glob.glob(os.path.join(root, "corpus", "*.md"))):
        base = os.path.basename(path)[:-3]
        if base == "README":
            continue
        doc = base.replace("-2026", "")
        for line in open(path, encoding="utf-8"):
            m = CLAUSE.match(line.strip())
            if m:
                chunks[f"{doc}#{m.group(1)}"] = m.group(2).strip()
    return chunks


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    chunks = load_chunks(root)
    keys = list(chunks)
    print(f"пунктов разобрано: {len(keys)}")

    gs = yaml.safe_load(open(os.path.join(root, "corpus", "golden-set.yaml"), encoding="utf-8"))
    all_q = gs["questions"]
    qs = [q for q in all_q if any(r in chunks for r in q["relevant"])]
    covered = sum(1 for q in all_q for r in q["relevant"] if r in chunks)
    total = sum(len(q["relevant"]) for q in all_q)
    print(f"вопросов пригодных: {len(qs)}/{len(all_q)}, ссылок покрыто: {covered}/{total}")
    print("(SH-45 не покрыт — нумерация Word, в тексте цифр нет)")

    e = Embedder()
    dv = e.embed([chunks[k] for k in keys], kind="document")

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    print()
    print(f"{'вариант запроса':22} {'recall@5':>9} {'recall@10':>10} {'MRR':>7}")
    for name, prep in ARMS.items():
        qv = e.embed([prep(q["q"]) for q in qs], kind="query")
        r5 = r10 = mrr = 0.0
        for q, v in zip(qs, qv):
            rel = {r for r in q["relevant"] if r in chunks}
            order = sorted(range(len(keys)), key=lambda i: -cos(v, dv[i]))
            top = [keys[i] for i in order]
            r5 += len(rel & set(top[:5])) / len(rel)
            r10 += len(rel & set(top[:10])) / len(rel)
            for pos, k in enumerate(top, 1):
                if k in rel:
                    mrr += 1 / pos
                    break
        n = len(qs)
        print(f"{name:22} {r5/n:9.3f} {r10/n:10.3f} {mrr/n:7.3f}")


if __name__ == "__main__":
    main()
