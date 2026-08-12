#!/usr/bin/env python
"""Корпус -> векторы -> Qdrant -> поиск. Шаг 17 из RAG-labs/lab-02-embeddings.md.

Сквозной прогон, соединяющий лаб 1 и лаб 2. Это ещё не пайплайн этапа 4:
разбор пунктов черновой, батчи наивные, замеров загрузки нет.

Игрушечная коллекция toy из лаб 1 не трогается.
"""

import glob
import json
import os
import re
import sys
import urllib.request
import uuid

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from rag_lab.embed import Embedder  # noqa: E402

COLLECTION = "corpus_v1"
CLAUSE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.{40,})$")
# Пространство имён для UUID5. Фиксированное: id пойнта должен зависеть
# только от doc_id и номера пункта, иначе повторная загрузка наплодит дублей.
NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def qd(method, path, body=None):
    req = urllib.request.Request(
        os.environ["QDRANT_URL"] + path,
        json.dumps(body).encode() if body is not None else None,
        {"api-key": os.environ["QDRANT_API_KEY"], "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if method == "DELETE":
            return None
        raise RuntimeError(f"{e.code}: {e.read().decode()[:300]}") from None


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    items = []
    for path in sorted(glob.glob(os.path.join(root, "corpus", "*.md"))):
        base = os.path.basename(path)[:-3]
        if base == "README":
            continue
        doc = base.replace("-2026", "")
        for line in open(path, encoding="utf-8"):
            m = CLAUSE.match(line.strip())
            if m:
                items.append((doc, m.group(1), m.group(2).strip()))

    e = Embedder()
    qd("DELETE", f"/collections/{COLLECTION}")
    qd("PUT", f"/collections/{COLLECTION}", {"vectors": {"size": e.dim, "distance": "Cosine"}})
    print(f"коллекция {COLLECTION} создана, размерность {e.dim} — прочитана у сервера")

    vecs = e.embed([t for _, _, t in items], kind="document")
    points = [
        {
            "id": str(uuid.uuid5(NS, f"{doc}#{num}")),
            "vector": v,
            "payload": {"doc_id": doc, "clause": num, "text": text},
        }
        for (doc, num, text), v in zip(items, vecs)
    ]
    qd("PUT", f"/collections/{COLLECTION}/points?wait=true", {"points": points})
    count = qd("POST", f"/collections/{COLLECTION}/points/count", {"exact": True})["result"]["count"]
    print(f"залито пойнтов: {count}")

    gs = yaml.safe_load(open(os.path.join(root, "corpus", "golden-set.yaml"), encoding="utf-8"))
    for q in gs["questions"][:3]:
        qv = e.embed_one(q["q"], kind="query")
        found = qd(
            "POST",
            f"/collections/{COLLECTION}/points/query",
            {"query": qv, "limit": 3, "with_payload": True},
        )["result"]["points"]
        print()
        print("вопрос:", q["q"])
        print("ждём  :", ", ".join(q["relevant"]))
        for p in found:
            pl = p["payload"]
            ref = f"{pl['doc_id']}#{pl['clause']}"
            mark = "ВЕРНО" if ref in q["relevant"] else "     "
            print(f"  {mark} {p['score']:.4f}  {ref:16} {pl['text'][:46]}")


if __name__ == "__main__":
    main()
