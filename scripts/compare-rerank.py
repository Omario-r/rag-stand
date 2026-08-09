"""Что даёт реранкер: прирост, задержка и цена глубины каскада. Этап 6.

Отвечает на четыре вопроса, и третий важнее первых двух:

1. Насколько лучше метрики с реранкером?
2. Сколько он добавляет к задержке?
3. СКОЛЬКО ВОПРОСОВ ВООБЩЕ ИЗМЕНИЛИ ОТВЕТ? Без этого счётчика разница
   в средних на 32 вопросах неотличима от шума
   ([[lab-style#Правило 10. Санитарная проверка не заменяет метрику]]).
4. Сколько кандидатов ему давать: 10, 20 или 50?

ВАЖНО ПРО СЧЁТЧИК ИЗ ПУНКТА 3. Он зависит от метрики, по которой считается,
и два счётчика по разным метрикам сравнивать между собой нельзя. recall@5
реагирует только на смену СОСТАВА пятёрки, nDCG@5 — на любую перестановку
ВНУТРИ неё. Реранкер переставляет, поэтому по nDCG@5 у него счётчик всегда
будет больше, и сравнивать его с числом из compare-embedders.py (там
считался recall@5) — значит сравнивать разноимённое. Поэтому печатаются
обе колонки, и сопоставима с эмбеддерами только вторая.

Запуск:  uv run python scripts/compare-rerank.py
"""

import sys
import time
from pathlib import Path

STAND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAND / "src"))

from rag_lab import search as S  # noqa: E402
from rag_lab import store  # noqa: E402
from rag_lab.corpus import build_chunks, chunk_by_clause, golden_set  # noqa: E402
from rag_lab.embed import Embedder  # noqa: E402
from rag_lab.metrics import evaluate_one, summarize  # noqa: E402
from rag_lab.rerank import Reranker  # noqa: E402

KS = (1, 5, 50)
DEPTHS = (10, 20, 50)


def head_of(pl: dict) -> str:
    head = pl.get("doc_type", "")
    if pl.get("section_title"):
        head += f", раздел «{pl['section_title']}»"
    return f"{head}\n{pl['text']}" if head else pl["text"]


def main() -> None:
    chunks = build_chunks()
    by_clause = chunk_by_clause(chunks)
    questions = golden_set()
    emb, cl, rr = Embedder(), store.client(), Reranker()

    if not rr.healthy():
        sys.exit("реранкер отвечает, но скоры битые — проверьте конвертацию модели")
    print("санитарная проверка реранкера пройдена: модель не битая")
    print("(она НЕ отвечает на вопрос, полезен ли реранкер — на него отвечают цифры ниже)\n")

    expected = [
        {by_clause[r] for r in q["relevant"]} for q in questions
    ]
    base = S.search_batch(cl, emb, [q["q"] for q in questions], limit=max(DEPTHS))

    variants = {}

    # --- 1. только вектор
    variants["только вектор"] = [[cid for cid, _, _ in res] for res in base]

    # --- 2. реранкер на разной глубине каскада, голый текст пункта
    lat = {}
    for depth in DEPTHS:
        t0 = time.perf_counter()
        out = []
        for q, res in zip(questions, base):
            # Реранкуется голова, хвост остаётся в порядке векторного поиска.
            # Иначе «реранкер, top-10» терял бы 40 уже найденных кандидатов
            # и проигрывал по recall@50 просто потому, что список короче.
            head, tail = res[:depth], res[depth:]
            order = rr.rank(q["q"], [pl["text"] for _, _, pl in head])
            out.append([head[i][0] for i, _ in order] + [cid for cid, _, _ in tail])
        lat[depth] = (time.perf_counter() - t0) / len(questions) * 1000
        variants[f"реранкер, top-{depth}"] = out

    # --- 3. реранкер с шапкой чанка, полная глубина
    out = []
    t0 = time.perf_counter()
    for q, res in zip(questions, base):
        order = rr.rank(q["q"], [head_of(pl) for _, _, pl in res])
        out.append([res[i][0] for i, _ in order])
    lat["с шапкой"] = (time.perf_counter() - t0) / len(questions) * 1000
    variants["реранкер, top-50, с шапкой"] = out

    # --- метрики
    rows = {}
    print(f"{'вариант':30s} {'r@1':>6} {'r@5':>6} {'hit@1':>7} {'MRR':>7} {'nDCG@5':>7} {'мс/запрос':>10}")
    print("-" * 82)
    for label, found in variants.items():
        rr_rows = [evaluate_one(f, rel, ks=KS) for f, rel in zip(found, expected)]
        rows[label] = rr_rows
        s = summarize(rr_rows, ks=KS)
        if "шапк" in label:
            ms = f"{lat['с шапкой']:.0f}"
        elif label.startswith("реранкер, top-"):
            ms = f"{lat[int(label.rsplit('-', 1)[1])]:.0f}"
        else:
            ms = "—"
        print(f"{label:30s} {s['recall@1']:6.3f} {s['recall@5']:6.3f} {s['hit@1']:7.3f} "
              f"{s['MRR']:7.3f} {s['nDCG@5']:7.3f} {ms:>10}")

    # --- hit@1 в штуках: доля скрывает, что за ней стоит всего несколько вопросов
    BASE = "только вектор"
    n_q = len(questions)
    print(f"\nВопросов с верным ответом на ПЕРВОМ месте (hit@1 в штуках, не в долях):")
    for label in variants:
        hits = sum(1 for r in rows[label] if r["hit@1"] == 1)
        print(f"  {label:30s} {hits:2d} из {n_q}")

    # --- правило 10: сколько вопросов вообще изменили ответ
    print(f"\nПопарно с «{BASE}». Счётчик зависит от метрики — колонки НЕ взаимозаменяемы.")
    print("С числом из compare-embedders.py сопоставима только колонка recall@5.")
    print(f"{'вариант':30s} {'Δ nDCG@5':>10} {'Δ MRR':>8} {'изм. nDCG@5':>13} {'изм. recall@5':>15}")
    print("-" * 82)
    for label in variants:
        if label == BASE:
            continue
        d_nd = [a["nDCG@5"] - b["nDCG@5"] for a, b in zip(rows[label], rows[BASE])]
        d_mrr = [a["MRR"] - b["MRR"] for a, b in zip(rows[label], rows[BASE])]
        d_r5 = [a["recall@5"] - b["recall@5"] for a, b in zip(rows[label], rows[BASE])]
        ch_nd = sum(1 for x in d_nd if abs(x) > 1e-9)
        ch_r5 = sum(1 for x in d_r5 if abs(x) > 1e-9)
        print(f"{label:30s} {sum(d_nd)/len(d_nd):+10.3f} {sum(d_mrr)/len(d_mrr):+8.3f}"
              f" {f'{ch_nd} из {len(d_nd)}':>13} {f'{ch_r5} из {len(d_r5)}':>15}")

    # --- где реранкер испортил
    best = "реранкер, top-50"
    worse = [
        (q["q"], b["позиция первого верного"], a["позиция первого верного"])
        for q, a, b in zip(questions, rows[best], rows[BASE])
        if a["MRR"] < b["MRR"]
    ]
    print(f"\nвопросов, где реранкер УХУДШИЛ позицию: {len(worse)}")
    for text, was, now in worse:
        print(f"  было {was} -> стало {now}: {text[:58]}")


if __name__ == "__main__":
    main()
