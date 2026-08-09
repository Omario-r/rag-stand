"""Базовая линия качества поиска на золотом наборе. Этап 5.

Считает recall@1/5/50, hit@k, MRR и nDCG@5 по 32 вопросам и печатает
разбор по каждому вопросу. Рядом — та же метрика для случайной выдачи:
без неё непонятно, много ли значит recall@50 на корпусе из 144 чанков.

ОГОВОРКА, БЕЗ КОТОРОЙ ЦИФРЫ ВВОДЯТ В ЗАБЛУЖДЕНИЕ. Это НЕ базовая линия
проекта. Корпус первой партии — 144 чанка, top-50 покрывает больше трети;
на таком объёме recall@50 упирается в потолок у любого метода, а разницу
между методами он не показывает вовсе (RAG/08-local-stand.md, «Корпус
первой партии не различает конфигурации»). Базовая линия ставится
на партии 2.

Запуск:
    uv run python scripts/evaluate.py            # базовый прогон
    uv run python scripts/evaluate.py --json     # строка для журнала
"""

import json
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

LIMIT = 50
KS = (1, 5, 50)


def expected(question: dict, by_clause: dict[str, str]) -> set[str]:
    """Ссылки золотого набора -> идентификаторы чанков, в которые они попали.

    Ссылка указывает на ПУНКТ, а поиск возвращает ЧАНК, и это не всегда
    одно и то же: подпункт 3.4.3.1 физически лежит внутри чанка AR-203#3.4.
    Считать напрямую по ссылкам — значит занизить recall из-за склейки,
    а не из-за качества поиска.

    Поле `superseded` в верные не идёт намеренно: это пункт, который
    допсоглашение отменило. На вопрос «какая СЕЙЧАС предельная сумма»
    он и должен считаться промахом.
    """
    out = set()
    for ref in question.get("relevant", []):
        cid = by_clause.get(ref)
        if cid is None:
            raise KeyError(f"ссылка {ref} не покрыта ни одним чанком")
        out.add(cid)
    return out


def main() -> None:
    chunks = build_chunks()
    by_clause = chunk_by_clause(chunks)
    all_ids = [ch.id for ch in chunks]
    questions = golden_set()

    emb = Embedder()
    cl = store.client()

    n_points = cl.count(store.COLLECTION, exact=True).count
    print(f"чанков в коллекции: {n_points}   вопросов: {len(questions)}   top-{LIMIT}")
    print(f"top-{LIMIT} покрывает {LIMIT / n_points:.0%} корпуса\n")

    t0 = time.perf_counter()
    results = S.search_batch(cl, emb, [q["q"] for q in questions], limit=LIMIT)
    elapsed = time.perf_counter() - t0

    rows, rnd_rows = [], []
    print(f"{'вопрос':52s} {'верных':>7} {'поз.':>5} {'r@5':>6} {'nDCG@5':>7}")
    print("-" * 82)
    for i, (q, res) in enumerate(zip(questions, results)):
        rel = expected(q, by_clause)
        found = [cid for cid, _, _ in res]
        row = evaluate_one(found, rel, ks=KS)
        rows.append(row)
        rnd_rows.append(
            evaluate_one(S.random_order(all_ids, LIMIT, seed=i), rel, ks=KS)
        )
        pos = row["позиция первого верного"]
        print(
            f"{q['q'][:52]:52s} {len(rel):>7} {str(pos or '—'):>5} "
            f"{row['recall@5']:6.2f} {row['nDCG@5']:7.3f}"
        )

    got = summarize(rows, ks=KS)
    rnd = summarize(rnd_rows, ks=KS)

    print("\n" + "=" * 82)
    print(f"{'метрика':12s} {'поиск':>9} {'случайно':>10}   что означает")
    print("-" * 82)
    meaning = {
        "recall@1": "доля верных пунктов на первом месте",
        "recall@5": "доля верных пунктов в пятёрке",
        "recall@50": "доля верных пунктов в кандидатах",
        "hit@1": "вопросов, где первый ответ верен",
        "hit@5": "вопросов, где верное есть в пятёрке",
        "hit@50": "вопросов, где верное есть в кандидатах",
        "MRR": "обратная позиция первого верного",
        "nDCG@5": "порядок и число верных вместе",
    }
    for k in ("recall@1", "recall@5", "recall@50", "hit@1", "hit@5", "hit@50", "MRR", "nDCG@5"):
        print(f"{k:12s} {got[k]:9.3f} {rnd[k]:10.3f}   {meaning[k]}")

    print(f"\nвремя на {len(questions)} запросов: {elapsed:.2f} с "
          f"({elapsed / len(questions) * 1000:.0f} мс на запрос, кэш вопросов тёплый)")

    # --- вопросы, на которых поиск промахнулся мимо пятёрки
    bad = [(q, r) for q, r in zip(questions, rows) if r["hit@5"] == 0]
    print(f"\nвопросов без верного ответа в пятёрке: {len(bad)}")
    for q, r in bad:
        print(f"  поз. {r['позиция первого верного'] or '—'}: {q['q']}")

    if "--json" in sys.argv:
        line = {
            "дата": time.strftime("%Y-%m-%d"),
            "коллекция": store.COLLECTION,
            "чанков": n_points,
            "вопросов": len(questions),
            **{k: round(v, 3) for k, v in got.items()},
        }
        print("\n" + json.dumps(line, ensure_ascii=False))
        log = STAND / "experiments.jsonl"
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"дописано в {log.name}")


if __name__ == "__main__":
    main()
