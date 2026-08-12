"""Сколько стоят приёмы быстрой загрузки. Пункт 4 этапа 4.

Справочник RAG/research/reference/qdrant.md перечисляет пять приёмов и цифры из чужих
статей. Здесь они проверяются на нашем железе: своя цифра вместо статейной.

Меряется скорость ЗАГРУЗКИ, не качество. Векторы синтетические (случайные
единичной длины) — для замера пропускной способности содержимое вектора
безразлично, а корпуса в 144 пойнта на такой замер не хватает.

Рабочая коллекция не трогается: всё идёт в отдельную, которая удаляется
в конце.

Запуск:  uv run python scripts/measure-upload.py [сколько пойнтов]
"""

import sys
import time
from pathlib import Path

import numpy as np
from qdrant_client import models

STAND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAND / "src"))

from rag_lab import store  # noqa: E402
from rag_lab.embed import Embedder  # noqa: E402

BENCH = "bench_tmp"
NAIVE_N = 500  # наивная загрузка идёт по одному, на 10 000 её никто не дождётся


def synth(n: int, dim: int, seed: int = 17) -> list[models.PointStruct]:
    """n случайных пойнтов. Векторы нормированы — как настоящие."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return [
        models.PointStruct(
            id=i,
            vector=v[i].tolist(),
            payload={"doc_id": f"DOC-{i % 30}", "clause_no": f"{i}", "has_table": False},
        )
        for i in range(n)
    ]


def fresh(cl, dim: int, threshold: int, with_indexes: bool = True) -> None:
    """Пустая коллекция под очередной замер — чтобы предыдущий не влиял."""
    if cl.collection_exists(BENCH):
        cl.delete_collection(BENCH)
    cl.create_collection(
        collection_name=BENCH,
        vectors_config=models.VectorParams(
            size=dim, distance=models.Distance.COSINE, on_disk=True
        ),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=threshold),
    )
    if with_indexes:
        for field, schema in store.PAYLOAD_INDEXES.items():
            if field in ("doc_id", "clause_no", "has_table"):
                cl.create_payload_index(BENCH, field_name=field, field_schema=schema)


REPEATS = 3


def timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    dim = Embedder().dim
    print(f"размерность {dim}, пойнтов в замере {n}\n")

    pts = synth(n, dim)
    grpc = store.client(prefer_grpc=True)
    rest = store.client(prefer_grpc=False)

    rows = []

    def bench(label, cl, threshold, **kw):
        """Медиана из REPEATS прогонов: разброс между прогонами доходит до 40 %,
        и по одному замеру отличить приём от шума нельзя."""
        times = []
        for _ in range(REPEATS):
            fresh(cl, dim, threshold)
            times.append(timed(lambda: store.upload(cl, BENCH, pts, **kw)))
        sec = median(times)
        rows.append((label, sec, n / sec))
        print(f"  {label:44s} {sec:7.2f} с   {n / sec:8.0f} пойнт/с"
              f"   (разброс {min(times):.2f}–{max(times):.2f})")

    # --- 1. наивно против пакетно -----------------------------------------
    print(f"1. Наивная загрузка против пакетной (на {NAIVE_N} пойнтах)")
    small = pts[:NAIVE_N]
    fresh(grpc, dim, 0)
    sec = timed(lambda: store.upload_naive(grpc, BENCH, small))
    print(f"  {'по одному, upsert(wait=True)':44s} {sec:7.2f} с   {NAIVE_N / sec:8.0f} пойнт/с")
    naive_rate = NAIVE_N / sec

    fresh(grpc, dim, 0)
    sec = timed(lambda: store.upload(grpc, BENCH, small, batch_size=128))
    print(f"  {'пакетно, batch=128, wait=False':44s} {sec:7.2f} с   {NAIVE_N / sec:8.0f} пойнт/с")
    print(f"  -> пакетная быстрее в {(NAIVE_N / sec) / naive_rate:.0f} раз\n")

    # --- 2. gRPC против REST ----------------------------------------------
    print(f"2. Транспорт (batch=128, wait=False, порог 0), {n} пойнтов")
    bench("gRPC", grpc, 0, batch_size=128)
    bench("REST", rest, 0, batch_size=128)
    print()

    # --- 3. wait ----------------------------------------------------------
    print("3. Параметр wait (gRPC, batch=128, порог 0)")
    bench("wait=False", grpc, 0, batch_size=128, wait=False)
    bench("wait=True", grpc, 0, batch_size=128, wait=True)
    print()

    # --- 4. размер батча --------------------------------------------------
    print("4. Размер батча (gRPC, wait=False, порог 0)")
    for bs in (32, 64, 128, 256, 512):
        bench(f"batch_size={bs}", grpc, 0, batch_size=bs)
    print()

    # --- 5. параллелизм ---------------------------------------------------
    # parallel > 1 порождает ПРОЦЕССЫ; работает только из-под __main__
    print("5. Параллелизм (gRPC, batch=128, wait=False, порог 0)")
    for par in (1, 2, 4):
        bench(f"parallel={par}", grpc, 0, batch_size=128, parallel=par)
    print()

    # --- 6. порог индексации ----------------------------------------------
    # Сравнивать 0 с рабочими 20000 бессмысленно: при 20000 индексация
    # на этом объёме и так не запускается (см. пункт 7), то есть оба режима
    # одинаковы. Чтобы приём вообще было с чем сравнивать, нужен порог,
    # который индексацию включает.
    print("6. Порог индексации во время загрузки (gRPC, batch=128)")
    bench("indexing_threshold=0 (индексация выключена)", grpc, 0, batch_size=128)
    t_off = rows[-1][1]
    bench("indexing_threshold=1000 (индексация идёт)", grpc, 1000, batch_size=128)
    t_on = rows[-1][1]
    print(f"  -> выключенная индексация экономит {(t_on - t_off) / t_on * 100:+.0f} % времени заливки")
    print("     (у рабочего порога 20000 КБ разницы с нулём нет вовсе:")
    print("      сегменты до него не дорастают, индексация не начинается)")

    # Но экономия не бесплатна: после загрузки граф всё равно надо построить.
    # А построится ли он вообще — зависит от размера СЕГМЕНТА, не коллекции.
    print("\n7. Сколько стоит построение графа после заливки")
    for threshold in (store.INDEXING_THRESHOLD, 1000):
        fresh(grpc, dim, 0)
        store.upload(grpc, BENCH, pts, batch_size=128)
        info = store.finish_indexing(grpc, BENCH, dim, threshold=threshold)
        print(f"  порог {threshold} КБ:")
        for k, v in info.items():
            print(f"    {k}: {v}")

    grpc.delete_collection(BENCH)
    print(f"\nвременная коллекция {BENCH} удалена")


if __name__ == "__main__":
    main()
