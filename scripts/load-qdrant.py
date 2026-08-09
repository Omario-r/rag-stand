"""Корпус -> чанки -> векторы -> Qdrant. Этап 4, рабочая загрузка.

Одна команда, после которой в коллекции лежит весь корпус с payload,
payload-индексами и построенным HNSW. Повторный запуск не плодит дубли:
идентификатор пойнта — UUID5 от адреса чанка.

Запуск (стенд поднят, .env загружен через set -a):

    uv run python scripts/load-qdrant.py            # обновить
    uv run python scripts/load-qdrant.py --recreate # снести и залить заново

`--recreate` нужен, когда изменился состав или адресация чанков: без него
пойнты старой нарезки останутся в коллекции и будут находиться поиском.

Весь код лежит под `if __name__ == "__main__"` намеренно: `upload` с
`parallel > 1` порождает процессы, и без этой защиты каждый работник
выполнил бы скрипт целиком (см. rag_lab/store.upload).
"""

import sys
import time
from pathlib import Path

STAND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAND / "src"))

from rag_lab import store  # noqa: E402
from rag_lab.corpus import build_chunks, coverage  # noqa: E402
from rag_lab.embed import Embedder  # noqa: E402


def main() -> None:
    recreate = "--recreate" in sys.argv

    chunks = build_chunks()
    covered, refs = coverage(chunks)
    missing = sorted(refs - covered)
    print(f"чанков: {len(chunks)}   ссылок золотого набора: {len(refs)}")
    if missing:
        # Непокрытая ссылка занижает recall молча: метрика посчитается,
        # цифра выйдет правдоподобная, и разбираться в ней никто не станет.
        sys.exit(f"НЕ ПОКРЫТО ссылок: {len(missing)} -> {missing[:5]}")

    emb = Embedder()
    dim = emb.dim
    print(f"размерность вектора: {dim} — прочитана у эмбеддера, не задана константой")

    t0 = time.perf_counter()
    vectors = emb.embed([ch.embed_text for ch in chunks], kind="document")
    print(f"векторизация: {time.perf_counter() - t0:.2f} с   кэш: {emb.stats()}")

    cl = store.client()
    store.create_collection(cl, store.COLLECTION, dim, recreate=recreate)

    points = store.to_points(chunks, vectors)
    t0 = time.perf_counter()
    store.upload(cl, store.COLLECTION, points)
    print(f"загрузка: {time.perf_counter() - t0:.2f} с ({len(points)} пойнтов)")

    print("возвращаю порог индексации и жду построения индекса...")
    for k, v in store.finish_indexing(cl, store.COLLECTION, dim).items():
        print(f"  {k}: {v}")

    # --- проверки, без которых загрузка не считается успешной
    info = cl.get_collection(store.COLLECTION)
    schema = sorted(info.payload_schema)
    print(f"\npayload-индексы: {schema}")
    if set(schema) != set(store.PAYLOAD_INDEXES):
        print(f"  ВНИМАНИЕ: ожидались {sorted(store.PAYLOAD_INDEXES)}")

    exact = cl.count(store.COLLECTION, exact=True).count
    print(f"пойнтов в коллекции (точный счёт): {exact}")
    if exact != len(chunks):
        print(f"  ВНИМАНИЕ: чанков {len(chunks)}, пойнтов {exact} — расхождение")

    # идемпотентность проверяется не рассуждением, а вторым проходом
    store.upload(cl, store.COLLECTION, points)
    again = cl.count(store.COLLECTION, exact=True).count
    print(f"после повторной загрузки тех же пойнтов: {again}"
          + ("  (дублей нет)" if again == exact else "  ПОЯВИЛИСЬ ДУБЛИ"))


if __name__ == "__main__":
    main()
