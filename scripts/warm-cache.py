"""Наполняет кэш векторов помаленьку, переживая падения сервера.

Зачем отдельный скрипт. На локальном стенде векторизация корпуса — один
вызов `emb.embed(...)` внутри `load-qdrant.py`, и делать вокруг него нечего.
На контуре партнёра иначе: его эмбеддер (сборка `b7176`) умирает после
~6000 суммарно векторизованных токенов и поднимается супервизором. Корпус
из 144 чанков — это ~12 000 токенов, то есть минимум одно падение на проход,
и загрузка «одним вызовом» не заканчивается никогда.

Отсюда разделение: этот скрипт отвечает только за то, чтобы все векторы
оказались в кэше, а `load-qdrant.py` потом отрабатывает на попаданиях в кэш
и падений уже не видит.

Запуск:

    uv run python scripts/warm-cache.py            # до победы
    uv run python scripts/warm-cache.py --passes 3 # не более трёх заходов

Идемпотентен: повторный запуск на полном кэше ничего не делает и ничего
не стоит.
"""

import sys
import time
from pathlib import Path

STAND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAND / "src"))

from rag_lab.corpus import build_chunks  # noqa: E402
from rag_lab.embed import Embedder, text_for  # noqa: E402


def main() -> None:
    passes = 100
    if "--passes" in sys.argv:
        passes = int(sys.argv[sys.argv.index("--passes") + 1])

    chunks = build_chunks()
    emb = Embedder()
    print(f"контур: {emb.url}")
    print(f"модель: {emb.identity()}")

    prepared = [text_for("document", c.embed_text, emb.scheme) for c in chunks]
    keys = [emb._key("document", t) for t in prepared]

    t_start = time.perf_counter()
    crashes = 0

    for n in range(1, passes + 1):
        have = emb._cached(keys)
        todo = [(k, t) for k, t in zip(keys, prepared) if k not in have]
        if not todo:
            print(f"\nвсе {len(keys)} векторов в кэше "
                  f"(заходов {n - 1}, падений {crashes}, "
                  f"{time.perf_counter() - t_start:.0f} с)")
            return

        print(f"\nзаход {n}: в кэше {len(have)}/{len(keys)}, осталось {len(todo)}")
        done = 0
        for part in emb._batches(todo):
            try:
                body = {"model": "default", "input": [t for _, t in part]}
                resp = emb._post("/v1/embeddings", body)
            except Exception as e:
                crashes += 1
                print(f"  оборвалось после {done} в этом заходе "
                      f"({type(e).__name__}), жду сервер")
                emb._wait_ready()
                break
            by_index = {d["index"]: d["embedding"] for d in resp["data"]}
            emb._store([(k, by_index[j]) for j, (k, _) in enumerate(part)])
            done += len(part)
            print(f"\r  +{done}", end="", flush=True)
        else:
            print()

    have = emb._cached(keys)
    sys.exit(f"\nне сошлось за {passes} заходов: в кэше {len(have)}/{len(keys)}")


if __name__ == "__main__":
    main()
