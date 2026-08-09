"""Настоящая длина чанков в токенах против оценки по символам.

Этап 3. Отвечает на два вопроса:

1. Проходит ли корпус предел эмбеддера (512 токенов у multilingual-e5-large,
   на который целимся ради конфигурации партнёра)?
2. Насколько врёт `estimate_tokens` — консервативная оценка len/3.4,
   которой пользуется разбор, когда llama.cpp не поднят?

Второе важнее первого: оценка нужна затем, чтобы чанкер работал без стенда,
но если она врёт в опасную сторону, ею нельзя пользоваться вовсе.

Запуск (стенд должен быть поднят):  uv run python scripts/measure-tokens.py
"""

import os
import sys
from pathlib import Path

STAND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAND / "src"))

from rag_lab.chunk import (  # noqa: E402
    CHARS_PER_TOKEN_WORST,
    MAX_TOKENS,
    chunk_document,
    estimate_tokens,
    parse,
    tokens_via_server,
)
from rag_lab.embed import Embedder  # noqa: E402

CORPUS = STAND / "corpus"
DOCS = {
    "TR-114-2026.md": ("TR-114", "перевозка"),
    "EKS-77-2026.md": ("EKS-77", "экспедирование"),
    "AR-203-2026.md": ("AR-203", "аренда_ТС"),
    "TR-114-2026-DS1.md": ("TR-114-DS1", "допсоглашение"),
    "SH-45-2026.docx": ("SH-45", "хранение"),
}

if "EMBEDDER_URL" not in os.environ:
    sys.exit("нет EMBEDDER_URL: source .env перед запуском")

emb = Embedder()
count = tokens_via_server(emb)

chunks = []
for name, (doc_id, doc_type) in DOCS.items():
    title, clauses = parse(CORPUS / name)
    chunks += chunk_document(doc_id, title, clauses, doc_type=doc_type)

rows = []
for ch in chunks:
    # мерим то, что реально уйдёт в вектор, — с шапкой, а не голый текст
    real = count(ch.embed_text)
    rows.append((ch.id, real, estimate_tokens(ch.embed_text), len(ch.embed_text)))

real_sorted = sorted(r[1] for r in rows)
n = len(rows)
print(f"чанков: {n}   предел, на который целимся: {MAX_TOKENS} токенов")
print(
    f"настоящие токены:  мин {real_sorted[0]}  мед {real_sorted[n // 2]}  "
    f"p90 {real_sorted[int(n * 0.9)]}  max {real_sorted[-1]}"
)
over = [r for r in rows if r[1] > MAX_TOKENS]
print(f"длиннее предела: {len(over)}" + (f"  {[r[0] for r in over]}" if over else ""))

print(f"\n--- оценка len/{CHARS_PER_TOKEN_WORST} против замера ---")
under = [r for r in rows if r[2] < r[1]]
print(f"оценка ЗАНИЗИЛА (опасное направление): {len(under)} из {n}")
for cid, real, est, _ in sorted(under, key=lambda r: r[1] - r[2], reverse=True)[:5]:
    print(f"  {cid:16s} замер {real:4d}  оценка {est:4d}  разница {est - real:+d}")

ratios = sorted(chars / real for _, real, _, chars in rows)
print(
    f"\nсимволов на токен: мин {ratios[0]:.2f}  мед {ratios[n // 2]:.2f}  "
    f"max {ratios[-1]:.2f}"
)
print(f"запас оценки в среднем: {sum(r[2] / r[1] for r in rows) / n:.2f}x")

worst = min(rows, key=lambda r: r[3] / r[1])
best = max(rows, key=lambda r: r[3] / r[1])
print(f"\nсамый плотный текст: {worst[0]} — {worst[3] / worst[1]:.2f} симв./токен")
print(f"самый разреженный:   {best[0]} — {best[3] / best[1]:.2f} симв./токен")
