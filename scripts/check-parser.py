"""Проверка парсера на корпусе и сверка адресов с золотым набором.

Вторая часть важнее первой: если идентификатор чанка не совпадает со ссылкой
из golden-set.yaml, recall не посчитать вовсе — совпадений не будет.

Запуск:  uv run python scripts/check-parser.py
"""

import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_lab.chunk import MAX_TOKENS, chunk_id, display, estimate_tokens, parse  # noqa: E402
from rag_lab.corpus import DOCS  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
# Число пунктов известно заранее только у партии 1 — оно записано
# в corpus/README.md, и расхождение означает, что парсер поехал.
# У партии 2 сверять не с чем: документы пишутся, а не описаны заранее,
# поэтому там проверяется только то, что разбор вообще состоялся.
EXPECTED = {"TR-114-2026.md": 46, "EKS-77-2026.md": 33, "AR-203-2026.md": 47,
            "TR-114-2026-DS1.md": 11, "SH-45-2026.docx": 31}

print("=== разбор ===")
produced: set[str] = set()
for name, (doc_id, _) in DOCS.items():
    want = EXPECTED.get(name)
    doc_title, clauses = parse(CORPUS / name)
    produced |= {chunk_id(doc_id, c) for c in clauses}
    levels = Counter(c.level for c in clauses)
    over = [c.number for c in clauses if estimate_tokens(c.text) > MAX_TOKENS]

    body = [c for c in clauses if c.part == "договор"]
    app = [c for c in clauses if c.part != "договор"]
    skipped = [c.number for c in clauses if not c.indexed]

    if want is None:
        mark, count = "—  ", f"{len(body)}"
    else:
        mark, count = ("ок " if len(body) == want else "РАЗОШЛОСЬ"), f"{len(body)}/{want}"
    print(f"{mark} {name}: {count}  уровни {dict(sorted(levels.items()))}"
          f"  таблиц {sum(c.has_table for c in clauses)}"
          + (f"  +приложение: {len(app)}" if app else "")
          + (f"  не в индекс: {skipped}" if skipped else "")
          + (f"  ПРЕВЫШАЮТ {MAX_TOKENS} ток.: {over}" if over else ""))

print("\n=== сверка с золотым набором ===")
gs = yaml.safe_load((CORPUS / "golden-set.yaml").read_text(encoding="utf-8"))
refs: set[str] = set()
for q in gs["questions"]:
    refs |= set(q.get("relevant", [])) | set(q.get("superseded", []))

missing = sorted(refs - produced)

print(f"ссылок в наборе: {len(refs)}   пунктов у парсера: {len(produced)}")
if missing:
    print(f"НЕ НАЙДЕНО парсером: {len(missing)} -> {missing}")
else:
    print("все ссылки золотого набора парсер воспроизводит")
