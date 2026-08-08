"""Проверка парсера на корпусе: сколько пунктов найдено и где он спотыкается.

Запуск:  uv run python scripts/check-parser.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_lab.chunk import MAX_TOKENS, address, estimate_tokens, parse  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
# Ожидания из corpus/README.md — если парсер разойдётся, видно сразу
EXPECTED = {"TR-114-2026": 46, "EKS-77-2026": 33, "AR-203-2026": 47, "TR-114-2026-DS1": 11}

total = 0
for name, want in EXPECTED.items():
    doc_title, clauses = parse(CORPUS / f"{name}.md")
    got = len(clauses)
    total += got
    levels = Counter(c.level for c in clauses)
    seq = sum(1 for c in clauses if c.is_sequential)
    tables = sum(1 for c in clauses if c.has_table)
    over = [c for c in clauses if estimate_tokens(c.text) > MAX_TOKENS]
    orphan = sum(1 for c in clauses if c.level > 1 and not c.parent)

    mark = "ок " if got == want else "РАЗОШЛОСЬ"
    print(f"\n{mark} {name}: {got} пунктов (ожидали {want})")
    print(f"    уровни: {dict(sorted(levels.items()))}")
    print(f"    сквозная нумерация: {seq}   с таблицей: {tables}   без родителя: {orphan}")
    if over:
        print(f"    ПРЕВЫШАЮТ {MAX_TOKENS} токенов: {len(over)} -> {[c.number for c in over]}")
    if clauses:
        c = clauses[min(3, len(clauses) - 1)]
        print(f"    пример адреса: {address(name.split('-2026')[0], c)}")

print(f"\nвсего пунктов: {total} (в corpus/README.md заявлено 168 на пять файлов,")
print("из них 31 в SH-45.docx — он здесь не разбирается, это РЕШЕНИЕ 2)")
