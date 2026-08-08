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

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
# Ожидания из corpus/README.md — если парсер разойдётся, видно сразу
EXPECTED = {"TR-114-2026": 46, "EKS-77-2026": 33, "AR-203-2026": 47, "TR-114-2026-DS1": 11}
# doc_id в золотом наборе короче имени файла
DOC_ID = {"TR-114-2026": "TR-114", "EKS-77-2026": "EKS-77",
          "AR-203-2026": "AR-203", "TR-114-2026-DS1": "TR-114-DS1"}

print("=== разбор ===")
produced: set[str] = set()
for name, want in EXPECTED.items():
    doc_title, clauses = parse(CORPUS / f"{name}.md")
    produced |= {chunk_id(DOC_ID[name], c) for c in clauses}
    levels = Counter(c.level for c in clauses)
    over = [c.number for c in clauses if estimate_tokens(c.text) > MAX_TOKENS]

    mark = "ок " if len(clauses) == want else "РАЗОШЛОСЬ"
    print(f"{mark} {name}: {len(clauses)}/{want}  уровни {dict(sorted(levels.items()))}"
          f"  таблиц {sum(c.has_table for c in clauses)}"
          + (f"  ПРЕВЫШАЮТ {MAX_TOKENS} ток.: {over}" if over else ""))
    if clauses:
        seq = next((c for c in clauses if c.is_sequential), clauses[0])
        print(f"     id={chunk_id(DOC_ID[name], seq)!r}  показ={display(DOC_ID[name], seq)!r}")

print("\n=== сверка с золотым набором ===")
gs = yaml.safe_load((CORPUS / "golden-set.yaml").read_text(encoding="utf-8"))
refs: set[str] = set()
for q in gs["questions"]:
    refs |= set(q.get("relevant", [])) | set(q.get("superseded", []))

sh45 = {r for r in refs if r.startswith("SH-45#")}
checkable = refs - sh45
missing = sorted(checkable - produced)

print(f"ссылок в наборе: {len(refs)}, из них на SH-45 (.docx, ещё не разбирается): {len(sh45)}")
print(f"проверяемых сейчас: {len(checkable)}")
if missing:
    print(f"НЕ НАЙДЕНО парсером: {len(missing)} -> {missing}")
else:
    print("все проверяемые ссылки золотого набора парсер воспроизводит")
