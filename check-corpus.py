"""Проверка золотого набора на согласованность с корпусом.

Битая ссылка в золотом наборе не вызывает ошибок — она просто занижает recall
и заставляет неделю искать проблему в пайплайне. Дешевле проверить сразу.

Запуск: uv run python check-corpus.py
"""

import re
import sys
from pathlib import Path

import yaml

CORPUS = Path("corpus")

# doc_id → файл. SH-45 в .docx: нумерация автоматическая, номеров в тексте нет,
# поэтому ссылки на его пункты позиционные и проверяются только на префикс.
DOCS = {
    "TR-114": CORPUS / "TR-114-2026.md",
    "TR-114-DS1": CORPUS / "TR-114-2026-DS1.md",
    "EKS-77": CORPUS / "EKS-77-2026.md",
    "AR-203": CORPUS / "AR-203-2026.md",
    "SH-45": CORPUS / "SH-45-2026.docx",
}
POSITIONAL = {"SH-45"}     # ссылки позиционные, номер в тексте не ищем


def load_text(path: Path) -> str:
    if path.suffix == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(path).paragraphs)
    return path.read_text(encoding="utf-8")


def clause_present(text: str, clause: str) -> bool:
    """Есть ли в тексте пункт с таким номером в начале строки."""
    pattern = rf"(?m)^\s*{re.escape(clause)}\.?\s"
    return re.search(pattern, text) is not None


def main() -> int:
    data = yaml.safe_load((CORPUS / "golden-set.yaml").read_text(encoding="utf-8"))
    questions = data["questions"]

    texts, missing_files = {}, []
    for doc_id, path in DOCS.items():
        if path.exists():
            texts[doc_id] = load_text(path)
        else:
            missing_files.append(f"{doc_id} → {path}")

    problems, checked, skipped = [], 0, 0
    seen_docs = set()

    for q in questions:
        for ref in q.get("relevant", []) + q.get("superseded", []):
            doc_id, _, clause = ref.partition("#")
            seen_docs.add(doc_id)
            if doc_id not in DOCS:
                problems.append(f"неизвестный документ {doc_id!r} в ссылке {ref}")
                continue
            if doc_id not in texts:
                continue
            if doc_id in POSITIONAL:
                skipped += 1
                continue
            checked += 1
            if not clause_present(texts[doc_id], clause):
                problems.append(f"пункт не найден: {ref}  (вопрос: {q['q'][:45]}…)")

    print(f"вопросов:            {len(questions)}")
    print(f"ссылок проверено:    {checked}")
    print(f"ссылок пропущено:    {skipped}  (позиционные, в файле нет номеров)")
    print(f"документов задето:   {len(seen_docs)} из {len(DOCS)}")

    unused = set(DOCS) - seen_docs
    if unused:
        print(f"\n⚠️  документы без единого вопроса: {', '.join(sorted(unused))}")
    if missing_files:
        print("\n❌ отсутствуют файлы:")
        for m in missing_files:
            print("   ", m)
    if problems:
        print(f"\n❌ проблем: {len(problems)}")
        for p in problems:
            print("   ", p)
        return 1

    print("\n✅ золотой набор согласован с корпусом")
    return 0


if __name__ == "__main__":
    sys.exit(main())
