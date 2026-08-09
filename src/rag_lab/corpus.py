"""Корпус: карта документов, сборка чанков, золотой набор.

Одно место, где записано, какие файлы составляют корпус и какой у них
`doc_id`. До этого карта была скопирована в двух скриптах, а нужна в пяти —
и расхождение копий тихо меняло бы состав индекса между замерами.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .chunk import chunk_document, parse

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus"

# файл -> (doc_id, тип договора). doc_id должен совпадать с golden-set.yaml.
DOCS = {
    "TR-114-2026.md": ("TR-114", "перевозка"),
    "EKS-77-2026.md": ("EKS-77", "экспедирование"),
    "AR-203-2026.md": ("AR-203", "аренда_ТС"),
    "TR-114-2026-DS1.md": ("TR-114-DS1", "допсоглашение"),
    "SH-45-2026.docx": ("SH-45", "хранение"),
}


def build_chunks(count_tokens=None) -> list:
    """Весь корпус -> список чанков в устойчивом порядке.

    Порядок важен: по нему считается соответствие векторам, и он же
    определяет порядок пойнтов при загрузке. Словари в питоне упорядочены,
    так что `DOCS` задаёт его однозначно.
    """
    chunks = []
    for name, (doc_id, doc_type) in DOCS.items():
        title, clauses = parse(CORPUS / name)
        kw = {"count_tokens": count_tokens} if count_tokens else {}
        chunks += chunk_document(doc_id, title, clauses, doc_type=doc_type, **kw)
    return chunks


def golden_set() -> list[dict]:
    """Вопросы золотого набора как есть."""
    data = yaml.safe_load((CORPUS / "golden-set.yaml").read_text(encoding="utf-8"))
    return data["questions"]


def coverage(chunks) -> tuple[set[str], set[str]]:
    """(все адреса пунктов, покрытые чанками; ссылки золотого набора).

    Совпадение считается по `clause_ids`, а не по идентификатору чанка:
    вопрос может указывать на подпункт, попавший внутрь чанка родителя.
    """
    covered = set()
    for ch in chunks:
        doc = ch.id.split("#")[0]
        covered |= {f"{doc}#{cid}" for cid in ch.clause_ids}

    refs = set()
    for q in golden_set():
        refs |= set(q.get("relevant", [])) | set(q.get("superseded", []))
    return covered, refs


def chunk_by_clause(chunks) -> dict[str, str]:
    """Адрес пункта -> идентификатор чанка, в который он попал.

    Нужен подсчёту метрик: ссылка золотого набора указывает на пункт,
    а найденным возвращается чанк.
    """
    out = {}
    for ch in chunks:
        doc = ch.id.split("#")[0]
        for cid in ch.clause_ids:
            out[f"{doc}#{cid}"] = ch.id
    return out
