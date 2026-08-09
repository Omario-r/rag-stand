"""Схема БД под договорное дело: история значений, а не одно значение.

Этап 7. Разбор — RAG/07-extraction.md, лаба — RAG/lab-07-extraction.md.

ЗАЧЕМ ТАК СЛОЖНО. Для крупной компании «договор» — это не файл, а ДЕЛО:
основной договор плюс допсоглашения, приложения, спецификации. Допсоглашение
меняет условия основного договора — срок, цену, ставку простоя.

Схема «одна строка = один файл» на этом ломается: система покажет цену
из основного договора, хотя полгода назад её изменили допником. Для задачи
«отслеживание договоров» это отказ продукта, а не мелкий дефект.

Поэтому SQL хранит не значение поля, а ИСТОРИЮ значений с датами,
а текущее вычисляется: значение из документа с наибольшей датой, где это
поле определено. Дороже в разработке; альтернатива — система, которой
нельзя доверять по определению.

Значение без прошедшей проверку цитаты сюда не попадает вовсе — для него
есть очередь на подтверждение человеком (`pending`).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
-- Дело: основной договор со всеми документами к нему
CREATE TABLE IF NOT EXISTS contract_case (
    case_id     TEXT PRIMARY KEY,
    title       TEXT
);

-- Контрагент опознаётся по ИНН, а не по названию: названий у одной
-- компании три штуки, ИНН один и проверяется контрольной суммой
CREATE TABLE IF NOT EXISTS party (
    inn         TEXT PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_party (
    case_id     TEXT NOT NULL REFERENCES contract_case(case_id),
    inn         TEXT NOT NULL REFERENCES party(inn),
    role        TEXT,                    -- как названа сторона в договоре
    PRIMARY KEY (case_id, inn)
);

-- Документ дела. doc_date задаёт порядок применения правок
CREATE TABLE IF NOT EXISTS document (
    doc_id      TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL REFERENCES contract_case(case_id),
    kind        TEXT NOT NULL,           -- основной | допсоглашение | приложение
    doc_date    TEXT NOT NULL,           -- ISO; по нему считается «текущее»
    effective_from TEXT                  -- дата вступления в силу, если отличается
);

-- Извлечённое значение поля. Одна строка = один факт из одного документа.
-- Строк на одно поле может быть много — это и есть история.
CREATE TABLE IF NOT EXISTS extracted_field (
    doc_id      TEXT NOT NULL REFERENCES document(doc_id),
    field       TEXT NOT NULL,
    value       TEXT,                    -- нормализованное значение
    raw         TEXT,                    -- как было написано в документе
    quote       TEXT NOT NULL,           -- цитата, прошедшая проверку вхождения
    clause      TEXT,                    -- номер пункта — ссылка для аудита
    operation   TEXT NOT NULL DEFAULT 'устанавливает',
                                         -- устанавливает | изменяет | исключает
    PRIMARY KEY (doc_id, field)
);

-- Очередь на подтверждение человеком: значения без прошедшей цитаты,
-- нераспознанные даты, ИНН с непройденной контрольной суммой
CREATE TABLE IF NOT EXISTS pending (
    doc_id      TEXT NOT NULL,
    field       TEXT NOT NULL,
    value       TEXT,
    quote       TEXT,
    reason      TEXT NOT NULL,
    PRIMARY KEY (doc_id, field)
);
"""

# Эффективное значение поля: из документа с наибольшей датой, где поле
# определено. Исключающая правка гасит поле — поэтому operation попадает
# в выборку, а решение принимает вызывающий код.
Q_EFFECTIVE = """
SELECT f.field, f.value, f.raw, f.quote, f.clause, f.operation, d.doc_id, d.doc_date
FROM extracted_field f
JOIN document d ON d.doc_id = f.doc_id
WHERE d.case_id = ?
  AND d.doc_date = (
      SELECT MAX(d2.doc_date) FROM extracted_field f2
      JOIN document d2 ON d2.doc_id = f2.doc_id
      WHERE d2.case_id = d.case_id AND f2.field = f.field
  )
ORDER BY f.field
"""

Q_HISTORY = """
SELECT d.doc_date, d.doc_id, d.kind, f.value, f.raw, f.operation, f.clause
FROM extracted_field f
JOIN document d ON d.doc_id = f.doc_id
WHERE d.case_id = ? AND f.field = ?
ORDER BY d.doc_date
"""


def connect(path: str | Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(SCHEMA)
    return db


def put_field(db, doc_id, field, value, raw, quote, clause, document_text,
              operation="устанавливает"):
    """Записать значение — но только если цитата прошла проверку.

    Не прошла — строка уходит в `pending`, а не в `extracted_field`.
    Это единственное место, где значение попадает в базу, и барьер стоит
    именно здесь, а не «где-нибудь в пайплайне»: барьер, который можно
    обойти, барьером не является.
    """
    from .extract import quote_ok

    if not quote_ok(document_text, quote):
        db.execute(
            "INSERT OR REPLACE INTO pending (doc_id, field, value, quote, reason)"
            " VALUES (?,?,?,?,?)",
            (doc_id, field, str(value), quote, "цитата не найдена в документе"),
        )
        return False
    db.execute(
        "INSERT OR REPLACE INTO extracted_field"
        " (doc_id, field, value, raw, quote, clause, operation) VALUES (?,?,?,?,?,?,?)",
        (doc_id, field, None if value is None else str(value), raw, quote, clause, operation),
    )
    return True


def effective(db, case_id: str) -> dict[str, sqlite3.Row]:
    """Текущие значения всех полей дела."""
    return {r["field"]: r for r in db.execute(Q_EFFECTIVE, (case_id,))}


def history(db, case_id: str, field: str) -> list[sqlite3.Row]:
    """Как значение поля менялось. То, что юрист и хочет увидеть."""
    return list(db.execute(Q_HISTORY, (case_id, field)))


def naive_by_file(db, doc_id: str) -> dict[str, str]:
    """Схема «одна строка = один файл» — для сравнения, как делать НЕ надо."""
    return {
        r["field"]: r["value"]
        for r in db.execute("SELECT field, value FROM extracted_field WHERE doc_id = ?", (doc_id,))
    }
