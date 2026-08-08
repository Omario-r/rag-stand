"""Разбор договора на дерево пунктов и нарезка на чанки.

Этап 3 роадмапа. Разбор решений — RAG/02-chunking.md.

Правило, вокруг которого всё построено: **чанк = смысловая единица документа
(пункт/подпункт), а не N символов**. Резка по символам ломает пункт посередине,
теряет иерархию и уводит номер в соседний чанк — все три дефекта не лечатся
ни моделью получше, ни реранкером.

Три вещи, которые этот модуль обязан делать правильно:

1. НОМЕР ПУНКТА — это адрес ответа. Юристу нужен «п. 4.2», а не «где-то
   в договоре». Если парсер потерял номер, чанк бесполезен, даже если текст
   в нём верный.

2. ЛИМИТ ДЛИНЫ СЧИТАЕТСЯ В ТОКЕНАХ, а не в символах. Замер 08.08.2026:
   переводной коэффициент непостоянен — 5,2 симв./токен на гладкой
   юридической прозе против 3,4 на тексте с номерами и датами. Настройка
   «2180 символов» проходит на кодексе и выходит за 512 токенов на всех
   наших договорах. Отсюда `count_tokens` — параметр, а не константа.

3. ТАБЛИЦА ОСТАЁТСЯ ПРИ СВОЁМ ПУНКТЕ. В AR-203 таблица стоит между 3.1 и 3.2
   и разрывает раздел; строка «Москва — Казань | 45 000» отдельным чанком
   не значит ничего.

Три места, помеченные РЕШЕНИЕ, ждут ответа Омара и вынесены в флаги — ядро
парсера от них не зависит, поэтому пишется и проверяется до ответа.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Предел эмбеддера в токенах. У multilingual-e5-large — 512, у Qwen3 больше,
# но целимся в меньший: конфигурация партнёра должна проходить тоже.
MAX_TOKENS = 512

# Целевой размер чанка (RAG/02-chunking.md). Точные числа подбираются замером
# на золотом наборе, эти — стартовые.
TARGET_TOKENS = 350
MIN_TOKENS = 40

# Консервативная оценка, когда токенизатора под рукой нет: берём худший
# из двух замеренных коэффициентов, чтобы не превысить предел молча.
CHARS_PER_TOKEN_WORST = 3.4


def estimate_tokens(text: str) -> int:
    """Оценка числа токенов по символам — заведомо с запасом.

    Нужна, чтобы разбор договора не требовал поднятого llama.cpp. Когда сервер
    есть, вместо этой функции передаётся точный счётчик: см. `tokens_via_server`.
    """
    return int(len(text) / CHARS_PER_TOKEN_WORST) + 1


def tokens_via_server(embedder) -> "callable":
    """Точный счётчик токенов через /tokenize работающего llama.cpp.

    Возвращает функцию, а не число: подставляется в `chunk_document`
    вместо `estimate_tokens`, когда стенд поднят.
    """

    def count(text: str) -> int:
        return len(embedder._post("/tokenize", {"content": text})["tokens"])

    return count


# --- распознавание структуры ---

# Заголовок раздела: "## 4. ОТВЕТСТВЕННОСТЬ СТОРОН" или "## Статья 3. Название"
RE_SECTION = re.compile(r"^#{1,6}\s+(?:(Статья|Раздел|Приложение)\s+)?(\d+)\.?\s*(.*)$")
# Заголовок документа: единственная "# ..." в начале файла
RE_TITLE = re.compile(r"^#\s+(?!\d)(.+)$")
# Номер пункта в начале абзаца: "4.2." / "3.4.3.1." / "17." — точка в конце
# необязательна, но точки внутри обязаны разделять только цифры.
RE_CLAUSE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(?=[«А-ЯЁA-Z])")
# Строка таблицы Markdown
RE_TABLE = re.compile(r"^\s*\|")


@dataclass
class Clause:
    """Пункт договора — узел дерева."""

    number: str  # "4.2" или "13" при сквозной нумерации; "" если номера нет
    text: str  # чистый текст пункта, без служебной шапки
    section_no: str = ""  # номер раздела/статьи, в котором лежит пункт
    section_title: str = ""
    level: int = 1  # глубина: 4.2 -> 2, 3.4.3.1 -> 4
    parent: str = ""  # number родителя, "" у верхнего уровня
    has_table: bool = False
    children: list[str] = field(default_factory=list)

    @property
    def is_sequential(self) -> bool:
        """Сквозная нумерация: номер пункта не связан с номером раздела.

        В EKS-77 пункты идут 1…33, а разделы называются «Статья N».
        Парсер, строящий дерево по точкам в номере, здесь дерева не построит.
        """
        return self.level == 1 and self.section_no and self.number != self.section_no


def parse(path: str | Path) -> tuple[str, list[Clause]]:
    """Файл договора -> (заголовок документа, плоский список пунктов).

    Дерево задаётся полями `parent`/`children`, а не вложенностью: так проще
    и сериализовать в payload, и обходить.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    doc_title = ""
    section_no = section_title = ""
    clauses: list[Clause] = []
    current: Clause | None = None
    buf: list[str] = []

    def flush():
        nonlocal current, buf
        if current is not None:
            current.text = "\n".join(buf).strip()
            clauses.append(current)
        current, buf = None, []

    for line in lines:
        if not doc_title and (m := RE_TITLE.match(line)):
            doc_title = m.group(1).strip()
            continue

        if m := RE_SECTION.match(line):
            flush()
            section_no, section_title = m.group(2), m.group(3).strip()
            continue

        if m := RE_CLAUSE.match(line):
            flush()
            number = m.group(1)
            current = Clause(
                number=number,
                text="",
                section_no=section_no,
                section_title=section_title,
                level=number.count(".") + 1,
                parent=number.rsplit(".", 1)[0] if "." in number else "",
            )
            buf = [line.strip()]
            continue

        if current is not None:
            if RE_TABLE.match(line):
                current.has_table = True
            buf.append(line)

    flush()

    # связать детей с родителями
    by_number = {c.number: c for c in clauses}
    for c in clauses:
        if c.parent and c.parent in by_number:
            by_number[c.parent].children.append(c.number)
        elif c.parent:
            # родителя нет в файле: "3.2.5" без "3.2" — не ошибка парсера,
            # так бывает в допсоглашениях, которые дополняют чужой документ
            c.parent = ""

    return doc_title, clauses


def address(doc_id: str, c: Clause, sequential_style: str = "section+clause") -> str:
    """Человекочитаемый адрес пункта — то, что увидит юрист в ответе.

    РЕШЕНИЕ 1 (ждёт ответа Омара): как адресовать пункт при сквозной
    нумерации, где номер пункта с номером статьи не связан.
      "section+clause" — «Статья 3, п. 13»: точнее для человека, но выглядит
                         как иерархия, которой в документе нет;
      "clause"         — «п. 13»: честно повторяет документ, но не даёт
                         контекста при показе списка находок.
    Ядро парсера от выбора не зависит: меняется только эта функция.
    """
    if not c.number:
        return f"{doc_id}, без номера"
    if c.is_sequential and sequential_style == "section+clause":
        return f"{doc_id}, Статья {c.section_no}, п. {c.number}"
    return f"{doc_id}, п. {c.number}"


def contextualize(doc_title: str, c: Clause, addr: str) -> str:
    """Текст, который пойдёт в вектор: адрес + раздел + сам пункт.

    Дёшево, а качество поиска растёт заметно — вектор «знает», что речь
    о неустойке в договоре перевозки, даже если слова «перевозка» в пункте
    нет (RAG/02-chunking.md). Чистый текст пункта хранится отдельно:
    он идёт в контекст модели и в цитаты, там шапка не нужна.
    """
    head = doc_title
    if c.section_no:
        head += f"\nРаздел {c.section_no}. {c.section_title} → {addr}"
    return f"{head}\n\n{c.text}"
