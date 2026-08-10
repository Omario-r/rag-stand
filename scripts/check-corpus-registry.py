"""Сверка документов корпуса с реестром: те ли реквизиты попали в текст.

ЗАЧЕМ ОТДЕЛЬНАЯ ПРОВЕРКА. Ни одна из существующих её не заменяет:
`check-parser.py` отвечает на вопрос «разобралось ли», `make-registry.py
--check` — «сходится ли реестр сам с собой». Вопрос «стоит ли в договоре тот
ИНН, который назначен этому делу» до сих пор не задавал никто, а ошибка здесь
не видна глазами: документ с чужим ИНН читается как обычный документ.

Цена ошибки — не косметическая. ИНН связывает договор с контрагентом в SQL;
подставленный из другого дела даёт правдоподобную, но неверную связь,
и заметить это можно будет только по метрике, которая уже испорчена.

Проверяется:

1. ИНН обеих сторон присутствует в тексте документа;
2. денежные суммы дела встречаются с прописью, и пропись сходится с цифрами;
3. номер договора стоит в документе;
4. допсоглашение и приложение ссылаются на номер своего договора.

ПОЧЕМУ СУММЫ НЕ СВЕРЯЮТСЯ СТРОКОЙ. Первая редакция этой проверки требовала
поле `в_тексте` дословно и обвинила пять документов из сорока восьми. Все
пять были правы, а неправа была проверка: в живом предложении сумма
склоняется — «уплачивает 1 800 (одну тысячу восемьсот) рублей», «в размере
250 (двухсот пятидесяти) рублей», — а реестр хранит именительный падеж.
Поэтому сверяется не строка, а значение: фрагмент разбирается
`extract.parse_amount_ru` — тем же кодом, который читает суммы в пайплайне, —
и требуется, чтобы цифры, пропись и реестр дали одно и то же число.

Запуск:  uv run python scripts/check-corpus-registry.py
"""

import re
import sys
from pathlib import Path

import yaml

STAND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAND / "src"))

from rag_lab.chunk import parse  # noqa: E402
from rag_lab.extract import parse_amount_ru  # noqa: E402

CORPUS = STAND / "corpus"

# «4 500 000 (четыре миллиона пятьсот тысяч)» — цифры и пропись рядом.
MONEY = re.compile(r"\d[\d  ]*\([^)]{3,120}\)")

# У SH-45 номера договора в извлечённом тексте нет и быть не может: файл
# создан со стилями автонумерации Word, и `python-docx` возвращает пункты
# без единой цифры. Это структурная ловушка партии 1, а не дефект документа.
NO_NUMBER_IN_TEXT = {"SH-45-2026.docx"}


def amounts_in(body: str) -> set[float]:
    """Числа, у которых цифры и пропись сошлись между собой.

    Расхождение цифр с прописью пайплайн ловит и показывает человеку —
    здесь оно означает, что автор написал пропись руками и ошибся.
    """
    good = set()
    for frag in MONEY.findall(body):
        a = parse_amount_ru(frag)
        if a.digits is not None and a.digits == a.words:
            good.add(a.digits)
    return good


def text_of(name: str) -> str:
    """Текст документа целиком. `.docx` читается тем же парсером, что и `.md`."""
    if name.endswith(".docx"):
        title, clauses = parse(CORPUS / name)
        return title + "\n" + "\n".join(c.text for c in clauses)
    return (CORPUS / name).read_text(encoding="utf-8")


def main() -> None:
    data = yaml.safe_load((CORPUS / "registry.yaml").read_text(encoding="utf-8"))
    problems: list[str] = []
    checked = 0

    for case in data["дела"]:
        main_file = case.get("файл")
        if not main_file or not (CORPUS / main_file).exists():
            continue
        checked += 1
        body = text_of(main_file)

        for side in case.get("стороны", []):
            if side["инн"] not in body:
                problems.append(
                    f"{main_file}: нет ИНН {side['инн']} ({side['роль']}, "
                    f"{side['название']})")

        if case["номер"] not in body and main_file not in NO_NUMBER_IN_TEXT:
            problems.append(f"{main_file}: нет номера договора {case['номер']}")

        seen = amounts_in(body)
        for key, m in case.items():
            if isinstance(m, dict) and "число" in m and m["число"] not in seen:
                problems.append(
                    f"{main_file}: суммы «{key}» = {m['число']} нет в тексте "
                    "с прописью, сходящейся с цифрами")

        # Документ, у которого нет своего договора в тексте, теряет связь
        # с делом: для допсоглашения это единственное место, где она записана.
        for key, what in (("файл_допсоглашения", "допсоглашение"),
                          ("файл_приложения", "приложение")):
            name = case.get(key)
            if not name:
                continue
            if not (CORPUS / name).exists():
                problems.append(f"{main_file}: {what} обещано реестром, "
                                f"но файла {name} нет")
                continue
            checked += 1
            sub = text_of(name)
            if case["номер"] not in sub:
                problems.append(f"{name}: не ссылается на номер {case['номер']} "
                                "— связь с договором потеряна")
            for side in case.get("стороны", []):
                if side["инн"] not in sub:
                    problems.append(f"{name}: нет ИНН {side['инн']} "
                                    f"({side['роль']})")

    for p in problems:
        print("  ✗", p)
    print(f"\nсверено документов: {checked} — "
          + ("расхождений нет" if not problems else f"расхождений {len(problems)}"))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
