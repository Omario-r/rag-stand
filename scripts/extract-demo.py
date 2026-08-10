"""Извлечение реквизитов договора в SQL. Этап 7.

Разделы 1-6 идут без модели, и их больше, чем самого извлечения:

  1. нормализация кодом — даты, суммы с проверкой прописи, ИНН;
  2. проверка цитат — барьер против тихой галлюцинации;
  3. схема БД с историей значений и сравнение с наивной «строка = файл».

Разделы 7-8 требуют поднятой генеративной модели (`./run-llm.sh`, порт 8084)
и переменной `LLM_URL` в `.env`.

Значения полей ниже взяты из корпуса РУКАМИ — это разметка, а не извлечение.
Она и нужна: без разметки нечем измерять accuracy извлечения, когда модель
появится (пункт 6 этапа 7 роадмапа).

Запуск:
    uv run python scripts/extract-demo.py            # все разделы подряд
    uv run python scripts/extract-demo.py 4          # только раздел 4
    uv run python scripts/extract-demo.py 2 3        # разделы 2 и 3

Разделы независимы и запускаются в любом порядке: раздел 6 поднимает свою
базу с нуля. Это сделано ради лабы — там каждый шаг выполняется отдельной
командой, а не читается как вывод одного большого прогона.
"""

import json
import sys
from pathlib import Path

STAND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAND / "src"))

from rag_lab import db as DB  # noqa: E402
from rag_lab.extract import (  # noqa: E402
    CORE_FIELDS,
    INT,
    MONEY,
    PERCENT,
    TRANSPORT_FIELDS,
    inn_valid,
    json_schema,
    normalize,
    parse_amount_ru,
    parse_date_ru,
    quote_ok,
)

CORPUS = STAND / "corpus"
DB_PATH = STAND / "cache" / "contracts.sqlite"

BASE = (CORPUS / "TR-114-2026.md").read_text(encoding="utf-8")
DS1 = (CORPUS / "TR-114-2026-DS1.md").read_text(encoding="utf-8")


def document() -> tuple[str, str]:
    """Договор для разделов 7 и 8: по умолчанию TR-114, иначе --doc ИМЯ.

    Заведено, чтобы проверить извлечение не на том документе, по которому
    правились описания полей. Правка описаний по одному договору и проверка
    на нём же — подгонка, и увидеть её можно только другим договором.
    """
    if "--doc" in sys.argv:
        name = sys.argv[sys.argv.index("--doc") + 1]
        path = CORPUS / (name if name.endswith(".md") else f"{name}.md")
        return path.name, path.read_text(encoding="utf-8")
    return "TR-114-2026.md", BASE

# Разметка руками: (поле, значение, как написано, цитата, пункт)
BASE_FIELDS = [
    ("amount", 4500000, "4 500 000 (четыре миллиона пятьсот тысяч) рублей 00 копеек",
     "Общая стоимость услуг по настоящему Договору не может превышать 4 500 000", "4.2"),
    ("payment_days", 10, "10 (десяти) банковских дней",
     "в течение 10 (десяти) банковских дней с даты получения счёта", "4.3"),
    ("demurrage_rate", 1500, "1 500 (одной тысячи пятисот) рублей за час",
     "в размере 1 500 (одной тысячи пятисот) рублей за час", "5.2"),
    ("date_end", "2026-12-31", "31 декабря 2026 года",
     "действует до 31 декабря 2026 года", "8.1"),
    ("jurisdiction", "Арбитражный суд города Москвы", "Арбитражного суда города Москвы",
     "спор передаётся на рассмотрение Арбитражного суда города Москвы", "9.2"),
]

DS1_FIELDS = [
    ("amount", 6200000, "6 200 000 (шесть миллионов двести тысяч) рублей 00 копеек",
     "не может превышать 6 200 000 (шесть миллионов двести тысяч) рублей", "1.1"),
    ("payment_days", 20, "20 (двадцати) банковских дней",
     "в течение 20 (двадцати) банковских дней", "1.2"),
    ("demurrage_rate", 1800, "1 800 (одной тысячи восьмисот) рублей за час",
     "в размере 1 800 (одной тысячи восьмисот) рублей за час", "1.3"),
    ("date_end", "2027-06-30", "30 июня 2027 года",
     "действует до 30 июня 2027 года", "1.4"),
]

INNS = ["7701234567", "7809876543", "6316543210", "7810111213",
        "6672334455", "666012345678"]


def part_1() -> None:
    schema = json_schema({**CORE_FIELDS, **TRANSPORT_FIELDS})
    print(f"полей в схеме: {len(schema['properties'])}"
          f"  (ядро {len(CORE_FIELDS)} + перевозка {len(TRANSPORT_FIELDS)})")
    print("устройство одного поля:")
    print(json.dumps(schema["properties"]["amount"], ensure_ascii=False, indent=1))
    print("\nСхема ограничивает вывод, но В ПРОМПТ НЕ ПОДСТАВЛЯЕТСЯ: модель её")
    print("не видит и без словесного описания заполнит валидный JSON чем попало.")


def part_2() -> None:
    for s in ['«12» марта 2026 г.', 'действует до 31 декабря 2026 года',
              '15.06.2026', 'до полного исполнения обязательств']:
        print(f"  {s:46s} -> {parse_date_ru(s)}")
    print("\nNone — не отказ, а честный ответ: неразобранная дата уходит в очередь.")


def part_3() -> None:
    cases = [
        "4 500 000 (четыре миллиона пятьсот тысяч) рублей 00 копеек",
        "6 200 000 (шесть миллионов двести тысяч) рублей 00 копеек",
        "1 500 (одной тысячи пятисот) рублей за час",
        "500 000 (пятьсот тысяч) рублей по каждому случаю",
        "4 500 000 (четыре миллиона пятьдесят тысяч) рублей",   # опечатка в прописи
    ]
    cases.append("0,1 (одной десятой) процента за каждый день просрочки")
    print(f"{'текст':58s} {'цифры':>10} {'пропись':>10}  вердикт")
    for s in cases:
        a = parse_amount_ru(s)
        print(f"  {s[:56]:56s} {a.digits!s:>10} {a.words!s:>10}  {a.verdict}")
    print("\nПредпоследняя строка — договор с опечаткой. Расхождение цифр и прописи")
    print("не ошибка кода, а сигнал: такое значение показывают человеку.")
    print("\nПоследняя — проценты. Вердиктов три, а не два: «не сверено» и")
    print("«расходится» — разные вещи, и слить их в один флаг значит либо")
    print("слать человеку половину корпуса, либо не слать ничего.")


def part_4() -> None:
    print("ИНН из нашего корпуса:")
    for inn in INNS:
        print(f"  {inn:14s} {len(inn)} знаков  {'валиден' if inn_valid(inn) else 'НЕ ВАЛИДЕН'}")
    print("\nНи один не проходит — и это НЕ ошибка кода. ИНН корпуса выдуманы")
    print("при генерации договоров без расчёта контрольной суммы (corpus/README.md).")
    print("Проверка на заведомо валидных, публично известных ИНН:")
    for inn, who in (("7707083893", "Сбербанк"), ("7736050003", "Газпром"),
                     ("500100732259", "12 знаков, ИП"), ("7707083894", "испорчен нарочно")):
        print(f"  {inn:14s} {len(inn)} знаков  "
              f"{'валиден' if inn_valid(inn) else 'НЕ ВАЛИДЕН':10s} {who}")
    print("\nСледствие для этапа 7: на текущем корпусе ВСЕ контрагенты уйдут")
    print("в очередь ручного сопоставления. Партии 2 нужны валидные ИНН,")
    print("иначе связку «контрагент = ИНН» не на чем проверять.")


def part_5() -> None:
    checks = [
        ("настоящая цитата", "Общая стоимость услуг по настоящему Договору не может превышать 4 500 000"),
        ("другие кавычки и пробелы", 'общая стоимость услуг по настоящему договору  не может превышать 4 500 000'),
        ("правдоподобная выдумка", "Общая стоимость услуг по настоящему Договору не может превышать 5 800 000"),
        ("пересказ своими словами", "Договор ограничен суммой четыре с половиной миллиона"),
        ("пустая строка", ""),
    ]
    for label, quote in checks:
        print(f"  {label:26s} {'ПРОШЛА' if quote_ok(BASE, quote) else 'отклонена'}")
    print("\nМодель может выдумать число, но не может выдумать цитату,")
    print("которая пройдёт проверку вхождения.")


def part_6() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    db = DB.connect(DB_PATH)

    db.execute("INSERT INTO contract_case VALUES (?,?)",
               ("TR-114", "Перевозка груза, Ромашка — ТрансЛогистик"))
    db.execute("INSERT INTO document VALUES (?,?,?,?,?)",
               ("TR-114", "TR-114", "основной", "2026-03-12", "2026-03-12"))
    db.execute("INSERT INTO document VALUES (?,?,?,?,?)",
               ("TR-114-DS1", "TR-114", "допсоглашение", "2026-06-15", "2026-07-01"))

    for doc_id, text, rows, op in (
        ("TR-114", BASE, BASE_FIELDS, "устанавливает"),
        ("TR-114-DS1", DS1, DS1_FIELDS, "изменяет"),
    ):
        ok = sum(DB.put_field(db, doc_id, f, v, raw, q, cl, text, operation=op)
                 for f, v, raw, q, cl in rows)
        print(f"  {doc_id:12s} записано полей: {ok} из {len(rows)}")
    db.commit()

    print("\n--- Как отвечает наивная схема «одна строка = один файл» ---")
    naive = DB.naive_by_file(db, "TR-114")
    for f in ("amount", "payment_days", "demurrage_rate", "date_end"):
        print(f"  {f:16s} {naive.get(f)}")

    print("\n--- Как отвечает схема с историей значений ---")
    eff = DB.effective(db, "TR-114")
    for f in ("amount", "payment_days", "demurrage_rate", "date_end", "jurisdiction"):
        r = eff.get(f)
        if r:
            print(f"  {f:16s} {r['value']:32s} <- {r['doc_id']}, п. {r['clause']}")

    print("\n--- История одного поля, как её увидит юрист ---")
    for r in DB.history(db, "TR-114", "amount"):
        print(f"  {r['doc_date']}  {r['kind']:14s} {r['operation']:12s} {r['value']:>10}  п. {r['clause']}")

    print("\nРазница в 1 700 000 рублей на одном поле. Наивная схема не ошибается")
    print("в вычислениях — она отвечает на другой вопрос, чем ей задали.")


def part_7() -> None:
    """Собственно извлечение: документ -> response_format -> заполненные поля.

    С флагом --nullable схема возвращается к первой редакции, где поле-объект
    допускал null. Это не музейный экспонат: именно та редакция дала null
    во всех 18 полях, и воспроизводимость отказа важнее его отсутствия —
    иначе главный урок этапа придётся читать, а не проходить.
    """
    from rag_lab.extract import prompt_for
    from rag_lab.llm import Llm

    name, text = document()
    fields = {**CORE_FIELDS, **TRANSPORT_FIELDS}
    llm = Llm()
    schema = json_schema(fields)
    prompt = prompt_for(fields, text)

    if "--nullable" in sys.argv:
        for prop in schema["properties"].values():
            prop["type"] = ["object", "null"]
        print("СХЕМА ВОЗВРАЩЕНА К ПЕРВОЙ РЕДАКЦИИ: поле-объект допускает null.\n"
              "Смотрите на долю null — остальные показатели тут бесполезны.\n")

    print(f"документ: {name}, {len(text)} символов, "
          f"{llm.tokens(text)} токенов по токенизатору ЭТОЙ модели")
    print(f"промпт с описанием {len(fields)} полей: {llm.tokens(prompt)} токенов\n")

    reply = llm.fill(prompt, schema)

    print(f"{'поле':18s} {'значение':32s} {'цитата':10s} пункт")
    print("-" * 74)
    filled = passed = 0
    for name in fields:
        got = reply.data.get(name)
        if not got or got.get("value") is None:
            print(f"  {name:16s} {'null':32s}")
            continue
        filled += 1
        value = str(got["value"])
        ok = quote_ok(text, got.get("quote", ""))
        passed += ok
        print(f"  {name:16s} {value[:30]:32s} {'прошла' if ok else 'ОТКЛОНЕНА':10s} "
              f"{got.get('clause') or '—'}")

    print("-" * 74)
    print(f"заполнено: {filled} из {len(fields)}   "
          f"цитата прошла: {passed} из {filled}   "
          f"null: {len(fields) - filled}")
    print(f"задержка: {reply.seconds:.1f} с   "
          f"вывод: {reply.completion_tokens} токенов "
          f"({reply.tokens_per_second:.1f} ток./с)")

    print("\n--- Нормализация: значение из цитаты, подсказка от модели ---")
    print(f"{'поле':18s} {'тип':8s} {'подсказка':12s} {'значение':12s} вердикт / причина")
    numeric = {MONEY, PERCENT, INT}
    for name, f in fields.items():
        got = reply.data.get(name) or {}
        # Порядок тот же, что в db.put_field: сначала цитата, потом разбор.
        # Нормализовать значение при непрошедшей цитате бессмысленно — оно
        # и так не попадёт в базу, а в выводе выглядело бы принятым.
        if got.get("value") is not None and not quote_ok(text, got.get("quote", "")):
            print(f"  {name:16s} {f.kind:8s} {str(got.get('value'))[:10]:12s} "
                  f"{'—':14s} -> pending: цитата не найдена в документе")
            continue
        n = normalize(f, got.get("value"), got.get("quote", ""))
        if not n.ok:
            note = f"-> pending: {n.reason}"
        elif f.kind in numeric and n.value is not None:
            note = f"цифры/пропись: {n.verdict}"
        else:
            note = ""
        print(f"  {name:16s} {f.kind:8s} {str(got.get('value'))[:10]:12s} "
              f"{str(n.value)[:12]:14s} {note}")

    print("\nЗначение берётся из цитаты, прошедшей проверку вхождением, а ответ")
    print("модели служит подсказкой: он говорит, какое из чисел цитаты имеется")
    print("в виду. Иначе сверка цифр с прописью пропадает — модель нормализует")
    print("сама, вопреки промпту, и пропись до кода не доходит.")

    print("\nnull — не провал, а честный ответ. Опасна обратная ошибка:")
    print("заполненное поле с цитатой, которой в документе нет.")


def part_8() -> None:
    """Воспроизводимость: три прогона одного и того же должны совпасть."""
    from rag_lab.extract import prompt_for
    from rag_lab.llm import Llm

    name, text = document()
    fields = {**CORE_FIELDS, **TRANSPORT_FIELDS}
    llm = Llm()
    schema, prompt = json_schema(fields), prompt_for(fields, text)
    print(f"документ: {name}")

    runs = []
    for i in range(3):
        r = llm.fill(prompt, schema)
        runs.append(r)
        print(f"  прогон {i + 1}: {r.seconds:5.1f} с, {r.completion_tokens} токенов вывода")

    same = all(r.data == runs[0].data for r in runs)
    print(f"\nтри прогона {'СОВПАЛИ' if same else 'РАЗОШЛИСЬ'} побайтно")
    if not same:
        for name in fields:
            vals = {json.dumps(r.data.get(name), ensure_ascii=False, sort_keys=True) for r in runs}
            if len(vals) > 1:
                print(f"  расходится {name}:")
                for v in vals:
                    print(f"    {v[:96]}")
    print("\nБез этого свойства ни accuracy, ни сравнение промптов не имеют смысла:")
    print("сдвиг метрики нельзя отличить от разброса модели.")


def part_9() -> None:
    print("Сделано и замерено выше: извлечение, доля null, проверка цитат,")
    print("воспроизводимость, задержка.")
    print("\nНе сделано, и упирается НЕ в модель, а в корпус:")
    print("  - accuracy ПО КАЖДОМУ ПОЛЮ: нужна разметка 10-15 договоров,")
    print("    на пяти документах партии 1 по 18 полям её не посчитать;")
    print("  - доля ложных значений: та же разметка;")
    print("  - связка «контрагент = ИНН»: ИНН партии 1 не проходят контрольную сумму.")


PARTS = {
    1: ("Схема ответа: значение приходит только с цитатой", part_1),
    2: ("Нормализация кодом: даты", part_2),
    3: ("Нормализация кодом: суммы и сверка прописи", part_3),
    4: ("Контрольная сумма ИНН", part_4),
    5: ("Проверка цитаты — барьер против тихой галлюцинации", part_5),
    6: ("История значений: договор + допсоглашение", part_6),
    7: ("Извлечение моделью: первый прогон", part_7),
    8: ("Воспроизводимость: три прогона подряд", part_8),
    9: ("Что осталось до готовности этапа", part_9),
}


def main() -> None:
    # значение --doc — имя файла, а не номер раздела: пропускаем его вместе
    # с самим флагом, иначе «7 --doc AR-203-2026» читается как два раздела
    argv = sys.argv[1:]
    if "--doc" in argv:
        i = argv.index("--doc")
        argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv if not a.startswith("-")]
    if args:
        try:
            wanted = [int(a) for a in args]
        except ValueError:
            sys.exit(f"аргументы — номера разделов 1..{len(PARTS)}, получено: {args}")
        unknown = [n for n in wanted if n not in PARTS]
        if unknown:
            sys.exit(f"нет таких разделов: {unknown}. Доступны 1..{len(PARTS)}")
    else:
        wanted = sorted(PARTS)

    for n in wanted:
        title, fn = PARTS[n]
        print(f"\n{'=' * 74}\n{n}. {title}\n{'=' * 74}")
        fn()


if __name__ == "__main__":
    main()
