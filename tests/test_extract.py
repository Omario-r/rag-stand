"""Тесты чистых функций извлечения. Этап 7.

ЧТО ЗДЕСЬ ЕСТЬ И ЧЕГО НЕТ. Здесь только то, что считается детерминированно
и без сети: разбор чисел, дат, ИНН, проверка цитаты, нормализация по типу
поля. Ни модели, ни Qdrant, ни эмбеддера — их проверяет `evaluate.py`
на золотом наборе, и это другой инструмент: метрика, а не тест.

Обратное тоже верно, и ради этого файл и заведён: **метрика не поймает
разборщик, который вместо отказа возвращает ноль**. Такое уже случалось
дважды, и оба раза находка оседала в комментарии — а комментарий не падает,
когда правило нарушено:

  * словарь числительных из одних именительных форм разбирал
    «1 500 (одной тысячи пятисот)» как 1000 и слал в очередь исправные
    договоры;
  * денежный разборщик на процентном поле «0,1» возвращал 0.0 — не отказ,
    а правдоподобное «неустойки нет».

Оба случая ниже — тестами, а не комментариями.

Запуск:
    uv run pytest -q
"""

import sys
from pathlib import Path

import pytest

STAND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAND / "src"))

from rag_lab.extract import (  # noqa: E402
    BOOL,
    DATE,
    ENUM,
    INT,
    MONEY,
    PERCENT,
    TEXT,
    РАСХОДИТСЯ,
    СХОДИТСЯ,
    НЕ_СВЕРЕНО,
    Field,
    dates_in,
    inn_valid,
    normalize,
    numbers_in,
    parse_amount_ru,
    parse_date_ru,
    pick_number,
    quote_ok,
    words_to_number,
)

# Цитаты — дословно из corpus/TR-114-2026.md, чтобы тест ловил расхождение
# с настоящим корпусом, а не с придуманным.
Q_AMOUNT = ("Общая стоимость услуг по настоящему Договору не может превышать "
            "4 500 000 (четыре миллиона пятьсот тысяч) рублей 00 копеек, включая НДС 20%.")
Q_PENALTY = ("За просрочку доставки груза Перевозчик уплачивает Заказчику неустойку "
             "в размере 0,1 (одной десятой) процента от стоимости перевозки за каждый "
             "день просрочки, но не более 10 (десяти) процентов от стоимости "
             "соответствующей перевозки.")
Q_DAYS = "Оплата производится Заказчиком в течение 10 (десяти) банковских дней"
Q_TERM = ("Настоящий Договор вступает в силу с момента подписания и действует "
          "до 31 декабря 2026 года.")


# --- Числительные во всех падежах ------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("четыре миллиона пятьсот тысяч", 4_500_000),
    ("одной тысячи пятисот", 1_500),          # родительный: ломал прежний словарь
    ("одной тысячи восьмисот", 1_800),
    ("двадцати", 20),
    ("десяти", 10),
    ("восемь", 8),
    ("шесть миллионов двести тысяч", 6_200_000),
    ("банковских дней", None),                # слов-числительных нет вовсе
])
def test_words_to_number(text, expected):
    assert words_to_number(text) == expected


def test_stem_matching_would_break_this():
    """«сто» не должно находиться внутри «стоимость» — сопоставление по слову."""
    assert words_to_number("стоимость перевозки") is None


# --- Разбор чисел ----------------------------------------------------------

def test_decimal_is_not_truncated_to_zero():
    """Регресс 10.08.2026: «0,1» разбиралось как 0.0 — ложный ноль."""
    assert numbers_in("0,1 процента")[0].digits == 0.1
    assert parse_amount_ru("0.1").digits == 0.1


def test_all_numbers_of_a_quote_are_found():
    assert [a.digits for a in numbers_in(Q_AMOUNT)] == [4_500_000, 0, 20]
    assert [a.digits for a in numbers_in(Q_PENALTY)] == [0.1, 10]


def test_words_are_attached_to_their_own_number():
    money, kopeck, vat = numbers_in(Q_AMOUNT)
    assert (money.digits, money.words, money.verdict) == (4_500_000, 4_500_000, СХОДИТСЯ)
    assert kopeck.words is None and vat.words is None


def test_fraction_words_are_not_compared():
    """«0,1 (одной десятой)» словарём целых читается как 1.

    Сверять их значило бы объявить расхождение там, где его нет, — а ложная
    тревога в такой проверке хуже её отсутствия.
    """
    assert numbers_in(Q_PENALTY)[0].verdict == НЕ_СВЕРЕНО


def test_typo_in_words_is_reported():
    """Расхождение цифр и прописи — сигнал человеку, а не повод «исправить»."""
    a = parse_amount_ru("4 500 000 (четыре миллиона пятьдесят тысяч) рублей")
    assert a.verdict == РАСХОДИТСЯ
    assert (a.digits, a.words) == (4_500_000, 4_050_000)


def test_no_words_is_not_a_mismatch():
    assert parse_amount_ru("1500 рублей").verdict == НЕ_СВЕРЕНО


# --- Выбор числа по подсказке ----------------------------------------------

def test_hint_picks_the_right_number_of_several():
    assert pick_number(Q_PENALTY, 0.1).digits == 0.1
    assert pick_number(Q_PENALTY, 10).digits == 10
    assert pick_number(Q_AMOUNT, 4_500_000).digits == 4_500_000


def test_hint_that_matches_nothing_returns_none():
    """Лучше отказ, чем «первое число из цитаты»: это второй тихий источник."""
    assert pick_number(Q_AMOUNT, 999) is None


# --- Даты ------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ('«12» марта 2026 г.', "2026-03-12"),
    ("31 декабря 2026 года", "2026-12-31"),
    ("15.06.2026", "2026-06-15"),
    ("до полного исполнения обязательств", None),
])
def test_parse_date_ru(text, expected):
    assert parse_date_ru(text) == expected


def test_dates_in_finds_all():
    assert dates_in(Q_TERM) == ["2026-12-31"]
    assert dates_in("с 12 марта 2026 года по 31 декабря 2026 года") == [
        "2026-03-12", "2026-12-31"]


# --- Проверка цитаты -------------------------------------------------------

DOC = "Пункт 4.2. " + Q_AMOUNT


def test_quote_survives_typography():
    assert quote_ok(DOC, "не может превышать 4 500 000")
    assert quote_ok(DOC, "НЕ  МОЖЕТ   превышать 4 500 000")     # регистр и пробелы


def test_invented_quote_is_rejected():
    assert not quote_ok(DOC, "не может превышать 5 800 000")
    assert not quote_ok(DOC, "Договор ограничен суммой четыре с половиной миллиона")
    assert not quote_ok(DOC, "")


# --- Нормализация по типу поля ---------------------------------------------

def test_money_comes_from_quote_not_from_model():
    """Модель вернула голое число — пропись берётся из цитаты и сходится."""
    n = normalize(Field(MONEY, "сумма"), 4500000, Q_AMOUNT)
    assert (n.ok, n.value, n.verdict) == (True, 4_500_000, СХОДИТСЯ)


def test_percent_field_does_not_become_zero():
    """Тот самый регресс: денежный разбор давал здесь 0.0."""
    n = normalize(Field(PERCENT, "неустойка"), 0.1, Q_PENALTY)
    assert (n.ok, n.value) == (True, 0.1)


def test_value_absent_from_quote_goes_to_pending():
    n = normalize(Field(MONEY, "сумма"), 5_800_000, Q_AMOUNT)
    assert n.ok is False and "нет в цитате" in n.reason


def test_words_mismatch_goes_to_pending():
    n = normalize(Field(MONEY, "сумма"), 4_500_000,
                  "сумма 4 500 000 (четыре миллиона пятьдесят тысяч) рублей")
    assert n.ok is False and "расходятся" in n.reason


def test_int_field_rejects_fraction():
    n = normalize(Field(INT, "дней"), 10, Q_DAYS)
    assert (n.ok, n.value) == (True, 10) and isinstance(n.value, int)
    assert normalize(Field(INT, "дней"), 0.1, Q_PENALTY).ok is False


def test_date_must_be_present_in_quote():
    assert normalize(Field(DATE, "дата"), "31 декабря 2026 года", Q_TERM).value == "2026-12-31"
    assert normalize(Field(DATE, "дата"), "01.01.2030", Q_TERM).ok is False


def test_bool_must_be_bool():
    assert normalize(Field(BOOL, "да/нет"), True, "любая цитата").value is True
    assert normalize(Field(BOOL, "да/нет"), "возможно", "любая цитата").ok is False


def test_enum_rejects_invented_category():
    f = Field(ENUM, "перевозка | экспедирование | аренда_ТС | хранение | иное")
    assert normalize(f, "перевозка", "цитата").ok is True
    assert normalize(f, "грузоперевозка", "цитата").ok is False


def test_null_is_legal():
    """Пустое поле честнее выдуманного — оно не ошибка и не уходит в очередь."""
    n = normalize(Field(MONEY, "сумма"), None, Q_AMOUNT)
    assert n.ok is True and n.value is None


def test_text_field_is_not_parsed():
    n = normalize(Field(TEXT, "суд"), "  Арбитражный суд города Москвы ", "цитата")
    assert n.value == "Арбитражный суд города Москвы"


# --- ИНН -------------------------------------------------------------------

@pytest.mark.parametrize("inn, valid", [
    ("7707083893", True),      # Сбербанк
    ("7736050003", True),      # Газпром
    ("500100732259", True),    # 12 знаков, ИП
    ("7707083894", False),     # испорчен нарочно
    ("7701234567", False),     # выдуман при генерации корпуса
    ("770708389", False),      # длина не 10 и не 12
    ("", False),
])
def test_inn_valid(inn, valid):
    assert inn_valid(inn) is valid
