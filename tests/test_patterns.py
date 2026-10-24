"""5 положительных и 5 отрицательных примеров на каждый шаблон (ПР 1)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nlpkit.patterns import find_all  # noqa: E402

# (текст, ожидаемое совпадение)
POSITIVE = {
    "date": [
        ("Родился 12.03.2024 в Будапеште", "12.03.2024"),
        ("Договор подписан 2024-03-12.", "2024-03-12"),
        ("Это случилось 20 августа 1000 года", "20 августа 1000 года"),
        ("Праздник отмечают 1 мая", "1 мая"),
        ("Закон принят 15 наурыз 2023", "15 наурыз 2023"),
    ],
    "email": [
        ("Пишите на info@example.com", "info@example.com"),
        ("a.b-c+d@mail.ru, ответим", "a.b-c+d@mail.ru"),
        ("Контакт: user_01@univ.edu.kz.", "user_01@univ.edu.kz"),
        ("support@sub.domain.org", "support@sub.domain.org"),
        ("Пишите (ivan@yandex.ru) срочно", "ivan@yandex.ru"),
    ],
    "url": [
        ("См. https://ru.wikipedia.org/wiki/Венгрия.", "https://ru.wikipedia.org/wiki/Венгрия"),
        ("Сайт www.example.kz работает", "www.example.kz"),
        ("Ссылка http://a.b/c?d=1&e=2, далее", "http://a.b/c?d=1&e=2"),
        ("Документ: https://gov.kz/docs/1.pdf", "https://gov.kz/docs/1.pdf"),
        ("Зайдите на HTTP://SITE.ORG/X!", "HTTP://SITE.ORG/X"),
    ],
    "phone_kz": [
        ("Тел. +7 701 123 45 67", "+7 701 123 45 67"),
        ("Звоните 8 (777) 123-45-67", "8 (777) 123-45-67"),
        ("Офис: +7 (727) 250-00-00", "+7 (727) 250-00-00"),
        ("Номер 87011234567", "87011234567"),
        ("WhatsApp: +77751234567.", "+77751234567"),
    ],
    "tenge": [
        ("Цена 5 000 тенге", "5 000 тенге"),
        ("Бюджет 2,5 млн тг.", "2,5 млн тг."),
        ("Взнос 12 500 ₸", "12 500 ₸"),
        ("Штраф ₸5000", "₸5000"),
        ("Стоит 100 тыс. тенге", "100 тыс. тенге"),
    ],
    "abbr": [
        ("Венгрия входит в ЕС", "ЕС"),
        ("Договор с НАТО", "НАТО"),
        ("Решение ООН", "ООН"),
        ("ВВП по данным GDP", "ВВП"),
        ("и т.д. и далее", "т.д."),
    ],
}

NEGATIVE = {
    "date": ["версия 1.2.3", "IP 192.168.1.1", "дата 32.13.2024",
             "в 1000 году", "наурыз — праздник"],
    "email": ["пишите user@", "упомянул @name в чате", "адрес x@y без домена",
              "почта mail.ru", "формула a@b"],
    "url": ["сайт example без схемы", "ru.wikipedia.org", "http: это протокол",
            "слово www отдельно", "файл 1.pdf"],
    "phone_kz": ["+7 912 345 67 89", "8 800 555 35 35", "1+7 701 123 45 67",
                 "счёт 7011234567", "код 701 123"],
    "tenge": ["5 000 рублей", "100 тенгенев", "1999 год",
              "цена 100 долларов", "просто 5000"],
    "abbr": ["XX век", "Венгрия", "ВЕНГРИЯ", "привет, мир", "Будапешт"],
}


@pytest.mark.parametrize("name", POSITIVE)
def test_positive(name):
    for text, expected in POSITIVE[name]:
        got = [m.text for m in find_all(name, text)]
        assert expected in got, f"{name}: {text!r} → {got}"


@pytest.mark.parametrize("name", NEGATIVE)
def test_negative(name):
    for text in NEGATIVE[name]:
        got = [m.text for m in find_all(name, text)]
        assert got == [], f"{name}: {text!r} → {got}"