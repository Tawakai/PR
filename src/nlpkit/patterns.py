"""Регулярные выражения для извлечения сущностей из текста (ПР 1)."""
import re
from typing import NamedTuple

MONTHS_RU = ("января|февраля|марта|апреля|мая|июня|июля|августа|"
             "сентября|октября|ноября|декабря")
MONTHS_KK = ("қаңтар|ақпан|наурыз|сәуір|мамыр|маусым|шілде|тамыз|"
             "қыркүйек|қазан|қараша|желтоқсан")
MONTHS = f"{MONTHS_RU}|{MONTHS_KK}"

_DAY = r"(?:0?[1-9]|[12]\d|3[01])"
_MON = r"(?:0?[1-9]|1[0-2])"

DATE = re.compile(
    rf"(?<![\d.]){_DAY}\.{_MON}\.(?:\d{{4}}|\d{{2}})(?!\d)"        # 12.03.2024
    rf"|(?<!\d)\d{{4}}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)"  # 2024-03-12
    rf"|(?<!\d){_DAY}\s+(?:{MONTHS})(?:\s+\d{{4}}(?:\s*(?:года|г\.))?)?",  # 12 марта 2024 года
    re.IGNORECASE,
)

EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
)

URL = re.compile(r"(?:https?://|www\.)[^\s<>\"'«»]+", re.IGNORECASE)

PHONE_KZ = re.compile(
    r"(?<!\w)(?:\+7|8)[\s-]?\(?7\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"
)

_NUM = r"\d+(?:[\s\u00a0]\d{3})*(?:[.,]\d+)?"
_MULT = r"(?:\s*(?:тыс\.?|млн|млрд|трлн)(?!\w))?"
_CUR = r"(?:тенге|тг\.?|₸|KZT)(?!\w)"
TENGE = re.compile(
    rf"(?<![\d.,]){_NUM}{_MULT}\s*{_CUR}"       # 5 000 тенге, 2,5 млн тг.
    rf"|₸\s*{_NUM}(?![\d])",                     # ₸5000
    re.IGNORECASE,
)

ABBR = re.compile(
    r"(?<!\w)(?:[А-ЯЁ]{2,6}|[A-Z]{2,6})(?!\w)"   # ООН, НАТО, GDP
    r"|(?<!\w)(?:[а-яё]\.){2,}",                 # т.д., т.е., т.п.
)
_ROMAN = re.compile(r"^[IVXLCDM]+$")   # XX век — не аббревиатура

PATTERNS = {
    "date": DATE,
    "email": EMAIL,
    "url": URL,
    "phone_kz": PHONE_KZ,
    "tenge": TENGE,
    "abbr": ABBR,
}


class Match(NamedTuple):
    start: int
    end: int
    text: str


def find_all(name: str, text: str) -> list[Match]:
    """Все совпадения шаблона name в тексте (с постобработкой)."""
    out = []
    for m in PATTERNS[name].finditer(text):
        s, e = m.start(), m.end()
        if name == "url":
            # хвостовая пунктуация — не часть ссылки
            while e > s and text[e - 1] in ".,;:!?)":
                e -= 1
        if name == "abbr" and _ROMAN.match(text[s:e]):
            continue
        out.append(Match(s, e, text[s:e]))
    return out