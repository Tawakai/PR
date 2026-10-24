"""Очистка текста (ПР 1): HTML, пробелы, кавычки, кодировка, латиница/кириллица."""
import html
import re
import unicodedata

_TAG = re.compile(r"<[^>]+>")
_HEADING = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.MULTILINE)   # == История ==
_INVISIBLE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad\u0301]")  # нулевая ширина, мягкий перенос, ударение
_DQ = str.maketrans({c: '"' for c in "«»„“”‟″"})
_SQ = str.maketrans({c: "'" for c in "‘’‚‛′"})
_DASH = str.maketrans({c: "-" for c in "‐‑‒−"})

# латинские буквы, неотличимые от кириллических
_LAT2CYR = str.maketrans("AaBCcEeHKMOoPpTXxyY", "АаВСсЕеНКМОоРрТХхуУ")
_CYR2LAT = {ord(c): l for c, l in zip("АаВСсЕеНКМОоРрТХхуУ", "AaBCcEeHKMOoPpTXxyY")}
_WORD = re.compile(r"\w+")


def _is_cyr(ch: str) -> bool:
    return "CYRILLIC" in unicodedata.name(ch, "")


def _is_lat(ch: str) -> bool:
    return "LATIN" in unicodedata.name(ch, "")


def fix_mixed_scripts(text: str) -> str:
    """В словах со смесью латиницы и кириллицы заменяет «двойников»
    на буквы преобладающего алфавита ('Вeнгpия' → 'Венгрия')."""
    def repl(m: re.Match) -> str:
        w = m.group(0)
        cyr = sum(_is_cyr(c) for c in w)
        lat = sum(_is_lat(c) for c in w)
        if not (cyr and lat):
            return w
        if cyr >= lat:
            fixed = w.translate(_LAT2CYR)
            return fixed if not any(_is_lat(c) for c in fixed) else w
        fixed = w.translate(_CYR2LAT)
        return fixed if not any(_is_cyr(c) for c in fixed) else w
    return _WORD.sub(repl, text)


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = _TAG.sub(" ", text)
    text = unicodedata.normalize("NFC", text)          # единая форма Unicode
    text = _INVISIBLE.sub("", text)
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = _HEADING.sub(r"\1", text)                   # разметка заголовков Википедии
    text = text.translate(_DQ).translate(_SQ).translate(_DASH)
    text = fix_mixed_scripts(text)
    text = re.sub(r"[ ]{2,}", " ", text)               # пробелы
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()