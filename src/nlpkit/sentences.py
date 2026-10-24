"""Сегментация на предложения тремя способами и сравнение (ПР 1)."""
import csv
import random
import re
from collections import Counter

_SPLIT = re.compile(r"(?<=[.!?…])[\"»)\]]*\s+(?=[\"«(\[]?[А-ЯЁӘҒҚҢӨҰҮҺІA-Z0-9])")
ABBR_BEFORE = {"г", "гг", "ул", "д", "см", "др", "т", "п", "им", "руб", "тыс", "млн",
               "млрд", "кв", "км", "чел", "напр", "проф", "акад", "ст", "стр", "рис", "в", "вв"}


def split_regex(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text) if s.strip()]


def split_nltk(text: str) -> list[str]:
    import nltk
    return [s.strip() for s in nltk.sent_tokenize(text, language="russian") if s.strip()]


_NLP = None


def split_spacy(text: str) -> list[str]:
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("ru_core_news_md", disable=["ner", "lemmatizer", "attribute_ruler"])
        _NLP.max_length = 2_000_000
    return [s.text.strip() for s in _NLP(text).sents if s.text.strip()]


METHODS = {"regex": split_regex, "nltk": split_nltk, "spacy": split_spacy}


def boundaries(text: str, sents: list[str]) -> set[int]:
    """Позиции концов предложений (кроме последнего) в исходной строке."""
    pos, out = 0, set()
    for s in sents[:-1]:
        i = text.find(s, pos)
        if i < 0:
            continue
        pos = i + len(s)
        out.add(pos)
    return out


def error_type(text: str, b: int) -> str:
    """Тип ошибки по слову перед границей."""
    left = text[:b].rstrip()
    word = left.split()[-1] if left.split() else ""
    if word.endswith("…") or word.endswith("..."):
        return "многоточие"
    if re.fullmatch(r"(?:[А-ЯЁA-Z]\.){1,3}", word):
        return "инициалы"
    if word.endswith(".") and word.rstrip(".").lower() in ABBR_BEFORE:
        return "сокращение"
    if re.search(r"\d\.$", word):
        return "число/дата с точкой"
    return "другое"


def compare(cfg: dict, methods: dict = METHODS, target: int = 200) -> None:
    """Берём абзацы корпуса, пока в них не наберётся ~target предложений.
    Эталон — мажоритарное голосование методов; расхождения классифицируем."""
    from nlpkit.cleaning import clean_text

    out = cfg["paths"]["results"] / "pr1"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["project"]["seed"])

    paras = []
    for p in sorted(cfg["paths"]["raw"].glob("*.txt")):
        for para in clean_text(p.read_text(encoding="utf-8")).split("\n"):
            if len(para) > 80:
                paras.append(para)
    rng.shuffle(paras)

    names = list(methods)
    errors = {n: Counter() for n in names}
    rows, total_sents = [], 0
    for para in paras:
        if total_sents >= target:
            break
        sents = {n: methods[n](para) for n in names}
        b = {n: boundaries(para, sents[n]) for n in names}
        all_b = set().union(*b.values())
        gold = {x for x in all_b if sum(x in b[n] for n in names) > len(names) / 2}
        total_sents += len(gold) + 1
        for x in sorted(all_b):
            wrong = [n for n in names if (x in b[n]) != (x in gold)]
            if not wrong:
                continue
            t = error_type(para, x)
            for n in wrong:
                errors[n][t] += 1
            rows.append([para[max(0, x - 60):x].replace("\n", " "),
                         para[x:x + 60].replace("\n", " "),
                         "есть в эталоне" if x in gold else "лишняя граница",
                         t, ",".join(wrong)])

    with open(out / "sentence_errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["до границы", "после границы", "что не так", "тип", "метод(ы) ошиблись"])
        w.writerows(rows)

    types = sorted({t for c in errors.values() for t in c})
    with open(out / "sentence_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["метод", "всего ошибок"] + types)
        for n in names:
            w.writerow([n, sum(errors[n].values())] + [errors[n][t] for t in types])
            print(f"{n:6s} ошибок: {sum(errors[n].values()):3d}  {dict(errors[n])}")
    print(f"Предложений в выборке (по эталону): {total_sents}; расхождения → {out / 'sentence_errors.csv'}")