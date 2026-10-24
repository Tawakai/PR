"""Оценка precision/recall регулярных выражений на ручной разметке (ПР 1)."""
import csv
import random
import re

from nlpkit.patterns import PATTERNS, find_all

SEP = " ; "
NAMES = list(PATTERNS)

# 40 синтетических строк: (текст, {шаблон: [эталонные совпадения]})
SYNTH = [
    ("Звоните по номеру +7 701 234 56 78 до 18:00.", {"phone_kz": ["+7 701 234 56 78"]}),
    ("Мой e-mail: aidar.k@gmail.com, пишите.", {"email": ["aidar.k@gmail.com"]}),
    ("Стоимость услуги составляет 15 000 тенге за месяц.", {"tenge": ["15 000 тенге"]}),
    ("Курс вырос до 450 ₸ за доллар.", {"tenge": ["450 ₸"]}),
    ("Встреча назначена на 05.11.2026 в офисе.", {"date": ["05.11.2026"]}),
    ("Подробнее на сайте https://www.kazinform.kz/news.", {"url": ["https://www.kazinform.kz/news"]}),
    ("Организация ООН и ЕС подписали соглашение.", {"abbr": ["ООН", "ЕС"]}),
    ("Горячая линия: 8 (727) 258-30-30.", {"phone_kz": ["8 (727) 258-30-30"]}),
    ("Пенсия составила 2,1 млн тг. в год.", {"tenge": ["2,1 млн тг."]}),
    ("Договор от 1 сентября 2023 года вступил в силу.", {"date": ["1 сентября 2023 года"]}),
    ("Версия библиотеки 3.11.4 вышла вчера.", {}),
    ("Сервер доступен по адресу 10.0.0.15.", {}),
    ("Звоните: +7 495 123 45 67.", {}),
    ("Цена 20 долларов, это около 9 000 тенге.", {"tenge": ["9 000 тенге"]}),
    ("XX век стал временем перемен.", {}),
    ("ВВП страны вырос на 3,5 %.", {"abbr": ["ВВП"]}),
    ("Пишите на info@univ.edu.kz или звоните +7 777 000 11 22.",
     {"email": ["info@univ.edu.kz"], "phone_kz": ["+7 777 000 11 22"]}),
    ("Наурыз мерекесі 21 наурыз күні өтеді.", {"date": ["21 наурыз"]}),
    ("Родился 3 мая 1985 г. в Алматы.", {"date": ["3 мая 1985 г."]}),
    ("Оплата: 1 500 KZT на карту.", {"tenge": ["1 500 KZT"]}),
    ("Переходите по ссылке www.example.org/page, там всё есть.", {"url": ["www.example.org/page"]}),
    ("Кроме того, и т.д. и т.п. не нужны.", {"abbr": ["т.д.", "т.п."]}),
    ("НАТО и США обсудили вопрос.", {"abbr": ["НАТО", "США"]}),
    ("Мой номер 87011234567, жду звонка.", {"phone_kz": ["87011234567"]}),
    ("Размер штрафа — 5 МРП, то есть около 18 000 тенге.",
     {"tenge": ["18 000 тенге"], "abbr": ["МРП"]}),
    ("Он работает с 2010 по 2015 год.", {}),
    ("Звонок стоил 100 тг за минуту.", {"tenge": ["100 тг"]}),
    ("Сроки: 01.02.24 – 15.03.24.", {"date": ["01.02.24", "15.03.24"]}),
    ("Адрес: ул. Абая, д. 5, офис 12.", {}),
    ("Свяжитесь: support@company.com. Ответ в течение дня.", {"email": ["support@company.com"]}),
    ("Курс 1 USD = 480 тенге по данным НБ РК.",
     {"tenge": ["480 тенге"], "abbr": ["USD", "НБ", "РК"]}),
    ("Hungary joined the EU in 2004.", {"abbr": ["EU"]}),
    ("Дата 2026-10-24 указана в ISO.", {"date": ["2026-10-24"], "abbr": ["ISO"]}),
    ("Скидка 50 %, акция до 31 декабря.", {"date": ["31 декабря"]}),
    ("Телефон 8 800 100 20 30 бесплатный.", {}),
    ("Стоимость 7 500 тенге, 12 500 тенге и 100 тысяч тенге.",
     {"tenge": ["7 500 тенге", "12 500 тенге", "100 тысяч тенге"]}),
    ("Сайт http://localhost:8080/index работает.", {"url": ["http://localhost:8080/index"]}),
    ("Пишите Aidar_K@Mail.Ru срочно.", {"email": ["Aidar_K@Mail.Ru"]}),
    ("Лекция 12 ноября в 10:00.", {"date": ["12 ноября"]}),
    ("Позвоните +7(701)123-45-67 или +7-701-123-45-67.",
     {"phone_kz": ["+7(701)123-45-67", "+7-701-123-45-67"]}),
]


def _wiki_sentences(cfg: dict) -> list[str]:
    sents = []
    for p in sorted(cfg["paths"]["raw"].glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
            s = s.strip()
            if 40 <= len(s) <= 300 and not s.startswith("="):
                sents.append(s)
    return sents


def _pred(text: str) -> dict:
    return {n: [m.text for m in find_all(n, text)] for n in NAMES}


def make_eval_set(cfg: dict, n_wiki: int = 60) -> None:
    path = cfg["paths"]["annotated"] / "regex_eval.csv"
    if path.exists():
        print(f"{path} уже существует, не перезаписываю (там ваша разметка)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["project"]["seed"])
    sents = _wiki_sentences(cfg)
    with_match = [s for s in sents if any(_pred(s).values())]
    chosen = rng.sample(with_match, n_wiki // 2)
    rest = [s for s in sents if s not in set(chosen)]
    chosen += rng.sample(rest, n_wiki - len(chosen))
    rng.shuffle(chosen)

    rows = []
    for text in chosen:   # предразметка: её нужно ПРОВЕРИТЬ и исправить
        pr = _pred(text)
        rows.append(["wiki", text] + [SEP.join(pr[n]) for n in NAMES])
    for text, gold in SYNTH:   # эталон написан вручную независимо от шаблонов
        rows.append(["synthetic", text] + [SEP.join(gold.get(n, [])) for n in NAMES])

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source", "text"] + NAMES)
        w.writerows(rows)
    print(f"Создан {path}: {len(rows)} строк. Проверьте колонки шаблонов в строках source=wiki.")


def _cell(v: str) -> set:
    return {x.strip() for x in v.split(SEP.strip()) if x.strip()}


def evaluate(cfg: dict) -> None:
    path = cfg["paths"]["annotated"] / "regex_eval.csv"
    out = cfg["paths"]["results"] / "pr1"
    out.mkdir(parents=True, exist_ok=True)
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    table = []
    for n in NAMES:
        tp = fp = fn = 0
        for r in rows:
            gold, pred = _cell(r[n]), {m.text for m in find_all(n, r["text"])}
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        table.append([n, tp, fp, fn, round(p, 3), round(rc, 3), round(f1, 3)])
        print(f"{n:9s} TP={tp:3d} FP={fp:3d} FN={fn:3d}  P={p:.3f}  R={rc:.3f}  F1={f1:.3f}")

    with open(out / "regex_precision_recall.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "TP", "FP", "FN", "precision", "recall", "F1"])
        w.writerows(table)
    print(f"→ {out / 'regex_precision_recall.csv'} (строк в разметке: {len(rows)})")