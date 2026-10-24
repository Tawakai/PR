"""Стемминг и лемматизация: сравнение на формах с эталонными леммами из UD (ПР 2)."""
import csv
import random
from pathlib import Path

import requests

UD_URLS = {
    "ru": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-GSD/master/ru_gsd-ud-test.conllu",
    "kk": "https://raw.githubusercontent.com/UniversalDependencies/UD_Kazakh-KTB/master/kk_ktb-ud-test.conllu",
}

# Казахский суффиксный стеммер: падежи, притяжательные, множественное; длинные суффиксы первыми
KK_SUFFIXES = sorted("""
лар лер дар дер тар тер ның нің дың дің тың тің
ға ге қа ке на не да де та те дан ден тан тен нан нен
ны ні ды ді ты ті мен бен пен
ым им ың ің мыз міз ыңыз іңіз сы сі ы і м ң
ған ген қан кен атын етін йтын йтін
ды ді ты ті ып іп
""".split(), key=len, reverse=True)


def kk_stem(w: str, rounds: int = 4) -> str:
    w = w.lower()
    for _ in range(rounds):
        for s in KK_SUFFIXES:
            if w.endswith(s) and len(w) - len(s) >= 2:
                w = w[: -len(s)]
                break
        else:
            break
    return w


def read_conllu(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) >= 4 and c[0].isdigit() and c[3] not in ("PUNCT", "PROPN", "NUM", "SYM", "X"):
            pairs.append((c[1].lower(), c[2].lower()))
    return pairs


def gold_sample(cfg: dict, lang: str) -> list[tuple[str, str]]:
    ext = cfg["paths"]["external"]
    ext.mkdir(parents=True, exist_ok=True)
    path = ext / UD_URLS[lang].split("/")[-1]
    if not path.exists():
        r = requests.get(UD_URLS[lang], timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    uniq = sorted({(f, l) for f, l in read_conllu(path) if f.isalpha() and l.isalpha()})
    rng = random.Random(cfg["project"]["seed"])
    rng.shuffle(uniq)
    return uniq[: cfg["pr2"]["n_lemma_words"]]


def lcp(a: str, b: str) -> str:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[:i]


def eval_stemmer(stem, pairs) -> dict:
    """Эталонная основа = общий префикс формы и леммы (приближение)."""
    ok = over = under = other = 0
    for form, lemma in pairs:
        s, g = stem(form), lcp(form, lemma)
        if s == g:
            ok += 1
        elif g.startswith(s):
            over += 1          # отрезали лишнего
        elif s.startswith(g):
            under += 1         # недорезали
        else:
            other += 1
    n = len(pairs)
    return {"accuracy": ok / n, "over": over / n, "under": under / n, "other": other / n}


def lemmatizers(lang: str) -> dict:
    out = {}
    if lang == "ru":
        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()
        out["pymorphy3"] = lambda ws: [morph.parse(w)[0].normal_form for w in ws]
        try:
            import spacy
            nlp = spacy.load("ru_core_news_md", disable=["ner", "parser"])
            out["spacy"] = lambda ws: [d[0].lemma_.lower() if len(d) else w
                                       for d, w in zip(nlp.pipe(ws), ws)]
        except Exception as e:
            print("spaCy пропущен:", e)
    try:
        import stanza
        try:
            nlp_s = stanza.Pipeline(lang, processors="tokenize,pos,lemma",
                                    tokenize_pretokenized=True, verbose=False)
        except Exception:
            stanza.download(lang, verbose=False)
            nlp_s = stanza.Pipeline(lang, processors="tokenize,pos,lemma",
                                    tokenize_pretokenized=True, verbose=False)
        out["stanza"] = lambda ws: [(s.words[0].lemma or w).lower()
                                    for s, w in zip(nlp_s([[w] for w in ws]).sentences, ws)]
    except Exception as e:
        print("Stanza пропущен:", e)
    return out


def evaluate(cfg: dict) -> None:
    from nltk.stem.snowball import SnowballStemmer
    out = cfg["paths"]["results"] / "pr2"
    out.mkdir(parents=True, exist_ok=True)
    stem_rows, lem_rows, err_rows = [], [], []
    snow = SnowballStemmer("russian").stem

    for lang in ("ru", "kk"):
        try:
            pairs = gold_sample(cfg, lang)
        except Exception as e:
            print(f"[{lang}] не удалось получить эталон UD: {e}")
            continue
        print(f"[{lang}] форм с эталонной леммой: {len(pairs)}")

        stemmers = ({"kk_suffix": kk_stem} if lang == "kk"
                    else {"snowball": snow, "kk_suffix(для сравнения)": kk_stem})
        for name, st in stemmers.items():
            r = eval_stemmer(st, pairs)
            stem_rows.append([lang, name] + [round(v, 3) for v in r.values()])
            print(f"  стеммер {name:24s} acc={r['accuracy']:.3f} over={r['over']:.3f} under={r['under']:.3f}")

        words = [f for f, _ in pairs]
        for name, fn in lemmatizers(lang).items():
            pred = fn(words)
            acc = sum(p == l for p, (_, l) in zip(pred, pairs)) / len(pairs)
            lem_rows.append([lang, name, round(acc, 3)])
            print(f"  лемматизатор {name:10s} accuracy={acc:.3f}")
            for w, p, (_, l) in zip(words, pred, pairs):
                if p != l:
                    err_rows.append([lang, name, w, l, p])

    with open(out / "stemming.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["язык", "стеммер", "accuracy", "over-stemming", "under-stemming", "прочее"])
        w.writerows(stem_rows)
    with open(out / "lemmatization.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["язык", "лемматизатор", "accuracy"])
        w.writerows(lem_rows)
    with open(out / "lemma_errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["язык", "система", "форма", "эталон", "предсказание"])
        w.writerows(err_rows[:60])