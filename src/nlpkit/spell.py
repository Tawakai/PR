"""Корректор опечаток по модели Норвига (noisy channel) + клавиатура и биграммы (ПР 2)."""
import csv
import math
import random
import re
from collections import Counter

from nlpkit.editdist import damerau_levenshtein, weighted_osa

ALPHA = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
ROWS = ["йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю"]
_POS = {c: (r, i) for r, row in enumerate(ROWS) for i, c in enumerate(row)}
LOG_EPS = math.log(1e-3)   # штраф за одну ошибку в канале


def adjacent(x: str, y: str) -> bool:
    if x not in _POS or y not in _POS or x == y:
        return False
    (r1, c1), (r2, c2) = _POS[x], _POS[y]
    return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1


def kbd_sub(x: str, y: str) -> float:
    return 0.5 if adjacent(x, y) else 1.0


def tokens(text: str) -> list[str]:
    return re.findall(r"[а-яё]+", text.lower())


def edits1(w: str) -> set[str]:
    sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    dele = {a + b[1:] for a, b in sp if b}
    trans = {a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1}
    rep = {a + c + b[1:] for a, b in sp if b for c in ALPHA}
    ins = {a + c + b for a, b in sp for c in ALPHA}
    return dele | trans | rep | ins


class Corrector:
    def __init__(self, train_tokens: list[str]):
        self.uni = Counter(train_tokens)
        self.bi = Counter(zip(train_tokens, train_tokens[1:]))
        self.total = sum(self.uni.values())
        self._cache: dict[str, list[str]] = {}

    def candidates(self, x: str) -> list[str]:
        if x in self._cache:
            return self._cache[x]
        e1 = edits1(x)
        cands = {w for w in e1 if w in self.uni}
        for e in e1:                                   # расстояние 2
            cands |= {w for w in edits1(e) if w in self.uni}
        self._cache[x] = list(cands)
        return self._cache[x]

    def p_uni(self, w: str) -> float:
        return self.uni[w] / self.total

    def p_ctx(self, w: str, prev: str) -> float:
        if not prev or prev not in self.uni:
            return self.p_uni(w)
        return 0.7 * self.bi[(prev, w)] / self.uni[prev] + 0.3 * self.p_uni(w)

    def rank(self, x: str, prev: str = "", mode: str = "base", k: int = 3) -> list[str]:
        scored = []
        for w in self.candidates(x):
            if mode == "base":                         # P(w) · ε^d
                d = damerau_levenshtein(x, w)
                score = math.log(self.p_uni(w)) + d * LOG_EPS
            else:                                      # клавиатура (+ контекст)
                cost = weighted_osa(x, w, sub=kbd_sub, trans=0.75)
                lm = self.p_ctx(w, prev) if mode == "kbd+bigram" else self.p_uni(w)
                score = math.log(lm) + cost * LOG_EPS
            scored.append((score, w))
        scored.sort(reverse=True)
        return [w for _, w in scored[:k]]


def _make_typo(w: str, rng: random.Random) -> str:
    for _ in range(20):
        t = w
        for _ in range(1 if rng.random() < 0.7 else 2):
            i = rng.randrange(len(t))
            op = rng.choice(["sub_adj", "sub", "del", "ins", "trans"])
            if op == "sub_adj":
                nb = [c for c in ALPHA if adjacent(t[i], c)]
                if nb:
                    t = t[:i] + rng.choice(nb) + t[i + 1:]
            elif op == "sub":
                t = t[:i] + rng.choice(ALPHA) + t[i + 1:]
            elif op == "del" and len(t) > 3:
                t = t[:i] + t[i + 1:]
            elif op == "ins":
                t = t[:i] + rng.choice(ALPHA) + t[i:]
            elif op == "trans" and i + 1 < len(t):
                t = t[:i] + t[i + 1] + t[i] + t[i + 2:]
        if t != w:
            return t
    return w + "ь"


def evaluate(cfg: dict) -> None:
    out = cfg["paths"]["results"] / "pr2"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["project"]["seed"])
    n_typos = cfg["pr2"]["n_typos"]

    toks = []
    for p in sorted(cfg["paths"]["clean"].glob("*.txt")):
        toks += tokens(p.read_text(encoding="utf-8"))
    cut = int(len(toks) * 0.9)
    train, test = toks[:cut], toks[cut:]
    cor = Corrector(train)

    real = cfg["paths"]["annotated"] / "typos_real.csv"
    if real.exists():
        with open(real, encoding="utf-8-sig") as f:
            cases = [(r["typo"].lower(), r["correct"].lower(), r.get("prev", "").lower())
                     for r in csv.DictReader(f)]
        src = f"реальные опечатки из {real.name}"
    else:
        idx = [i for i in range(1, len(test)) if len(test[i]) >= 4 and test[i] in cor.uni]
        rng.shuffle(idx)
        cases = []
        for i in idx:
            typo = _make_typo(test[i], rng)
            if typo not in cor.uni:
                cases.append((typo, test[i], test[i - 1]))
            if len(cases) == n_typos:
                break
        src = "синтетические опечатки (генерация по клавиатуре)"
    print(f"Тест корректора: {len(cases)} опечаток, {src}")

    rows, errors = [], []
    for mode in ["base", "kbd", "kbd+bigram"]:
        a1 = a3 = 0
        for typo, gold, prev in cases:
            top = cor.rank(typo, prev, mode, 3)
            a1 += bool(top) and top[0] == gold
            a3 += gold in top
            if mode == "kbd+bigram" and (not top or top[0] != gold):
                errors.append([typo, gold, prev, " | ".join(top)])
        rows.append([mode, round(a1 / len(cases), 3), round(a3 / len(cases), 3)])
        print(f"{mode:11s} acc@1={a1 / len(cases):.3f}  acc@3={a3 / len(cases):.3f}")

    with open(out / "spell_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["вариант", "accuracy@1", "accuracy@3"])
        w.writerows(rows)
    with open(out / "spell_errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["опечатка", "эталон", "предыдущее слово", "топ-3 системы"])
        w.writerows(errors[:30])