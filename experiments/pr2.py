import csv
import random
import time

from nlpkit import morph, spell, tokenization
from nlpkit.config import results_dir
from nlpkit.editdist import damerau_levenshtein, levenshtein
from nlpkit.spell import ALPHA


def edit_distance_part(cfg: dict, out) -> None:
    from rapidfuzz.distance import OSA, Levenshtein
    rng = random.Random(cfg["project"]["seed"])
    rand = lambda: "".join(rng.choice(ALPHA) for _ in range(rng.randint(3, 12)))
    pairs = [(rand(), rand()) for _ in range(1000)]

    rows = []
    for name, own, ref in [("Левенштейн", levenshtein, Levenshtein.distance),
                           ("Дамерау–Левенштейн (OSA)", damerau_levenshtein, OSA.distance)]:
        t = time.perf_counter()
        mine = [own(a, b) for a, b in pairs]
        t_own = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        theirs = [ref(a, b) for a, b in pairs]
        t_ref = (time.perf_counter() - t) * 1000
        agree = sum(m == r for m, r in zip(mine, theirs))
        rows.append([name, round(t_own, 1), round(t_ref, 2), f"{agree}/1000"])
        print(f"{name:26s} свой={t_own:7.1f} мс  rapidfuzz={t_ref:6.2f} мс  совпало {agree}/1000")

    with open(out / "editdist.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["алгоритм", "время, мс (1000 пар), свой", "время, мс, rapidfuzz", "совпадений с rapidfuzz"])
        w.writerows(rows)

    with open(out / "ops_examples.txt", "w", encoding="utf-8") as f:
        for a, b in [("котик", "кот"), ("корова", "короваа"), ("стол", "стлоб"), ("венгрия", "вегнрия")]:
            f.write(f"{a} → {b}\n  Левенштейн: {levenshtein(a, b, True)}\n"
                    f"  Дамерау–Л.: {damerau_levenshtein(a, b, True)}\n")


def main(cfg: dict) -> None:
    out = results_dir(cfg, "pr2")
    print("== Расстояния редактирования ==")
    edit_distance_part(cfg, out)
    print("== Корректор опечаток ==")
    spell.evaluate(cfg)
    print("== Токенизаторы ==")
    tokenization.compare(cfg)
    print("== Стемминг и лемматизация ==")
    morph.evaluate(cfg)
