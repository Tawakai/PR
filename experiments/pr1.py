import csv

from nlpkit.cleaning import clean_text
from nlpkit.config import results_dir
from nlpkit.patterns import PATTERNS, find_all
from nlpkit.sentences import compare


def main(cfg: dict) -> None:
    out = results_dir(cfg, "pr1")
    raw, clean = cfg["paths"]["raw"], cfg["paths"]["clean"]
    clean.mkdir(parents=True, exist_ok=True)

    # 1. очистка всего корпуса
    n_words = 0
    for p in sorted(raw.glob("*.txt")):
        t = clean_text(p.read_text(encoding="utf-8"))
        (clean / p.name).write_text(t, encoding="utf-8")
        n_words += len(t.split())
    print(f"Очищено файлов: {len(list(clean.glob('*.txt')))}, слов: {n_words}")

    # 2. сколько совпадений шаблонов на корпусе
    with open(out / "pattern_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "matches"])
        for name in PATTERNS:
            c = sum(len(find_all(name, p.read_text(encoding="utf-8")))
                    for p in clean.glob("*.txt"))
            w.writerow([name, c])
            print(f"{name:9s} {c}")

    # 3. сравнение сегментации на ~200 предложениях
    compare(cfg)