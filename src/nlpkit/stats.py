"""Статистики корпуса, законы Ципфа и Хипса, ключевые слова по LLR (ПР 3)."""
import csv
import math
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

from nlpkit.corpus import tokens  # noqa: E402


def _lsq(x, y):
    """Линейная регрессия МНК: y = a*x + b, возвращает a, b, R²."""
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(a), float(b), float(r2)


def corpus_stats(docs, lang_before, seed, out) -> tuple:
    toks_per_doc = [tokens(d["text"]) for d in docs]
    all_toks = [t for ts in toks_per_doc for t in ts]
    cnt = Counter(all_toks)
    n, v = len(all_toks), len(cnt)
    lens = np.array([len(ts) for ts in toks_per_doc])
    sample = all_toks[:]
    random.Random(seed).shuffle(sample)
    s10 = sample[:10000]
    years = Counter(d["year"] for d in docs if d["year"])
    cats = Counter(d["primary_category"] for d in docs)
    lang = Counter(d["lang"] for d in docs)
    total_lang = sum(lang_before.values()) or 1
    st = {
        "n_docs": len(docs), "n_tokens": n, "n_types": v,
        "ttr": round(v / n, 4), "ttr_10k": round(len(set(s10)) / len(s10), 4),
        "hapax": sum(1 for c in cnt.values() if c == 1),
        "len_mean": round(float(lens.mean()), 1), "len_median": float(np.median(lens)),
        "len_p05": float(np.percentile(lens, 5)), "len_p95": float(np.percentile(lens, 95)),
        "len_min": int(lens.min()), "len_max": int(lens.max()),
        "year_min": min(years), "year_max": max(years),
        "date_min": min(d["published"] for d in docs), "date_max": max(d["published"] for d in docs),
        "lang_share_collected": {k: round(c / total_lang, 4) for k, c in lang_before.items()},
        "lang_share_corpus": {k: round(c / len(docs), 4) for k, c in lang.items()},
        "n_keyword_docs": sum(d["query"] == "keyword" for d in docs),
        "n_base_docs": sum(d["query"] == "base" for d in docs),
    }
    st["hapax_share"] = round(st["hapax"] / v, 4)

    with open(out / "corpus_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["показатель", "значение"])
        for k, val in st.items():
            w.writerow([k, val])
    with open(out / "year_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["год", "документов"])
        w.writerows(sorted(years.items()))
    with open(out / "category_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["primary_category", "документов"])
        w.writerows(cats.most_common())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lens, bins=40)
    ax.set_xlabel("токенов в документе")
    ax.set_ylabel("документов")
    ax.set_title("Распределение длин документов")
    fig.tight_layout()
    fig.savefig(out / "doc_lengths.png", dpi=150)
    plt.close(fig)
    return st, cnt, toks_per_doc


def zipf(cnt: Counter, out, core=(10, 1000)) -> dict:
    freqs = np.array(sorted(cnt.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    lx, ly = np.log(ranks), np.log(freqs)
    a_full, b_full, r2_full = _lsq(lx, ly)
    r0, r1 = core[0], min(core[1], len(freqs))
    a_core, b_core, r2_core = _lsq(lx[r0 - 1:r1], ly[r0 - 1:r1])
    res = {"alpha_full": round(-a_full, 3), "r2_full": round(r2_full, 3),
           "alpha_core": round(-a_core, 3), "r2_core": round(r2_core, 3),
           "core_ranks": f"{r0}-{r1}"}

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(ranks, freqs, ".", ms=3, label="данные")
    xs = np.array([1, len(freqs)], dtype=float)
    ax.loglog(xs, np.exp(b_full) * xs ** a_full, "-", lw=1.2,
              label=f"МНК, все ранги: α={-a_full:.2f}, R²={r2_full:.2f}")
    xs2 = np.array([r0, r1], dtype=float)
    ax.loglog(xs2, np.exp(b_core) * xs2 ** a_core, "-", lw=2.2,
              label=f"МНК, ранги {r0}–{r1}: α={-a_core:.2f}, R²={r2_core:.2f}")
    ax.set_xlabel("ранг")
    ax.set_ylabel("частота")
    ax.set_title("Закон Ципфа")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "zipf.png", dpi=150)
    plt.close(fig)

    top = cnt.most_common(100)
    with open(out / "rank_freq_top100.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "word", "freq"])
        for i, (wd, c) in enumerate(top, 1):
            w.writerow([i, wd, c])
    return res


def heaps(toks_per_doc, seed, out, n_min=1000) -> dict:
    order = list(range(len(toks_per_doc)))
    random.Random(seed).shuffle(order)
    seen, n, N, V = set(), 0, [], []
    for i in order:
        for t in toks_per_doc[i]:
            seen.add(t)
        n += len(toks_per_doc[i])
        N.append(n)
        V.append(len(seen))
    N, V = np.array(N, float), np.array(V, float)
    m = N >= n_min
    beta, logk, r2 = _lsq(np.log(N[m]), np.log(V[m]))
    res = {"K": round(math.exp(logk), 2), "beta": round(beta, 3), "r2": round(r2, 4)}

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(N, V, ".", ms=3, label="данные")
    ax.loglog(N[m], np.exp(logk) * N[m] ** beta, "-", lw=2,
              label=f"V = K·N^β: K={res['K']}, β={beta:.2f}, R²={r2:.3f}")
    ax.set_xlabel("токенов N")
    ax.set_ylabel("типов V")
    ax.set_title("Закон Хипса")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "heaps.png", dpi=150)
    plt.close(fig)
    return res


def llr_terms(corp: Counter, ref: Counter, top=50, min_count=5):
    """Log-likelihood ratio (Dunning, G²): слова, избыточные в корпусе относительно эталона."""
    n1, n2 = sum(corp.values()), sum(ref.values())
    rows = []
    for w, a in corp.items():
        if a < min_count or len(w) < 3 or w in ENGLISH_STOP_WORDS:
            continue
        b = ref.get(w, 0)
        if a / n1 <= b / n2:
            continue
        e1 = n1 * (a + b) / (n1 + n2)
        e2 = n2 * (a + b) / (n1 + n2)
        g2 = 2 * (a * math.log(a / e1) + (b * math.log(b / e2) if b else 0.0))
        rows.append((w, a, b, round(g2, 1), round((a / n1) / ((b + 0.5) / n2), 1)))
    rows.sort(key=lambda r: -r[3])
    return rows[:top]


def keywords(cnt: Counter, ref_text: str, out, top=50) -> list:
    ref = Counter(tokens(ref_text))
    rows = llr_terms(cnt, ref, top)
    with open(out / "keywords_llr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "count_corpus", "count_reference", "LLR_G2", "ratio_smoothed"])
        w.writerows(rows)
    print(f"Эталонный корпус: {sum(ref.values())} токенов; ключевых слов: {len(rows)}")
    return rows