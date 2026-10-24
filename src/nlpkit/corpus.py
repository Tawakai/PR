"""Построение корпуса: очистка, фильтрация мусора, SimHash-дедупликация, язык (ПР 3)."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict

import numpy as np

from nlpkit.cleaning import clean_text          # очистка из ПР 1
from nlpkit.config import ROOT, results_dir
from nlpkit.opts import get_opts

RAW = ROOT / "data/raw_arxiv/arxiv_raw.jsonl"
WORD = re.compile(r"[a-z]+(?:[-'][a-z]+)*")
_MATH = re.compile(r"\$[^$]*\$")
_WITHDRAWN = re.compile(
    r"(?:paper|article|submission|manuscript)\s+(?:has been|is|was)\s+withdrawn|withdrawn by the authors?",
    re.I)


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def clean_abstract(t: str) -> str:
    return clean_text(_MATH.sub(" ", t))          # формулы $...$ вырезаем


# ---------- SimHash ----------
def simhash(text: str, bits: int = 64) -> int:
    w = tokens(text)
    sh = [" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))]
    hs = np.array([int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "big") for s in sh],
                  dtype=np.uint64)
    m = (hs[:, None] >> np.arange(bits, dtype=np.uint64)) & np.uint64(1)
    v = m.sum(axis=0).astype(np.int64) * 2 - len(hs)
    return sum(1 << i for i in range(bits) if v[i] > 0)


def dedup_simhash(docs: list[dict], k: int = 3) -> tuple[list[dict], list[tuple]]:
    """Оставляет первый из группы почти-дубликатов (расстояние Хэмминга ≤ k).
    Хэш делится на k+1 блоков: при расстоянии ≤ k хотя бы один блок совпадает
    (принцип Дирихле), поэтому сравниваем только документы с общим блоком."""
    assert 1 <= k <= 3
    nb, size = k + 1, 64 // (k + 1)

    def blocks(h):
        return [(i, (h >> (i * size)) & ((1 << size) - 1)) if i < nb - 1 else (i, h >> (i * size))
                for i in range(nb)]

    buckets, hashes, kept, dropped = defaultdict(list), [], [], []
    for d in docs:
        h = simhash(d["text"])
        bl = blocks(h)
        cand = {j for b in bl for j in buckets[b]}
        dup = next((j for j in cand if bin(h ^ hashes[j]).count("1") <= k), None)
        if dup is not None:
            dropped.append((d["id"], kept[dup]["id"]))
            continue
        for b in bl:
            buckets[b].append(len(kept))
        hashes.append(h)
        kept.append(d)
    return kept, dropped


# ---------- язык ----------
def detect_lang(text: str) -> str:
    from langdetect import DetectorFactory, detect, LangDetectException
    DetectorFactory.seed = 0
    try:
        return detect(text)
    except LangDetectException:
        return "unk"


# ---------- основной конвейер ----------
def build_corpus(cfg: dict):
    o = get_opts(cfg, "pr3")
    if not RAW.exists():
        raise SystemExit(f"Нет {RAW}. Сначала выполните: python run.py download-arxiv")
    raw = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line]
    log = [("загружено из arXiv API", 0, len(raw))]

    def stage(name, docs, keep):
        out = [d for d in docs if keep(d)]
        log.append((name, len(docs) - len(out), len(out)))
        return out

    docs = []
    for d in raw:
        d["title"] = clean_abstract(d["title"])
        d["abstract"] = clean_abstract(d["abstract"])
        d["text"] = f'{d["title"]}. {d["abstract"]}'
        d["n_tokens"] = len(tokens(d["abstract"]))
        docs.append(d)

    docs = stage("пустые/слишком короткие/длинные аннотации", docs,
                 lambda d: o["min_words"] <= d["n_tokens"] <= o["max_words"])

    def not_junk(d):
        chars = [c for c in d["abstract"] if not c.isspace()]
        alpha = sum(c.isalpha() for c in chars) / max(1, len(chars))
        return not _WITHDRAWN.search(d["abstract"]) and alpha >= 0.6
    docs = stage("мусор (withdrawn, мало букв)", docs, not_junk)

    n0 = len(docs)
    docs, dropped = dedup_simhash(docs, o["simhash_max_dist"])
    log.append((f"дубликаты (SimHash, расстояние ≤ {o['simhash_max_dist']})", n0 - len(docs), len(docs)))

    for d in docs:
        d["lang"] = detect_lang(d["text"])
    lang_before = Counter(d["lang"] for d in docs)
    docs = stage(f"язык не из {o['langs_allowed']}", docs, lambda d: d["lang"] in o["langs_allowed"])

    n0 = len(docs)
    docs = docs[: o["target_docs"]]
    log.append((f"обрезка до {o['target_docs']}", n0 - len(docs), len(docs)))
    if len(docs) < o["target_docs"]:
        print(f"ВНИМАНИЕ: документов {len(docs)} < {o['target_docs']}. "
              f"Увеличьте arxiv.fetch_docs и перезапустите download-arxiv.")

    out_docs = []
    for d in docs:
        out_docs.append({
            "id": d["id"], "title": d["title"], "abstract": d["abstract"], "text": d["text"],
            "published": d["published"], "year": int(d["published"][:4]) if d["published"] else None,
            "primary_category": d["primary_category"], "categories": d["categories"],
            "lang": d["lang"], "n_tokens": d["n_tokens"], "query": d["query"],
            "source": "arXiv API (export.arxiv.org)", "license": "CC0 (метаданные arXiv)",
        })
    path = cfg["paths"]["corpus"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in out_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    out = results_dir(cfg, "pr3")
    with open(out / "filter_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["этап", "удалено", "осталось"])
        w.writerows(log)
    for name, rem, left in log:
        print(f"  {name:55s} -{rem:5d}  осталось {left}")
    print(f"Корпус: {len(out_docs)} документов → {path}")
    return out_docs, log, lang_before


def load_corpus(cfg: dict) -> list[dict]:
    path = cfg["paths"]["corpus"]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]