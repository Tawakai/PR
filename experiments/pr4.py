import csv, random
from pathlib import Path
from nlpkit.config import results_dir
from nlpkit.ngram import build_counts, add_k_probs, perplexity, tokenize
from nlpkit.tagger import read_conllu, train_hmm, evaluate_hmm, train_crf


def main(cfg):
    out = results_dir(cfg, "pr4")
    raw = cfg["paths"]["clean"]

    # 1. Загрузка корпуса
    texts = [p.read_text(encoding="utf-8") for p in sorted(raw.glob("*.txt"))]
    all_tokens = []
    for t in texts:
        all_tokens += tokenize(t)
    rng = random.Random(cfg["project"]["seed"])
    rng.shuffle(all_tokens)
    n = len(all_tokens)
    tr = all_tokens[: int(n * 0.8)]
    dev = all_tokens[int(n * 0.8): int(n * 0.9)]
    test = all_tokens[int(n * 0.9):]

    # 2. N-граммы и перплексия
    vocab = set(tr)
    rows = []
    for order in [1, 2, 3]:
        counts = build_counts(tr, order)
        for k in [0.01, 0.1, 1.0]:
            ppl = perplexity(" ".join(test[:5000]), counts, vocab, k)
            rows.append([f"n={order}", f"add-k={k}", round(ppl, 2)])
    with open(out / "ngram_perplexity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "smoothing", "perplexity"])
        w.writerows(rows)

    # 3. POS-теггеры (русский UD)
    ud_path = Path("data/external/ru_gsd-ud-test.conllu")
    if not ud_path.exists():
        import requests
        ud_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(cfg["pr4"]["ud"]["ru"], timeout=60)
        ud_path.write_bytes(r.content)
    sents = read_conllu(ud_path)
    split = int(len(sents) * 0.8)
    train_s = sents[:split]
    test_s = sents[split:]

    A, B, tags, tidx = train_hmm(train_s)
    hmm_acc = evaluate_hmm(test_s, A, B, tags, tidx)
    print(f"HMM accuracy: {hmm_acc:.3f}")

    crf = train_crf(train_s)
    crf_pred = crf.predict([[w for w, _ in s] for s in test_s])
    gold = [[t for _, t in s] for s in test_s]
    crf_acc = sum(p == g for ps, gs in zip(crf_pred, gold) for p, g in zip(ps, gs)) / sum(len(g) for g in gold)
    print(f"CRF accuracy: {crf_acc:.3f}")

    with open(out / "pos_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy"])
        w.writerow(["HMM", round(hmm_acc, 3)])
        w.writerow(["CRF", round(crf_acc, 3)])