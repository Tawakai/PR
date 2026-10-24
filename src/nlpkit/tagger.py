"""HMM и CRF POS-теггеры (ПР 4)."""
from collections import defaultdict
import numpy as np


def read_conllu(path):
    sents, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            c = line.split("\t")
            if len(c) >= 4 and c[0].isdigit():
                cur.append((c[1].lower(), c[3]))
    if cur:
        sents.append(cur)
    return sents


def train_hmm(sents):
    trans = defaultdict(lambda: defaultdict(int))
    emit = defaultdict(lambda: defaultdict(int))
    tags = set()
    for sent in sents:
        prev = "<S>"
        for w, t in sent:
            trans[prev][t] += 1
            emit[t][w] += 1
            tags.add(t)
            prev = t
        trans[prev]["</S>"] += 1
    tags = sorted(tags)
    # добавляем <S> и </S> в индексы для матрицы переходов
    all_tags = ["<S>"] + tags + ["</S>"]
    tidx = {t: i for i, t in enumerate(all_tags)}
    A = np.zeros((len(all_tags), len(all_tags)))
    for p, d in trans.items():
        if p not in tidx:
            continue
        tot = sum(d.values())
        for t, c in d.items():
            if t in tidx:
                A[tidx[p], tidx[t]] = c / tot
    B = {}
    for t, d in emit.items():
        tot = sum(d.values())
        B[t] = {w: c / tot for w, c in d.items()}
    return A, B, tags, tidx  # tags без <S></S>, tidx с ними


def viterbi(words, A, B, tags, tidx):
    n, m = len(words), len(tags)
    # dp для реальных тегов (без <S>, </S>)
    real_idx = [tidx[t] for t in tags]
    dp = np.full((n, m), -np.inf)
    back = np.zeros((n, m), dtype=int)

    start = tidx["<S>"]
    for j, t in enumerate(tags):
        dp[0, j] = np.log(A[start, real_idx[j]] + 1e-9) + np.log(B.get(t, {}).get(words[0], 1e-6))

    for i in range(1, n):
        for j, t in enumerate(tags):
            scores = dp[i - 1] + np.log(A[real_idx, real_idx[j]] + 1e-9)
            best = int(np.argmax(scores))
            dp[i, j] = scores[best] + np.log(B.get(t, {}).get(words[i], 1e-6))
            back[i, j] = best

    best_last = int(np.argmax(dp[-1]))
    path = [best_last]
    for i in range(n - 1, 0, -1):
        path.append(back[i, path[-1]])
    path.reverse()
    return [tags[i] for i in path]


def evaluate_hmm(sents, A, B, tags, tidx):
    correct = total = 0
    for sent in sents:
        words = [w for w, _ in sent]
        gold = [t for _, t in sent]
        pred = viterbi(words, A, B, tags, tidx)
        correct += sum(p == g for p, g in zip(pred, gold))
        total += len(gold)
    return correct / total


def train_crf(sents):
    import sklearn_crfsuite
    def word2features(sent, i):
        w = sent[i][0]
        return {
            "w": w,
            "suf1": w[-1:],
            "suf2": w[-2:],
            "suf3": w[-3:],
            "suf4": w[-4:],
            "upper": w.isupper(),
            "digit": any(ch.isdigit() for ch in w),
            "prev": sent[i - 1][0] if i > 0 else "<S>",
            "next": sent[i + 1][0] if i < len(sent) - 1 else "</S>",
        }
    X = [[word2features(s, i) for i in range(len(s))] for s in sents]
    y = [[t for _, t in s] for s in sents]
    crf = sklearn_crfsuite.CRF(max_iterations=100, algorithm="lbfgs")
    crf.fit(X, y)
    return crf