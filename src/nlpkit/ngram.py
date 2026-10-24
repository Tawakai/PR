"""N-граммные языковые модели со сглаживанием (ПР 4)."""
import math
from collections import Counter, defaultdict


def tokenize(text):
    return text.lower().split()


def build_counts(train_tokens, n):
    counts = defaultdict(Counter)
    for i in range(len(train_tokens) - n + 1):
        ctx = tuple(train_tokens[i:i + n - 1])
        counts[ctx][train_tokens[i + n - 1]] += 1
    return counts


def add_k_probs(counts, vocab, k):
    probs = {}
    for ctx, c in counts.items():
        total = sum(c.values()) + k * len(vocab)
        probs[ctx] = {w: (c[w] + k) / total for w in vocab}
    return probs


def perplexity(text, counts, vocab, k):
    tokens = tokenize(text)
    n = max(len(c) for c in counts) + 1 if counts else 1
    logp = 0.0
    N = 0
    for i in range(len(tokens) - n + 1):
        ctx = tuple(tokens[i:i + n - 1])
        w = tokens[i + n - 1]
        c = counts.get(ctx, Counter())
        total = sum(c.values()) + k * len(vocab)
        p = (c[w] + k) / total
        logp += math.log(p)
        N += 1
    return math.exp(-logp / N) if N else float("inf")