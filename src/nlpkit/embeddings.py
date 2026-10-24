"""Эмбеддинги: PPMI+SVD, Word2Vec, FastText, визуализация (ПР 6)."""
import numpy as np
from collections import Counter, defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def build_cooccurrence(tokens, window=5, min_count=5):
    # 1. Считаем частоты
    freq = Counter(tokens)
    # 2. Оставляем только частые слова и даём им стабильные индексы
    id2w = {}
    vocab = {}
    for w, c in freq.items():
        if c >= min_count:
            idx = len(vocab)
            vocab[w] = idx
            id2w[idx] = w
    # 3. Считаем кооккурренцию ТОЛЬКО для слов из vocab
    cooc = defaultdict(Counter)
    for i, w in enumerate(tokens):
        if w not in vocab:
            continue
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i != j and tokens[j] in vocab:
                cooc[w][tokens[j]] += 1
    return cooc, vocab, id2w


def ppmi_matrix(cooc, vocab, id2w):
    n = len(vocab)
    mat = np.zeros((n, n))
    total = 0
    row_sum = Counter()
    col_sum = Counter()
    for w, c in cooc.items():
        for ctx, cnt in c.items():
            total += cnt
            row_sum[w] += cnt
            col_sum[ctx] += cnt
    for w, c in cooc.items():
        for ctx, cnt in c.items():
            # гарантируем, что w и ctx есть в vocab
            if w not in vocab or ctx not in vocab:
                continue
            pmi = np.log((cnt * total) / (row_sum[w] * col_sum[ctx]))
            mat[vocab[w], vocab[ctx]] = max(pmi, 0)
    return mat


def svd_embeddings(mat, dim=100):
    svd = TruncatedSVD(n_components=dim)
    emb = svd.fit_transform(mat)
    return normalize(emb)


def train_word2vec(tokens, dim=100, window=5, sg=1, neg=5, min_count=5):
    from gensim.models import Word2Vec
    sents = [tokens[i:i + 20] for i in range(0, len(tokens), 20)]
    model = Word2Vec(sents, vector_size=dim, window=window, sg=sg,
                     negative=neg, min_count=min_count, workers=4)
    return model


def train_fasttext(tokens, dim=100, window=5, min_count=5):
    from gensim.models import FastText
    sents = [tokens[i:i + 20] for i in range(0, len(tokens), 20)]
    model = FastText(sents, vector_size=dim, window=window, min_count=min_count)
    return model


def nearest_neighbors(model, word, topn=10):
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        return []


def analogy(model, a, b, c, topn=3):
    try:
        return model.wv.most_similar(positive=[b, c], negative=[a], topn=topn)
    except KeyError:
        return []


def visualize(embeddings, words, method="tsne", path="emb.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    if method == "tsne":
        proj = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)
    elif method == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2).fit_transform(embeddings)
    else:
        import umap
        proj = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(12, 8))
    plt.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.5)
    for i, w in enumerate(words):
        plt.annotate(w, (proj[i, 0], proj[i, 1]), fontsize=8)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def load_pretrained(path):
    from gensim.models import KeyedVectors
    return KeyedVectors.load_word2vec_format(path, binary=False)