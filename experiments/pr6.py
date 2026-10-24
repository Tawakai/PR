import csv, random
from pathlib import Path
import numpy as np
from nlpkit.config import results_dir
from nlpkit.embeddings import (build_cooccurrence, ppmi_matrix, svd_embeddings,
                                train_word2vec, train_fasttext, nearest_neighbors,
                                visualize)
from nlpkit.cleaning import clean_text


def main(cfg):
    out = results_dir(cfg, "pr6")
    raw = cfg["paths"]["clean"]
    texts = [p.read_text(encoding="utf-8") for p in sorted(raw.glob("*.txt"))]
    tokens = []
    for t in texts:
        tokens += clean_text(t).lower().split()
    print(f"Токенов: {len(tokens)}")

    # 1. PPMI + SVD
    cooc, vocab, id2w = build_cooccurrence(tokens, window=5, min_count=5)
    print(f"Словарь PPMI: {len(vocab)}")
    mat = ppmi_matrix(cooc, vocab, id2w)
    emb_ppmi = svd_embeddings(mat, dim=100)

    # 2. Word2Vec skip-gram
    w2v_sg = train_word2vec(tokens, dim=100, window=5, sg=1)
    print(f"W2V SG словарь: {len(w2v_sg.wv)}")

    # 3. Word2Vec CBOW
    w2v_cbow = train_word2vec(tokens, dim=100, window=5, sg=0)

    # 4. FastText
    ft = train_fasttext(tokens, dim=100, window=5)
    print(f"FastText словарь: {len(ft.wv)}")

    # 5. Ближайшие соседи для 20 слов
    words = [w for w, _ in sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:20]]
    rows = []
    for w in words:
        nn_ft = nearest_neighbors(ft, w, topn=5)
        rows.append([w, " | ".join(f"{x[0]}({x[1]:.2f})" for x in nn_ft)])
    with open(out / "nearest_neighbors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["слово", "топ-5 FastText"])
        w.writerows(rows)

    # 6. Визуализация 300 слов (топ по частоте)
    top_words = [w for w, _ in sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:300]]
    idx = [vocab[w] for w in top_words if w in vocab]
    top_words = [w for w in top_words if w in vocab]
    for method in ["pca", "tsne"]:
        path = out / f"viz_{method}.png"
        visualize(emb_ppmi[idx], top_words, method=method, path=str(path))
        print(f"Сохранено: {path}")

    # 7. Внешняя оценка: TF-IDF vs средние эмбеддинги на псевдо-задаче
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import numpy as np

    y = np.array([1 if len(t) > 2000 else 0 for t in texts])
    if len(set(y)) < 2:
        print("В корпусе только один класс, внешняя оценка пропущена")
    else:
        vec = TfidfVectorizer(max_features=2000)
        X_tfidf = vec.fit_transform(texts)
        clf = LogisticRegression(max_iter=1000)
        cv_tfidf = cross_val_score(clf, X_tfidf, y, cv=3, scoring="f1_macro")

        # средние эмбеддинги FastText
        X_emb = []
        for t in texts:
            toks = clean_text(t).lower().split()
            vecs = [ft.wv[w] for w in toks if w in ft.wv]
            X_emb.append(np.mean(vecs, axis=0) if vecs else np.zeros(100))
        X_emb = np.array(X_emb)
        cv_emb = cross_val_score(clf, X_emb, y, cv=3, scoring="f1_macro")

        print(f"F1 TF-IDF: {cv_tfidf.mean():.3f} ± {cv_tfidf.std():.3f}")
        print(f"F1 FastText avg: {cv_emb.mean():.3f} ± {cv_emb.std():.3f}")
        with open(out / "extrinsic.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature", "macro_f1_mean", "macro_f1_std"])
            w.writerow(["TF-IDF", round(cv_tfidf.mean(), 3), round(cv_tfidf.std(), 3)])
            w.writerow(["FastText avg", round(cv_emb.mean(), 3), round(cv_emb.std(), 3)])

    # 8. Контекстные XLM-R (если есть)
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModel.from_pretrained("xlm-roberta-base")
        model.eval()
        sample = texts[0][:512]
        inputs = tok(sample, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        print(f"XLM-R hidden states: {len(out.hidden_states)} слоёв")
    except Exception as e:
        print("XLM-R пропущен:", e)
