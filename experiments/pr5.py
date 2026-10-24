import csv, random
from pathlib import Path
from nlpkit.config import results_dir
from nlpkit.syntax import parse_spacy, render_tree, extract_features, path_between


def main(cfg):
    out = results_dir(cfg, "pr5")
    raw = cfg["paths"]["clean"]

    # 1. 50 предложений из корпуса
    texts = []
    for p in sorted(raw.glob("*.txt")):
        for para in p.read_text(encoding="utf-8").split("\n"):
            if len(para) > 80:
                texts.append(para)
    rng = random.Random(cfg["project"]["seed"])
    rng.shuffle(texts)
    texts = texts[:50]

    docs = parse_spacy(texts)

    # 2. Визуализация деревьев (первые 10)
    vis_dir = out / "trees"
    vis_dir.mkdir(exist_ok=True)
    for i, doc in enumerate(docs[:10]):
        render_tree(doc, vis_dir / f"tree_{i:02d}.svg")

    # 3. Извлечение признаков
    rows = []
    for i, doc in enumerate(docs):
        f = extract_features(doc)
        rows.append([i, f["depth"], f["clauses"], len(f["triplets"])])
    with open(out / "syntax_features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "depth", "clauses", "triplets"])
        w.writerows(rows)
    print(f"Извлечено признаков для {len(rows)} предложений")

    # 4. UAS/LAS на ручной разметке (упрощённо: сравниваем spaCy и Stanza)
    try:
        import stanza
        nlp_stanza = stanza.Pipeline("ru", processors="tokenize,pos,lemma,depparse", verbose=False)
        stanza_docs = [nlp_stanza(t) for t in texts[:20]]
        # Считаем UAS/LAS между spaCy и Stanza как "эталон"
        uas_correct = las_correct = total = 0
        for doc_sp, doc_st in zip(docs[:20], stanza_docs):
            # Упрощённое выравнивание по индексам токенов
            sp_tokens = [t for t in doc_sp]
            st_tokens = [w for s in doc_st.sentences for w in s.words]
            for a, b in zip(sp_tokens, st_tokens):
                total += 1
                if a.head.i == b.head - 1 if hasattr(b, "head") else False:
                    uas_correct += 1
                    if a.dep_ == b.deprel:
                        las_correct += 1
        print(f"UAS≈{uas_correct/total:.3f} LAS≈{las_correct/total:.3f}")
    except Exception as e:
        print("Stanza пропущен:", e)

    # 5. CoNLL-U: 30 предложений из корпуса → файл
    conllu_path = out / "sample_30.conllu"
    with open(conllu_path, "w", encoding="utf-8") as f:
        for doc in docs[:30]:
            for i, token in enumerate(doc, 1):
                f.write(f"{i}\t{token.text}\t{token.lemma_}\t{token.pos_}\t_\t"
                        f"{token.morph}\t{token.head.i + 1}\t{token.dep_}\t_\t_\n")
            f.write("\n")
    print(f"CoNLL-U: {conllu_path}")

    # 6. Классификатор с синтаксическими признаками vs без
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score
    import numpy as np

    # Псевдо-метки: длина > 200 символов = 1
    y = np.array([1 if len(t) > 200 else 0 for t in texts])
    vec = TfidfVectorizer(max_features=1000)
    X_tfidf = vec.fit_transform(texts)

    # + синтаксические признаки
    syn_feats = []
    for doc in docs:
        f = extract_features(doc)
        syn_feats.append([f["depth"], f["clauses"], len(f["triplets"])])
    from scipy.sparse import hstack, csr_matrix
    X_full = hstack([X_tfidf, csr_matrix(np.array(syn_feats))])

    clf = LogisticRegression(max_iter=1000)
    cv_tfidf = cross_val_score(clf, X_tfidf, y, cv=3, scoring="f1_macro", error_score="raise")
    cv_full = cross_val_score(clf, X_full, y, cv=3, scoring="f1_macro", error_score="raise")

    print(f"F1 без синтаксиса: {cv_tfidf.mean():.3f} ± {cv_tfidf.std():.3f}")
    print(f"F1 с синтаксисом: {cv_full.mean():.3f} ± {cv_full.std():.3f}")

    with open(out / "classification.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature_set", "macro_f1_mean", "macro_f1_std"])
        w.writerow(["TF-IDF", round(cv_tfidf.mean(), 3), round(cv_tfidf.std(), 3)])
        w.writerow(["TF-IDF + syntax", round(cv_full.mean(), 3), round(cv_full.std(), 3)])