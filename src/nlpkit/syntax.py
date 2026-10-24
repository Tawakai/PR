"""Парсинг и синтаксические признаки (ПР 5)."""
import spacy
from spacy import displacy

_nlp = None

def get_spacy():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("ru_core_news_md")
        _nlp.max_length = 2_000_000
    return _nlp


def parse_spacy(texts):
    nlp = get_spacy()
    return [nlp(t) for t in texts]


def render_tree(doc, path):
    svg = displacy.render(doc, style="dep", jupyter=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def extract_features(doc):
    """Триплеты SVO, глубина дерева, число клауз."""
    feats = {"depth": 0, "clauses": 0, "triplets": []}
    for token in doc:
        # глубина
        d = 0
        t = token
        while t.head != t:
            d += 1
            t = t.head
        feats["depth"] = max(feats["depth"], d)
        if token.dep_ in ("ccomp", "xcomp", "advcl", "acl"):
            feats["clauses"] += 1
    for token in doc:
        if token.pos_ == "VERB":
            subj = [c for c in token.children if c.dep_ in ("nsubj", "nsubj:pass")]
            obj = [c for c in token.children if c.dep_ in ("obj", "iobj")]
            if subj and obj:
                feats["triplets"].append((subj[0].text, token.text, obj[0].text))
    return feats


def path_between(doc, ent1, ent2):
    """Путь зависимостей между двумя сущностями."""
    for token in doc:
        if token.text == ent1:
            path = []
            t = token
            while t.head != t:
                path.append(t.dep_)
                t = t.head
            for token2 in doc:
                if token2.text == ent2:
                    path2 = []
                    t = token2
                    while t.head != t:
                        path2.append(t.dep_)
                        t = t.head
                    return path, path2
    return [], []