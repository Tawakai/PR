"""Настройки ПР 3 по умолчанию. config.yaml можно не трогать:
блоки arxiv: и pr3: (если добавите) переопределяют значения ниже."""

DEFAULTS = {
    "arxiv": {
        "fetch_docs": 5600,       # сколько скачать (запас на фильтры и дубликаты)
        "keyword_docs": 1500,     # из них по ключевым фразам про неидеальную плазму
        "batch": 100,
        "pause_sec": 3.0,         # правила arXiv API: не чаще 1 запроса в 3 с
        "base_category": "physics.plasm-ph",
        "keyword_query": (
            '(abs:"strongly coupled plasma" OR abs:"non-ideal plasma" OR abs:"nonideal plasma"'
            ' OR abs:"warm dense matter" OR abs:"strongly coupled Coulomb"'
            ' OR abs:"one-component plasma" OR abs:"dense plasma")'
            ' AND (cat:physics.plasm-ph OR cat:cond-mat.stat-mech OR cat:astro-ph.SR'
            ' OR cat:physics.atom-ph OR cat:cond-mat.mtrl-sci)'
        ),
    },
    "pr3": {
        "target_docs": 5000,
        "min_words": 40,
        "max_words": 600,
        "simhash_max_dist": 3,    # допустимо 1..3 (блочный поиск по 4 частям хэша)
        "langs_allowed": ["en"],
        "ref_pages": 2500,        # статей английской Википедии для эталонного корпуса
        "top_terms": 50,
        "ann_round1": 200,
        "ann_round2": 100,
    },
}


def get_opts(cfg: dict, key: str) -> dict:
    out = dict(DEFAULTS[key])
    out.update(cfg.get(key) or {})
    return out