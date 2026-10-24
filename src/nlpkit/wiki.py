"""Загрузка статей Википедии через MediaWiki API (ПР 1)."""
import json
import re
import time

import requests

HEADERS = {"User-Agent": "nlp-course-student/0.1 (educational project)"}


def _api(lang: str, **params) -> dict:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params.update(format="json", formatversion=2)
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get_links(lang: str, title: str) -> list[str]:
    """Заголовки статей, на которые ссылается данная (основное пространство имён)."""
    titles, cont = [], {}
    while True:
        data = _api(lang, action="query", titles=title, prop="links",
                    plnamespace=0, pllimit="max", **cont)
        page = data["query"]["pages"][0]
        titles += [link["title"] for link in page.get("links", [])]
        if "continue" not in data:
            return titles
        cont = data["continue"]


def get_text(lang: str, title: str) -> str:
    """Полный текст статьи без разметки."""
    data = _api(lang, action="query", titles=title, prop="extracts",
                explaintext=1, redirects=1)
    page = data["query"]["pages"][0]
    return page.get("extract", "")


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def download(cfg: dict) -> None:
    wcfg = cfg["wiki"]
    lang = wcfg["lang"]
    raw_dir = cfg["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    titles = [wcfg["seed_title"]] + get_links(lang, wcfg["seed_title"])
    print(f"Кандидатов: {len(titles)}")

    meta, total = [], 0
    for title in titles:
        if total >= wcfg["target_words"]:
            break
        try:
            text = get_text(lang, title)
        except requests.RequestException as e:
            print(f"  пропуск {title}: {e}")
            continue
        n = count_words(text)
        if n < wcfg["min_article_words"]:
            continue
        idx = len(meta) + 1
        (raw_dir / f"{idx:04d}.txt").write_text(text, encoding="utf-8")
        meta.append({
            "file": f"{idx:04d}.txt",
            "title": title,
            "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "lang": lang,
            "words": n,
            "license": "CC BY-SA 4.0",
        })
        total += n
        print(f"  [{idx}] {title}: {n} слов (всего {total})")
        time.sleep(wcfg["pause_sec"])

    (raw_dir / "sources.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {len(meta)} статей, {total} слов → {raw_dir}")

def download_kk(cfg: dict) -> None:
    """Казахская Википедия (статья о Венгрии и связанные) → data/raw_kk (для ПР 2)."""
    kcfg, lang = cfg["wiki_kk"], "kk"
    raw_dir = cfg["paths"]["raw"].parent / "raw_kk"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seed = kcfg["seed_title"]
    if not get_text(lang, seed):    # заголовок не найден: берём первый результат поиска
        hits = _api(lang, action="query", list="search", srsearch=seed, srlimit=1)["query"]["search"]
        seed = hits[0]["title"] if hits else seed
        print("Стартовая статья:", seed)
    titles = list(dict.fromkeys([seed] + get_links(lang, seed)))
    total, n = 0, 0
    for title in titles:
        if total >= kcfg["target_words"]:
            break
        text = get_text(lang, title)
        words = count_words(text)
        if words < kcfg["min_article_words"]:
            continue
        n += 1
        (raw_dir / f"{n:04d}.txt").write_text(text, encoding="utf-8")
        total += words
        print(f"  [{n}] {title}: {words} слов (всего {total})")
        time.sleep(cfg["wiki"]["pause_sec"])
    print(f"Готово: {n} статей, {total} слов → {raw_dir}")