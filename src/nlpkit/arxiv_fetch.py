"""Сбор метаданных статей arXiv через официальный API (ПР 3)."""
import json
import re
import time
import xml.etree.ElementTree as ET

import requests

from nlpkit.config import ROOT
from nlpkit.opts import get_opts

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
HEADERS = {"User-Agent": "nlp-course-student/0.1 (educational project)"}
RAW = ROOT / "data/raw_arxiv/arxiv_raw.jsonl"


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        raw_id = e.findtext("a:id", "", NS)
        if "/abs/" not in raw_id:          # сообщение об ошибке API, а не статья
            continue
        prim = e.find("x:primary_category", NS)
        out.append({
            "id": re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1]),
            "title": _ws(e.findtext("a:title", "", NS)),
            "abstract": _ws(e.findtext("a:summary", "", NS)),
            "published": (e.findtext("a:published", "", NS) or "")[:10],
            "primary_category": prim.get("term") if prim is not None else "",
            "categories": [c.get("term") for c in e.findall("a:category", NS)],
        })
    return out


def _get(params: dict, pause: float, retries: int = 4) -> str:
    for k in range(retries):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r.text
            print(f"  HTTP {r.status_code}, жду и повторяю")
        except requests.RequestException as e:
            print(f"  сеть: {e}, жду и повторяю")
        time.sleep(pause * (k + 2))
    raise RuntimeError("arXiv API не отвечает, попробуйте позже")


def search(query: str, n: int, batch: int, pause: float, sort_by: str,
           seen: set, tag: str) -> list[dict]:
    docs, start = [], 0
    while len(docs) < n:
        xml = _get({"search_query": query, "start": start, "max_results": batch,
                    "sortBy": sort_by, "sortOrder": "descending"}, pause)
        entries = parse_feed(xml)
        if not entries:
            break
        for d in entries:
            if d["id"] not in seen:
                seen.add(d["id"])
                d["query"] = tag
                docs.append(d)
        start += batch
        print(f"  [{tag}] получено {len(docs)} из {n}")
        time.sleep(pause)
    return docs[:n]


def download(cfg: dict) -> None:
    a = get_opts(cfg, "arxiv")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    print("1/2 статьи по ключевым фразам (неидеальная плазма)")
    kw = search(a["keyword_query"], a["keyword_docs"], a["batch"], a["pause_sec"],
                "relevance", seen, "keyword")
    print("2/2 статьи категории", a["base_category"])
    base = search(f"cat:{a['base_category']}", a["fetch_docs"] - len(kw), a["batch"],
                  a["pause_sec"], "submittedDate", seen, "base")
    docs = kw + base
    with open(RAW, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Готово: {len(docs)} записей ({len(kw)} keyword + {len(base)} base) → {RAW}")