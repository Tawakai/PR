"""Эталонный корпус: случайные статьи английской Википедии (вводные абзацы), ПР 3."""
import time

import requests

from nlpkit.config import ROOT
from nlpkit.opts import get_opts

REF = ROOT / "data/external/wiki_en_ref.txt"


def ensure_reference(cfg: dict) -> str:
    if REF.exists():
        return REF.read_text(encoding="utf-8")
    from nlpkit.wiki import _api              # тот же клиент MediaWiki, что в ПР 1
    target = get_opts(cfg, "pr3")["ref_pages"]
    REF.parent.mkdir(parents=True, exist_ok=True)
    seen, texts, fails = set(), [], 0
    print(f"Скачиваю эталон: {target} случайных статей en.wikipedia.org (вводные абзацы)")
    while len(texts) < target and fails < 20:
        try:
            data = _api("en", action="query", generator="random", grnnamespace=0, grnlimit=20,
                        prop="extracts", exintro=1, explaintext=1, exlimit="max")
        except requests.RequestException as e:
            fails += 1
            print("  сеть:", e)
            time.sleep(2)
            continue
        for p in data.get("query", {}).get("pages", []):
            t = (p.get("extract") or "").strip().replace("\n", " ")
            if p["pageid"] not in seen and len(t.split()) >= 30:
                seen.add(p["pageid"])
                texts.append(t)
        if len(texts) % 500 < 20:
            print(f"  {len(texts)}/{target}")
        time.sleep(0.2)
    REF.write_text("\n".join(texts), encoding="utf-8")
    return REF.read_text(encoding="utf-8")