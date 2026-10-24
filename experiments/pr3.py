import json
import time

from nlpkit import annotation, datasheet, stats
from nlpkit.config import results_dir
from nlpkit.corpus import build_corpus
from nlpkit.opts import get_opts
from nlpkit.refwiki import ensure_reference


def main(cfg: dict) -> None:
    t0 = time.time()
    o = get_opts(cfg, "pr3")
    seed = cfg["project"]["seed"]
    out = results_dir(cfg, "pr3")

    print("== 1. Корпус: очистка, фильтры, дедупликация, язык ==")
    docs, log, lang_before = build_corpus(cfg)

    print("== 2. Статистики ==")
    st, cnt, toks_per_doc = stats.corpus_stats(docs, lang_before, seed, out)
    for k in ("n_docs", "n_tokens", "n_types", "ttr", "ttr_10k", "len_median", "lang_share_collected"):
        print(f"  {k:22s} {st[k]}")

    print("== 3. Ципф и Хипс ==")
    z = stats.zipf(cnt, out)
    h = stats.heaps(toks_per_doc, seed, out)
    print("  Ципф:", z)
    print("  Хипс:", h)

    print("== 4. Ключевые слова (LLR против en.wikipedia) ==")
    terms = stats.keywords(cnt, ensure_reference(cfg), out, o["top_terms"])
    print("  топ-15:", ", ".join(t[0] for t in terms[:15]))

    print("== 5. Разметка ==")
    ann = annotation.run_rounds(cfg, out)

    summary = {"stats": st, "zipf": z, "heaps": h, "terms": terms, "filter_log": log,
               "simhash_k": o["simhash_max_dist"], "annotation": ann}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("== 6. Datasheet ==")
    datasheet.write(cfg, summary)
    print(f"ПР 3: готово за {time.time() - t0:.0f} с. Результаты: {out}")