"""Разметка вдвоём: выборки для раундов, Cohen's κ, Krippendorff's α, разбор расхождений (ПР 3)."""
import csv
import json
import random
import re
from collections import Counter

from nlpkit.corpus import load_corpus
from nlpkit.opts import get_opts

LABELS = ["nonideal", "other"]
# только для стратификации выборки (чтобы в 200 примерах были оба класса); аннотаторам НЕ показывается
WEAK = re.compile(r"non-?ideal|strongly[- ]coupled|warm dense matter|coupling parameter|"
                  r"degenerate plasma|dense plasma|coulomb (?:crystal|coupling|liquid)", re.I)

GUIDE_V1 = """# Инструкция по разметке, версия 1 (ЧЕРНОВИК: отредактируйте под свою диссертацию)

**Задача.** Бинарная классификация аннотаций arXiv: относится ли работа к неидеальной плазме.

**Единица разметки.** Заголовок + аннотация статьи. Решение принимается только по этому тексту.

## Классы
- `nonideal`: центральный объект или главный результат работы связан с плазмой в режиме сильной
  неидеальности: параметр связи Γ ≳ 1 (энергия межчастичного взаимодействия сравнима с кинетической
  или больше неё), сильно связанная (strongly coupled) плазма, неидеальные поправки к уравнению
  состояния, ионизации, транспорту, теплая плотная материя (warm dense matter), плотная вырожденная плазма.
- `other`: всё остальное. Идеальная или слабосвязанная плазма (токамаки, космическая плазма, плазменные
  разряды, диагностика), работы не про плазму.

## Пограничные случаи (правила по умолчанию)
1. Пылевая плазма: `nonideal`, только если в аннотации явно сильная связь (кристалл, жидкость, Γ).
2. Ультрахолодная плазма: `nonideal`, если упомянута сильная связь или корреляции.
3. Кварк-глюонная плазма: `other` (не кулоновская неидеальность, если не сказано иного).
4. Слово «dense» без указания на сильные межчастичные корреляции: `other`.
5. Если неидеальность упомянута вскользь (одно предложение), а работа про другое: `other`.
6. Не уверен: выбери тот класс, который ближе по правилам 1–5, и запиши номер примера в обсуждение.

## Как размечать
Каждый размечает независимо, не смотрит на разметку другого. Время на пример: до 1 минуты.
Допустимые значения поля `label`: `nonideal` или `other` (маленькими буквами, без пробелов).
"""

LS_CONFIG = """<View>
  <Header value="$title"/>
  <Text name="text" value="$text"/>
  <Choices name="label" toName="text" choice="single" showInline="true">
    <Choice value="nonideal"/>
    <Choice value="other"/>
  </Choices>
</View>
"""


def ann_dir(cfg):
    return cfg["paths"]["annotated"]


# ---------- создание раунда ----------
def make_round(cfg: dict, rnd: int, n: int) -> bool:
    d = ann_dir(cfg) / f"round{rnd}"
    if d.exists():
        return False
    docs = load_corpus(cfg)
    used = set()
    for prev in ann_dir(cfg).glob("round*/sample.json"):
        used |= set(json.loads(prev.read_text())["ids"])
    pool = [x for x in docs if x["id"] not in used]
    rng = random.Random(cfg["project"]["seed"] + rnd)
    pos = [x for x in pool if WEAK.search(x["text"])]
    neg = [x for x in pool if not WEAK.search(x["text"])]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos))
    chosen = pos[:k] + neg[: n - k]
    rng.shuffle(chosen)
    order_b = chosen[:]
    random.Random(cfg["project"]["seed"] + 100 + rnd).shuffle(order_b)

    d.mkdir(parents=True, exist_ok=True)
    (d / "sample.json").write_text(json.dumps(
        {"ids": [x["id"] for x in chosen], "n": len(chosen), "weak_positive": k}), encoding="utf-8")
    for name, order in (("A", chosen), ("B", order_b)):
        with open(d / f"ann_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "title", "text", "label"])
            for x in order:
                w.writerow([x["id"], x["title"], x["abstract"], ""])
    (d / "labelstudio_tasks.json").write_text(json.dumps(
        [{"data": {"doc_id": x["id"], "title": x["title"], "text": x["abstract"]}} for x in chosen],
        ensure_ascii=False, indent=1), encoding="utf-8")
    (ann_dir(cfg) / "labelstudio_config.xml").write_text(LS_CONFIG, encoding="utf-8")
    g1 = ann_dir(cfg) / "guidelines_v1.md"
    if not g1.exists():
        g1.write_text(GUIDE_V1, encoding="utf-8")
    if rnd == 2:
        g2 = ann_dir(cfg) / "guidelines_v2.md"
        if not g2.exists():
            g2.write_text(g1.read_text(encoding="utf-8").replace("версия 1", "версия 2") +
                          "\n## Изменения после раунда 1\n(заполните по итогам разбора расхождений)\n",
                          encoding="utf-8")
    print(f"Раунд {rnd}: создано {len(chosen)} примеров → {d}")
    return True


# ---------- метрики согласия ----------
def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in set(a) | set(b)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def krippendorff_alpha_nominal(a: list, b: list) -> float:
    """α для двух аннотаторов без пропусков, номинальная шкала (через матрицу совпадений)."""
    n = 2 * len(a)
    nc = Counter(a) + Counter(b)
    do = 2 * sum(x != y for x, y in zip(a, b))            # несовпавшие упорядоченные пары
    de_den = n * n - sum(c * c for c in nc.values())
    return 1 - (n - 1) * do / de_den if de_den else 1.0


def _interpret(k: float) -> str:
    """Шкала Landis & Koch (1977)."""
    if k < 0:
        return "хуже случайного"
    for lim, txt in [(0.2, "слабое"), (0.4, "удовлетворительное"), (0.6, "умеренное"), (0.8, "значительное")]:
        if k < lim:
            return txt
    return "почти полное"


def _read(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    key = "doc_id" if rows and "doc_id" in rows[0] else "id"
    return {r[key]: (r.get("label") or "").strip() for r in rows}, rows


def agreement(cfg: dict, rnd: int, out) -> dict | None:
    d = ann_dir(cfg) / f"round{rnd}"
    la, rows_a = _read(d / "ann_A.csv")
    lb, _ = _read(d / "ann_B.csv")
    fa = sum(bool(v) for v in la.values())
    fb = sum(bool(v) for v in lb.values())
    if fa < len(la) or fb < len(lb):
        print(f"Раунд {rnd}: разметка не закончена (A: {fa}/{len(la)}, B: {fb}/{len(lb)})")
        return None
    bad = [(k, v) for src in (la, lb) for k, v in src.items() if v not in LABELS]
    if bad or set(la) != set(lb):
        print(f"Раунд {rnd}: недопустимые метки или разные наборы документов: {bad[:5]}")
        return None
    ids = list(la)
    a, b = [la[i] for i in ids], [lb[i] for i in ids]
    kappa, alpha = cohen_kappa(a, b), krippendorff_alpha_nominal(a, b)
    agree = sum(x == y for x, y in zip(a, b)) / len(a)
    conf = Counter(zip(a, b))
    res = {"round": rnd, "n": len(ids), "percent_agreement": round(agree, 3),
           "cohen_kappa": round(kappa, 3), "krippendorff_alpha": round(alpha, 3),
           "interpretation": _interpret(kappa),
           "A_counts": dict(Counter(a)), "B_counts": dict(Counter(b)),
           "confusion_A_rows_B_cols": {f"{x}|{y}": conf.get((x, y), 0) for x in LABELS for y in LABELS},
           "n_disagreements": sum(x != y for x, y in zip(a, b))}
    (out / f"agreement_r{rnd}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    adj = d / "adjudication.csv"
    if not adj.exists():
        text = {r["doc_id"] if "doc_id" in r else r["id"]: r for r in rows_a}
        with open(adj, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "title", "text", "label_A", "label_B", "final", "comment"])
            for i in ids:
                if la[i] != lb[i]:
                    w.writerow([i, text[i]["title"], text[i]["text"], la[i], lb[i], "", ""])
    print(f"Раунд {rnd}: n={res['n']}, совпадение {agree:.1%}, κ={kappa:.3f} ({res['interpretation']}), "
          f"α={alpha:.3f}, расхождений {res['n_disagreements']}")
    print(f"  Разберите расхождения в {adj}: заполните колонку final (и comment: почему)")
    return res


def gold(cfg: dict) -> dict | None:
    rows, status = [], []
    for rnd in (1, 2):
        d = ann_dir(cfg) / f"round{rnd}"
        if not (d / "adjudication.csv").exists():
            continue
        la, _ = _read(d / "ann_A.csv")
        lb, _ = _read(d / "ann_B.csv")
        with open(d / "adjudication.csv", encoding="utf-8-sig") as f:
            fin = {r["doc_id"]: r["final"].strip() for r in csv.DictReader(f)}
        missing = [i for i in la if la[i] != lb[i] and fin.get(i) not in LABELS]
        if missing:
            status.append(f"раунд {rnd}: не разрешено {len(missing)} расхождений")
            continue
        for i in la:
            rows.append([i, la[i] if la[i] == lb[i] else fin[i], rnd,
                         "agreed" if la[i] == lb[i] else "adjudicated"])
    for s in status:
        print("  gold:", s)
    if not rows:
        return None
    path = ann_dir(cfg) / "gold.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "label", "round", "source"])
        w.writerows(rows)
    bal = Counter(r[1] for r in rows)
    print(f"Эталонная разметка: {len(rows)} примеров → {path}  {dict(bal)}")
    return {"n": len(rows), "balance": dict(bal)}


def run_rounds(cfg: dict, out) -> dict:
    o = get_opts(cfg, "pr3")
    res = {}
    if make_round(cfg, 1, o["ann_round1"]):
        print("  Дальше: разметьте round1/ann_A.csv (вы) и round1/ann_B.csv (коллега), независимо.")
        print("  Либо Label Studio: импорт labelstudio_tasks.json + labelstudio_config.xml, экспорт CSV.")
    r1 = agreement(cfg, 1, out)
    if r1:
        res["round1"] = r1
        if make_round(cfg, 2, o["ann_round2"]):
            print("  Раунд 2: сначала обновите инструкцию (guidelines_v2.md), потом размечайте round2/.")
        r2 = agreement(cfg, 2, out) if (ann_dir(cfg) / "round2").exists() else None
        if r2:
            res["round2"] = r2
    g = gold(cfg)
    if g:
        res["gold"] = g
    return res