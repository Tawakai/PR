import csv
import numpy as np
import torch
from nlpkit.config import results_dir
from nlpkit.classifier import (baseline_tfidf, build_vocab, TextDataset,
                                BiLSTM, GRUModel, TextCNN, BiLSTMAttn,
                                train_model, evaluate)
from nlpkit.cleaning import clean_text


def main(cfg):
    out = results_dir(cfg, "pr7")
    raw = cfg["paths"]["clean"]
    texts = [clean_text(p.read_text(encoding="utf-8")) for p in sorted(raw.glob("*.txt"))]
    print(f"Документов: {len(texts)}")

    lengths = [len(t.split()) for t in texts]
    med = np.median(lengths)
    y = np.array([1 if l > med else 0 for l in lengths])
    print(f"Класс 0: {(y == 0).sum()}, класс 1: {(y == 1).sum()}")

    if len(set(y)) < 2:
        print("Один класс — ПР 7 не выполнима на этом корпусе без разметки")
        return

    print("== Baseline ==")
    res = baseline_tfidf(texts, y)
    for name, (m, s) in res.items():
        print(f"{name}: macro-F1 = {m:.3f} ± {s:.3f}")

    from sklearn.model_selection import train_test_split
    tr_texts, te_texts, tr_y, te_y = train_test_split(texts, y, test_size=0.2, random_state=42)
    tr_texts, dev_texts, tr_y, dev_y = train_test_split(tr_texts, tr_y, test_size=0.1, random_state=42)

    vocab = build_vocab(texts, min_count=2)
    print(f"Словарь: {len(vocab)}")

    tr_ds = TextDataset(tr_texts, tr_y, vocab)
    dev_ds = TextDataset(dev_texts, dev_y, vocab)
    te_ds = TextDataset(te_texts, te_y, vocab)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=4, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=4)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=4)

    models = {
        "BiLSTM": BiLSTM(len(vocab), 100, 64, 2),
        "GRU": GRUModel(len(vocab), 100, 64, 2),
        "TextCNN": TextCNN(len(vocab), 100, 64, 2),
        "BiLSTM+Attn": BiLSTMAttn(len(vocab), 100, 64, 2),
    }

    results = []
    from sklearn.metrics import f1_score
    for name, model in models.items():
        print(f"== {name} ==")
        model = train_model(model, tr_loader, dev_loader, epochs=5)
        preds, golds = evaluate(model, te_loader)
        f1 = f1_score(golds, preds, average="macro")
        results.append([name, round(f1, 3)])
        print(f"{name}: F1 = {f1:.3f}")

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "macro_f1"])
        for name, (m, s) in res.items():
            w.writerow([f"baseline_{name}", round(m, 3)])
        w.writerows(results)
    print(f"Сохранено: {out / 'results.csv'}")