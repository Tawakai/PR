"""Baseline (TF-IDF) + BiLSTM/GRU/CNN на PyTorch (ПР 7)."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter


# ---------- Baseline ----------
def baseline_tfidf(texts, y):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), analyzer="word")
    X = vec.fit_transform(texts)
    results = {}
    for name, clf in [("LogReg", LogisticRegression(max_iter=1000)),
                      ("SVM", LinearSVC())]:
        scores = cross_val_score(clf, X, y, cv=3, scoring="f1_macro")
        results[name] = (scores.mean(), scores.std())
    return results


# ---------- Vocab ----------
def build_vocab(texts, min_count=2):
    c = Counter()
    for t in texts:
        c.update(t.split())
    words = [w for w, n in c.items() if n >= min_count]
    return {w: i + 1 for i, w in enumerate(words)}


# ---------- Dataset ----------
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.max_id = len(vocab)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = []
        for w in self.texts[i].split()[:self.max_len]:
            idx = self.vocab.get(w, 0)
            if idx > self.max_id:
                idx = 0
            ids.append(idx)
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.labels[i])


# ---------- Models ----------
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_classes, pretrained=None, freeze=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        if pretrained is not None:
            self.emb.weight.data.copy_(torch.tensor(pretrained))
            if freeze:
                self.emb.weight.requires_grad = False
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        pooled = out.mean(dim=1)
        return self.fc(pooled)


class GRUModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.gru(e)
        return self.fc(out.mean(dim=1))


class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_filters, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in [3, 4, 5]])
        self.fc = nn.Linear(num_filters * 3, num_classes)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        outs = [torch.relu(c(e)).max(dim=2)[0] for c in self.convs]
        return self.fc(torch.cat(outs, dim=1))


class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        scores = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        ctx = (out * scores.unsqueeze(-1)).sum(dim=1)
        return self.fc(ctx)


# ---------- Training ----------
def train_model(model, train_loader, dev_loader, epochs=5, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dev_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        print(f"epoch {epoch+1}: loss={total_loss/len(train_loader):.3f}, dev_acc={correct/total:.3f}")
    return model


def evaluate(model, loader):
    device = next(model.parameters()).device
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds += model(x).argmax(dim=1).cpu().tolist()
            golds += y.tolist()
    return preds, golds