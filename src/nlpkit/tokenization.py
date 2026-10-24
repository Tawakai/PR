"""Сравнение токенизаторов: пробельный, regex, nltk, spaCy, BPE, SentencePiece (ПР 2)."""
import csv
import random
import re

from nlpkit.cleaning import clean_text


def load_split(cfg: dict, lang: str):
    d = cfg["paths"]["clean"] if lang == "ru" else cfg["paths"]["raw"].parent / "raw_kk"
    paras = []
    for p in sorted(d.glob("*.txt")):
        for para in clean_text(p.read_text(encoding="utf-8")).split("\n"):
            if len(para) > 20:
                paras.append(para)
    random.Random(cfg["project"]["seed"]).shuffle(paras)
    cut = int(len(paras) * 0.9)
    return paras[:cut], paras[cut:]


def _word_level(fn):
    def run(train, test):
        tr = [t.lower() for p in train for t in fn(p)]
        te = [t.lower() for p in test for t in fn(p)]
        vocab = set(tr)
        oov = sum(t not in vocab for t in te) / len(te)
        return len(vocab), oov, len(te)
    return run


def _subword_bpe(vocab_size):
    def run(train, test):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.train_from_iterator(
            [p.lower() for p in train],
            trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"]))
        enc = [tok.encode(p.lower()).tokens for p in test]
        n = sum(len(e) for e in enc)
        unk = sum(t == "[UNK]" for e in enc for t in e)
        return tok.get_vocab_size(), unk / n, n
    return run


def _subword_spm(vocab_size, out_dir, lang):
    def run(train, test):
        import sentencepiece as spm
        f = out_dir / f"spm_train_{lang}.txt"
        f.write_text("\n".join(p.lower() for p in train), encoding="utf-8")
        prefix = str(out_dir / f"spm_{lang}")
        spm.SentencePieceTrainer.train(input=str(f), model_prefix=prefix, vocab_size=vocab_size,
                                       hard_vocab_limit=False, character_coverage=1.0)
        sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
        ids = [i for p in test for i in sp.encode(p.lower())]
        unk = sum(i == sp.unk_id() for i in ids)
        return sp.get_piece_size(), unk / len(ids), len(ids)
    return run


def compare(cfg: dict) -> None:
    out = cfg["paths"]["results"] / "pr2"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang in ("ru", "kk"):
        train, test = load_split(cfg, lang)
        if not test:
            print(f"[{lang}] нет текста, пропуск (для kk: python run.py download-kk)")
            continue
        n_words = sum(len(p.split()) for p in test)
        vs = cfg["pr2"]["vocab_size"][lang]

        import nltk
        import spacy
        try:
            sp_tok = spacy.blank(lang).tokenizer
        except Exception:
            sp_tok = spacy.blank("xx").tokenizer
        methods = {
            "пробельный": _word_level(str.split),
            "regex": _word_level(lambda s: re.findall(r"\w+|[^\w\s]", s)),
            "nltk.word_tokenize": _word_level(lambda s: nltk.word_tokenize(s, language="russian")),
            "spaCy": _word_level(lambda s: [t.text for t in sp_tok(s)]),
            f"BPE({vs})": _subword_bpe(vs),
            f"SentencePiece({vs})": _subword_spm(vs, out, lang),
        }
        for name, fn in methods.items():
            try:
                vocab, oov, n_tok = fn(train, test)
            except Exception as e:
                print(f"[{lang}] {name}: ошибка {e}")
                continue
            rows.append([lang, name, vocab, round(oov * 100, 2), round(n_tok / n_words, 3)])
            print(f"[{lang}] {name:22s} словарь={vocab:6d}  OOV={oov * 100:5.2f}%  токенов/слово={n_tok / n_words:.3f}")

    with open(out / "tokenizers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["язык", "токенизатор", "размер словаря", "OOV, %", "токенов на слово"])
        w.writerows(rows)