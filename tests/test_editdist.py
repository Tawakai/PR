import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nlpkit.editdist import damerau_levenshtein, levenshtein  # noqa: E402


def _apply(a, ops):
    return "".join(o[2] for o in ops if o[0] != "delete")


def test_known():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert damerau_levenshtein("ab", "ba") == 1
    assert levenshtein("ab", "ba") == 2


def test_random_pairs_vs_rapidfuzz():
    from rapidfuzz.distance import OSA, Levenshtein
    rng = random.Random(0)
    for _ in range(1000):
        a = "".join(rng.choice("абвг") for _ in range(rng.randint(0, 10)))
        b = "".join(rng.choice("абвг") for _ in range(rng.randint(0, 10)))
        assert levenshtein(a, b) == Levenshtein.distance(a, b)
        assert damerau_levenshtein(a, b) == OSA.distance(a, b)


def test_ops_reconstruct_target():
    rng = random.Random(1)
    for _ in range(300):
        a = "".join(rng.choice("абв") for _ in range(rng.randint(0, 8)))
        b = "".join(rng.choice("абв") for _ in range(rng.randint(0, 8)))
        for fn in (levenshtein, damerau_levenshtein):
            d, ops = fn(a, b, True)
            assert sum(o[0] != "match" for o in ops) == d
            assert _apply(a, ops) == b