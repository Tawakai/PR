"""Расстояния Левенштейна и Дамерау–Левенштейна (OSA) с восстановлением операций (ПР 2)."""
from typing import Callable

INF = float("inf")


def _matrix(a: str, b: str, sub: Callable, ins: float, dele: float, trans: float):
    n, m = len(a), len(b)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i * dele
    for j in range(1, m + 1):
        d[0][j] = j * ins
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0.0 if a[i - 1] == b[j - 1] else sub(a[i - 1], b[j - 1])
            best = min(d[i - 1][j] + dele, d[i][j - 1] + ins, d[i - 1][j - 1] + c)
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]
                    and a[i - 1] != b[j - 1]):
                best = min(best, d[i - 2][j - 2] + trans)
            d[i][j] = best
    return d


def _backtrace(a, b, d, sub, ins, dele, trans):
    i, j, out = len(a), len(b), []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = a[i - 1] == b[j - 1]
            c = 0.0 if same else sub(a[i - 1], b[j - 1])
            if d[i][j] == d[i - 1][j - 1] + c:
                out.append(("match" if same else "substitute", a[i - 1], b[j - 1]))
                i, j = i - 1, j - 1
                continue
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]
                    and a[i - 1] != b[j - 1] and d[i][j] == d[i - 2][j - 2] + trans):
                out.append(("transpose", a[i - 2:i], b[j - 2:j]))
                i, j = i - 2, j - 2
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + dele:
            out.append(("delete", a[i - 1], ""))
            i -= 1
        else:
            out.append(("insert", "", b[j - 1]))
            j -= 1
    return out[::-1]


def _unit(x, y):
    return 1.0


def levenshtein(a: str, b: str, ops: bool = False):
    """Расстояние Левенштейна. При ops=True возвращает (расстояние, список операций)."""
    d = _matrix(a, b, _unit, 1.0, 1.0, INF)
    dist = int(d[-1][-1])
    return (dist, _backtrace(a, b, d, _unit, 1.0, 1.0, INF)) if ops else dist


def damerau_levenshtein(a: str, b: str, ops: bool = False):
    """Дамерау–Левенштейн (вариант OSA: + перестановка соседних символов)."""
    d = _matrix(a, b, _unit, 1.0, 1.0, 1.0)
    dist = int(d[-1][-1])
    return (dist, _backtrace(a, b, d, _unit, 1.0, 1.0, 1.0)) if ops else dist


def weighted_osa(a: str, b: str, sub: Callable = _unit, ins: float = 1.0,
                 dele: float = 1.0, trans: float = 1.0) -> float:
    """OSA с весами операций (для взвешенной модели ошибок)."""
    return _matrix(a, b, sub, ins, dele, trans)[-1][-1]