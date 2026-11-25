"""Métriques d'évaluation d'un résumé.

* ``length_accuracy`` : respect de la longueur demandée (métrique du projet initial) ;
* ``rouge_l``         : recouvrement avec un résumé de référence écrit à la main ;
* ``key_facts``       : part des faits essentiels du texte présents dans le résumé ;
* ``support``         : part des mots porteurs de sens du résumé présents dans la
                        source — un indicateur simple de **fidélité** (un résumé qui
                        invente du contenu obtient un score bas) ;
* ``invented_numbers``: nombres du résumé absents de la source (hallucination nette).
"""
from __future__ import annotations

from .text import content_stems, fold, numbers, words


def length_accuracy(n_words: int, target: int) -> float:
    return max(0.0, 1.0 - abs(n_words - target) / target)


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> float:
    c = [fold(w) for w in words(candidate)]
    r = [fold(w) for w in words(reference)]
    if not c or not r:
        return 0.0
    lcs = _lcs(c, r)
    p, rec = lcs / len(c), lcs / len(r)
    return 0.0 if lcs == 0 else 2 * p * rec / (p + rec)


def key_facts(summary: str, facts: list[list[str]]) -> float:
    """Chaque fait est une liste d'alternatives ; il est couvert si l'une d'elles apparaît."""
    if not facts:
        return 0.0
    text = fold(summary)
    hit = sum(any(fold(alt) in text for alt in alternatives) for alternatives in facts)
    return hit / len(facts)


def support(summary: str, source: str) -> float:
    s = content_stems(summary)
    if not s:
        return 0.0
    src = set(content_stems(source))
    return sum(t in src for t in s) / len(s)


def invented_numbers(summary: str, source: str) -> int:
    return len(numbers(summary) - numbers(source))


def evaluate(summary: str, source: str, target: int, reference: str | None = None,
             facts: list[list[str]] | None = None) -> dict[str, float]:
    n = len(words(summary))
    out = {
        "words": n,
        "length_accuracy": length_accuracy(n, target),
        "support": support(summary, source),
        "invented_numbers": invented_numbers(summary, source),
    }
    if reference:
        out["rouge_l"] = rouge_l(summary, reference)
    if facts:
        out["key_facts"] = key_facts(summary, facts)
    return out
