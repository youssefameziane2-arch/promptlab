"""Méthodes de référence sans LLM (extractives).

Comparer un LLM à « rien » ne prouve pas grand-chose ; le comparer à des
méthodes simples et déterministes dit si le LLM apporte vraiment quelque chose.

* ``lead``       : les premières phrases du texte, jusqu'à la longueur cible ;
* ``centrality`` : sélection des phrases les plus centrales (graphe de similarité
                   façon TextRank), remises dans l'ordre du texte.
"""
from __future__ import annotations

from .text import content_stems, count_words, sentences


def _fit(selected: list[str], n_words: int) -> str:
    out, total = [], 0
    for s in selected:
        c = count_words(s)
        if out and total + c > n_words * 1.1:
            break
        out.append(s)
        total += c
    return " ".join(out)


def lead(text: str, n_words: int) -> str:
    return _fit(sentences(text), n_words)


def centrality(text: str, n_words: int, iterations: int = 30, damping: float = 0.85) -> str:
    sents = sentences(text)
    bags = [set(content_stems(s)) for s in sents]
    n = len(sents)
    if n <= 1:
        return text
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and bags[i] and bags[j]:
                sim[i][j] = len(bags[i] & bags[j]) / (len(bags[i]) + len(bags[j])) ** 0.5
    score = [1.0] * n
    for _ in range(iterations):
        score = [(1 - damping) + damping * sum(
            sim[j][i] / (sum(sim[j]) or 1) * score[j] for j in range(n)) for i in range(n)]
    ranked = sorted(range(n), key=lambda i: -score[i])
    chosen, total = [], 0
    for i in ranked:
        c = count_words(sents[i])
        if chosen and total + c > n_words * 1.1:
            continue
        chosen.append(i)
        total += c
        if total >= n_words * 0.9:
            break
    return " ".join(sents[i] for i in sorted(chosen))


BASELINES = {"lead": lead, "centrality": centrality}
