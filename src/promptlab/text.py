"""Outils texte partagés : mots, phrases, mots porteurs de sens (français)."""
from __future__ import annotations

import re
import unicodedata

WORD = re.compile(r"[\wÀ-ÿ'’-]+", re.UNICODE)
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")

STOPWORDS = set("""
a à au aux avec ce ces cet cette ci comme dans de des du elle elles en entre est et été être
il ils la le les leur leurs lors mais même ne ni nous on ou où par pas peu plus pour qu que
qui sa sans se ses si son sont sous sur ta te tes toi ton tous tout toute toutes très tu un une
vos votre vous y ainsi afin alors aussi autre autres avant après chaque d l s n c j m qu' d'
dont donc encore enfin fait font lorsque ont peut plusieurs souvent tandis car celle celui ceux
cela ceci elles-mêmes lui eux the of and to in is for on
""".split())


def words(text: str) -> list[str]:
    return WORD.findall(text)


def count_words(text: str) -> int:
    return len(words(text))


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def fold(text: str) -> str:
    """Minuscules sans accents : comparaison robuste."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn").replace("’", "'")


def stem(token: str) -> str:
    """Racinisation volontairement simple (préfixe) : « routeurs » ~ « routeur »."""
    token = fold(token).strip("'-")
    token = token.split("'")[-1]
    return token[:6] if len(token) > 6 else token


def content_stems(text: str) -> list[str]:
    out = []
    for w in words(text):
        f = fold(w).split("'")[-1]
        if len(f) > 3 and f not in STOPWORDS:
            out.append(stem(f))
    return out


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))
