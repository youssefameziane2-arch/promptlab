"""Génération contrôlée : produire → nettoyer → vérifier → réparer.

La sortie d'un LLM est traitée comme celle de n'importe quel composant logiciel :
elle passe des **vérifications** explicites (longueur, format, chiffres inventés).
En cas d'échec, le modèle reçoit un retour *précis* sur le défaut constaté et
réécrit sa réponse (au plus ``max_attempts`` fois). En dernier recours, une
troncature propre (fin de phrase) garantit une sortie utilisable, et le
résultat est marqué comme tel.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLMClient
from .techniques import TECHNIQUES
from .text import count_words, numbers, sentences, words

PREFIXES = re.compile(
    r"^\s*(?:\*\*)?(?:voici\s+(?:le|un)\s+résumé[^:]*|résumé(?:\s+du\s+texte)?|summary|synthèse)"
    r"\s*[:：]\s*(?:\*\*)?\s*", re.IGNORECASE)
CUT_MARKERS = re.compile(
    r"\n\s*(?:#{1,6}\s|\d+\.\s|[-*•]\s|question|exercice|note\s*:|remarque\s*:|follow-up)",
    re.IGNORECASE)


def clean(raw: str) -> str:
    text = raw.strip().strip("`").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # modèles « raisonneurs »
    m = CUT_MARKERS.search("\n" + text)
    if m and m.start() > 0:
        text = text[: m.start() - 1]
    text = PREFIXES.sub("", text)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"\s+", " ", text).strip().strip('"«» ')
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def checks(summary: str, source: str, n_words: int, tolerance: float) -> list[Check]:
    n = count_words(summary)
    low, high = round(n_words * (1 - tolerance)), round(n_words * (1 + tolerance))
    invented = sorted(numbers(summary) - numbers(source))
    return [
        Check("non_empty", n > 0, "résumé vide"),
        Check("length", low <= n <= high, f"{n} mots au lieu de {n_words} (tolérance {low}-{high})"),
        Check("single_paragraph", "\n" not in summary and not re.search(r"(^|\s)[-*•]\s", summary),
              "le résumé doit être un paragraphe, sans liste"),
        Check("no_invented_numbers", not invented,
              f"nombres absents du texte source : {', '.join(invented)}"),
        Check("ends_cleanly", summary.rstrip().endswith((".", "!", "?", "…", "»")),
              "le résumé doit se terminer par une phrase complète"),
    ]


def repair_prompt(previous: str, failed: list[Check], n_words: int) -> str:
    issues = "\n".join(f"- {c.detail}" for c in failed)
    return (f"Ton résumé précédent ne respecte pas les consignes :\n{issues}\n\n"
            f"Résumé précédent :\n{previous}\n\n"
            f"Réécris-le en {n_words} mots, en corrigeant ces points, sans rien ajouter "
            "qui ne soit dans le texte. Réponds uniquement avec le résumé.")


def truncate(summary: str, n_words: int) -> str:
    """Coupe à la dernière phrase complète qui tient dans la cible (sinon au mot)."""
    kept, total = [], 0
    for s in sentences(summary):
        c = count_words(s)
        if total + c > n_words and kept:
            break
        kept.append(s)
        total += c
    text = " ".join(kept)
    if count_words(text) > n_words * 1.15:
        text = " ".join(words(text)[:n_words]) + "…"
    return text


@dataclass
class Attempt:
    text: str
    failed: list[str]
    seconds: float


@dataclass
class SummaryResult:
    technique: str
    summary: str
    n_words_target: int
    attempts: list[Attempt] = field(default_factory=list)
    truncated: bool = False
    passed: bool = False

    @property
    def seconds(self) -> float:
        return sum(a.seconds for a in self.attempts)

    @property
    def word_count(self) -> int:
        return count_words(self.summary)


class Summarizer:
    def __init__(self, client: LLMClient, max_attempts: int = 3, tolerance: float = 0.15,
                 repair: bool = True) -> None:
        self.client = client
        self.max_attempts = max_attempts
        self.tolerance = tolerance
        self.repair = repair

    def summarize(self, text: str, technique: str = "few_shot", n_words: int = 50,
                  seed: int = 0) -> SummaryResult:
        tech = TECHNIQUES[technique]
        system, prompt = tech.system(), tech.prompt(text, n_words)
        result = SummaryResult(technique, "", n_words)
        attempts = self.max_attempts if self.repair else 1
        summary = ""
        for i in range(attempts):
            completion = self.client.complete(system, prompt, seed=seed + 1000 * i)
            summary = clean(completion.text)
            failed = [c for c in checks(summary, text, n_words, self.tolerance) if not c.ok]
            result.attempts.append(Attempt(summary, [c.name for c in failed], completion.seconds))
            if not failed:
                result.summary, result.passed = summary, True
                return result
            prompt = (tech.prompt(text, n_words) + "\n\n" +
                      repair_prompt(summary, failed, n_words))
        # Repli : chaque tentative est candidate, telle quelle ou tronquée proprement si elle
        # est trop longue ; on garde celle qui échoue le moins de vérifications, puis la plus
        # proche de la longueur cible, puis la plus ancienne. (Le banc a montré qu'une
        # réécriture « réparée » trop courte pouvait être pire que la 1re tentative tronquée.)
        candidates = []
        for rank, a in enumerate(result.attempts):
            options = [(a.text, False)]
            if count_words(a.text) > n_words * (1 + self.tolerance):
                options.append((truncate(a.text, n_words), True))
            for summary_text, was_truncated in options:
                n_failed = sum(not c.ok for c in checks(summary_text, text, n_words, self.tolerance))
                candidates.append(((n_failed, abs(count_words(summary_text) - n_words), rank),
                                   summary_text, was_truncated))
        _, final, result.truncated = min(candidates, key=lambda c: c[0])
        result.summary = final
        result.passed = all(c.ok for c in checks(final, text, n_words, self.tolerance))
        return result
