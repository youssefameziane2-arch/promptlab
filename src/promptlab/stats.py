"""Statistiques pour comparer honnêtement des techniques sur un petit corpus.

Avec quelques documents, une différence de moyenne peut être due au hasard.
On rapporte donc un intervalle de confiance (bootstrap) et, pour comparer deux
techniques, un bootstrap **apparié** : on rééchantillonne les mêmes couples
(document, longueur, graine) pour les deux techniques.
"""
from __future__ import annotations

import random
import statistics


def mean_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
            seed: int = 7) -> tuple[float, float, float]:
    if not values:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    m = statistics.fmean(values)
    if len(values) == 1:
        return m, m, m
    boots = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return m, lo, hi


def paired_diff(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 7) -> dict:
    """Différence moyenne a − b, IC 95 % et part des rééchantillons où a > b."""
    assert len(a) == len(b) and a, "séries appariées de même longueur requises"
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    return {
        "diff": statistics.fmean(diffs),
        "ci_low": boots[int(0.025 * n_boot)],
        "ci_high": boots[int(0.975 * n_boot) - 1],
        "p_better": sum(x > 0 for x in boots) / n_boot,
        "significant": boots[int(0.025 * n_boot)] > 0 or boots[int(0.975 * n_boot) - 1] < 0,
    }
