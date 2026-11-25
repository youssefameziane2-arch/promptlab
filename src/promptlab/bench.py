"""Banc d'évaluation reproductible.

Plan d'expérience : systèmes × documents × longueurs cibles × graines.
* systèmes = méthodes de référence extractives (+ les 5 techniques de prompting
  si un LLM est fourni, chacune avec et sans boucle de vérification/réparation) ;
* chaque mesure est enregistrée (CSV) ; le rapport donne moyennes et IC 95 %,
  et des comparaisons appariées (technique vs zero-shot, réparation vs sans).
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .baselines import BASELINES
from .documents import CorpusDoc, load_corpus
from .llm import LLMClient
from .metrics import evaluate
from .stats import mean_ci, paired_diff
from .summarizer import Summarizer, checks
from .techniques import TECHNIQUES

METRICS = ["key_facts", "rouge_l", "support", "length_accuracy"]


@dataclass
class BenchConfig:
    lengths: list[int] = field(default_factory=lambda: [30, 60])
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    techniques: list[str] = field(default_factory=lambda: list(TECHNIQUES))
    with_ablation: bool = True     # mesurer aussi chaque technique sans boucle de réparation
    tolerance: float = 0.15


def run(client: LLMClient | None, config: BenchConfig, corpus: list[CorpusDoc] | None = None,
        progress=None) -> list[dict]:
    corpus = corpus or load_corpus()
    rows: list[dict] = []
    systems: list[tuple[str, str, bool]] = [(f"baseline:{b}", b, False) for b in BASELINES]
    if client is not None:
        for t in config.techniques:
            systems.append((f"{t}+repair", t, True))
            if config.with_ablation:
                systems.append((t, t, False))
    total = len(systems) * len(corpus) * len(config.lengths) * len(config.seeds)
    done = 0
    for system, key, repair in systems:
        for doc in corpus:
            for n in config.lengths:
                for seed in config.seeds:
                    if system.startswith("baseline:"):
                        if seed != config.seeds[0]:
                            continue  # déterministe : une seule graine suffit
                        summary, attempts, seconds, truncated = BASELINES[key](doc.text, n), 1, 0.0, False
                    else:
                        res = Summarizer(client, tolerance=config.tolerance, repair=repair).summarize(
                            doc.text, key, n, seed)
                        summary, attempts = res.summary, len(res.attempts)
                        seconds, truncated = res.seconds, res.truncated
                    m = evaluate(summary, doc.text, n, doc.reference, doc.facts)
                    passed = all(c.ok for c in checks(summary, doc.text, n, config.tolerance))
                    rows.append({"system": system, "doc": doc.id, "target": n, "seed": seed,
                                 **m, "passed": passed, "attempts": attempts,
                                 "seconds": round(seconds, 3), "truncated": truncated,
                                 "summary": summary})
                    done += 1
                    if progress:
                        progress(done, total, system)
    return rows


def _by_system(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["system"], []).append(r)
    return out


def _paired(rows_a: list[dict], rows_b: list[dict], metric: str) -> dict | None:
    index = {(r["doc"], r["target"], r["seed"]): r[metric] for r in rows_b}
    pairs = [(r[metric], index[k]) for r in rows_a
             if (k := (r["doc"], r["target"], r["seed"])) in index]
    if len(pairs) < 2:
        return None
    a, b = zip(*pairs, strict=True)
    return paired_diff(list(a), list(b))


def summarize_rows(rows: list[dict]) -> dict:
    systems = _by_system(rows)
    table = {}
    for name, rs in systems.items():
        entry = {"n": len(rs)}
        for metric in METRICS:
            entry[metric] = mean_ci([r[metric] for r in rs])
        entry["passed"] = statistics.fmean(r["passed"] for r in rs)
        entry["invented_numbers"] = statistics.fmean(r["invented_numbers"] for r in rs)
        entry["attempts"] = statistics.fmean(r["attempts"] for r in rs)
        entry["seconds"] = statistics.fmean(r["seconds"] for r in rs)
        table[name] = entry
    comparisons = []
    if "zero_shot+repair" in systems:
        for t in TECHNIQUES:
            name = f"{t}+repair"
            if t != "zero_shot" and name in systems:
                for metric in ("key_facts", "rouge_l"):
                    d = _paired(systems[name], systems["zero_shot+repair"], metric)
                    if d:
                        comparisons.append({"a": name, "b": "zero_shot+repair", "metric": metric, **d})
    for t in TECHNIQUES:
        if f"{t}+repair" in systems and t in systems:
            d = _paired(systems[f"{t}+repair"], systems[t], "length_accuracy")
            if d:
                comparisons.append({"a": f"{t}+repair", "b": t, "metric": "length_accuracy", **d})
    return {"table": table, "comparisons": comparisons}


def to_markdown(summary: dict, model: str, config: BenchConfig, n_docs: int) -> str:
    pct = lambda ci: f"{100 * ci[0]:.0f} % [{100 * ci[1]:.0f}–{100 * ci[2]:.0f}]"  # noqa: E731
    lines = [
        "# Résultats du banc",
        "",
        f"Modèle : `{model}` · {n_docs} documents · longueurs {config.lengths} mots · "
        f"graines {config.seeds} · tolérance de longueur ±{int(config.tolerance * 100)} %",
        "",
        "Moyenne et intervalle de confiance à 95 % (bootstrap).",
        "",
        "| Système | Faits clés | ROUGE-L | Fidélité (support) | Respect longueur | Vérifs OK "
        "| Nombres inventés | Tentatives | Temps (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order = sorted(summary["table"], key=lambda s: (not s.startswith("baseline"), s))
    for name in order:
        e = summary["table"][name]
        lines.append(
            f"| `{name}` | {pct(e['key_facts'])} | {pct(e['rouge_l'])} | {pct(e['support'])} "
            f"| {pct(e['length_accuracy'])} | {100 * e['passed']:.0f} % | {e['invented_numbers']:.2f} "
            f"| {e['attempts']:.2f} | {e['seconds']:.1f} |")
    if summary["comparisons"]:
        lines += ["", "## Comparaisons appariées", "",
                  "| A | B | Métrique | Écart A − B | IC 95 % | Significatif |",
                  "|---|---|---|---|---|---|"]
        for c in summary["comparisons"]:
            lines.append(f"| `{c['a']}` | `{c['b']}` | {c['metric']} | {100 * c['diff']:+.1f} pts "
                         f"| [{100 * c['ci_low']:+.1f} ; {100 * c['ci_high']:+.1f}] "
                         f"| {'oui' if c['significant'] else 'non'} |")
    lines += ["", "*Faits clés* : part des faits essentiels du texte présents dans le résumé. "
              "*Fidélité* : part des mots porteurs de sens du résumé présents dans la source. "
              "*Vérifs OK* : longueur, format et absence de nombre inventé respectés.", ""]
    return "\n".join(lines)


def save(rows: list[dict], summary: dict, markdown: str, out: str | Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    (out / "RESULTS.md").write_text(markdown, encoding="utf-8")
    try:
        _chart(summary, out / "chart.png")
    except ImportError:  # matplotlib optionnel
        pass
    return out


def _chart(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(summary["table"], key=lambda s: (not s.startswith("baseline"), s))
    metrics = [("key_facts", "Faits clés"), ("support", "Fidélité"), ("length_accuracy", "Longueur")]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 3.8))
    width = 0.8 / len(metrics)
    for k, (m, label) in enumerate(metrics):
        means = [summary["table"][n][m][0] for n in names]
        err = [[means[i] - summary["table"][n][m][1] for i, n in enumerate(names)],
               [summary["table"][n][m][2] - means[i] for i, n in enumerate(names)]]
        ax.bar([i + k * width for i in range(len(names))], means, width, yerr=err, capsize=2, label=label)
    ax.set_xticks([i + width for i in range(len(names))])
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
