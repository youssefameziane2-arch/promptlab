"""Ligne de commande.

    promptlab summarize doc.pdf -t few_shot -n 60 --model ollama:qwen2.5:3b
    promptlab bench --model ollama:qwen2.5:3b --cache results/llm_cache.json
    promptlab bench --model replay:ollama:qwen2.5:3b --cache results/llm_cache.json   # rejeu
    promptlab bench                                   # méthodes de référence seules (CI)
    promptlab gui
"""
from __future__ import annotations

import argparse
import sys

from . import bench
from .documents import load_corpus, read_document
from .llm import make_client
from .metrics import evaluate
from .summarizer import Summarizer
from .techniques import TECHNIQUES


def cmd_summarize(args) -> int:
    text = read_document(args.file)
    client = make_client(args.model, args.cache)
    res = Summarizer(client, repair=not args.no_repair).summarize(text, args.technique, args.words)
    print(res.summary, "\n")
    for i, a in enumerate(res.attempts, 1):
        print(f"tentative {i}: {', '.join(a.failed) or 'OK'} ({a.seconds:.1f}s)")
    m = evaluate(res.summary, text, args.words)
    print(f"{res.word_count} mots · longueur {100 * m['length_accuracy']:.0f} % · "
          f"fidélité {100 * m['support']:.0f} % · {'tronqué' if res.truncated else ''}")
    return 0


def cmd_bench(args) -> int:
    client = make_client(args.model, args.cache) if args.model else None
    config = bench.BenchConfig(lengths=args.lengths, seeds=list(range(args.seeds)),
                               with_ablation=not args.no_ablation)
    corpus = load_corpus()

    def progress(done, total, system):
        print(f"\r[{done}/{total}] {system:<28}", end="", file=sys.stderr, flush=True)

    rows = bench.run(client, config, corpus, progress)
    print(file=sys.stderr)
    summary = bench.summarize_rows(rows)
    md = bench.to_markdown(summary, args.model or "aucun (méthodes de référence)", config, len(corpus))
    bench.save(rows, summary, md, args.out)
    print(md)
    return 0


def cmd_gui(_args) -> int:  # pragma: no cover
    from .gui import main

    main()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="promptlab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("summarize")
    p.add_argument("file")
    p.add_argument("-t", "--technique", choices=list(TECHNIQUES), default="few_shot")
    p.add_argument("-n", "--words", type=int, default=50)
    p.add_argument("--model", default="ollama:qwen2.5:3b")
    p.add_argument("--cache")
    p.add_argument("--no-repair", action="store_true")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("bench")
    p.add_argument("--model", help="ollama:<m> | openai:<m> | replay:<backend:model>")
    p.add_argument("--cache", help="fichier de cache des réponses (enregistrement / rejeu)")
    p.add_argument("--lengths", type=int, nargs="+", default=[30, 60])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("gui")
    p.set_defaults(func=cmd_gui)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
