"""Chargement des documents (TXT, Markdown, DOCX, PDF) et du corpus d'évaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"


def read_document(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        from docx import Document  # python-docx

        return "\n".join(p.text for p in Document(path).paragraphs if p.text.strip())
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    raise ValueError(f"format non pris en charge : {suffix}")


@dataclass
class CorpusDoc:
    id: str
    text: str
    reference: str
    facts: list[list[str]]


def load_corpus(data_dir: str | Path = DATA) -> list[CorpusDoc]:
    data_dir = Path(data_dir)
    refs = json.loads((data_dir / "references.json").read_text(encoding="utf-8"))
    docs = []
    for path in sorted((data_dir / "docs").glob("*.txt")):
        ref = refs[path.stem]
        docs.append(CorpusDoc(path.stem, path.read_text(encoding="utf-8").strip(),
                              ref["summary"], ref["key_facts"]))
    return docs
