import json

from conftest import faithful_responder

from promptlab import bench
from promptlab.documents import load_corpus, read_document
from promptlab.llm import ReplayClient, ScriptedClient
from promptlab.techniques import TECHNIQUES


def test_bench_runs_end_to_end_and_writes_reports(tmp_path, faithful):
    config = bench.BenchConfig(lengths=[30], seeds=[0], techniques=["zero_shot", "few_shot"])
    corpus = load_corpus()[:3]
    rows = bench.run(faithful, config, corpus)
    systems = {r["system"] for r in rows}
    assert {"baseline:lead", "zero_shot+repair", "few_shot", "few_shot+repair"} <= systems
    summary = bench.summarize_rows(rows)
    assert any(c["metric"] == "key_facts" for c in summary["comparisons"])
    md = bench.to_markdown(summary, "scripted", config, len(corpus))
    out = bench.save(rows, summary, md, tmp_path)
    assert (out / "runs.csv").exists() and (out / "chart.png").exists()
    assert "Comparaisons appariées" in md


def test_cache_makes_runs_replayable(tmp_path):
    cache = tmp_path / "cache.json"
    live = ScriptedClient(faithful_responder, name="ollama:demo")
    live.cache.path = cache
    first = live.complete("sys", "Résume le texte suivant en 12 mots environ.\n\nTexte :\nA B C D", 0)
    replay = ReplayClient(cache, "ollama:demo")
    again = replay.complete("sys", "Résume le texte suivant en 12 mots environ.\n\nTexte :\nA B C D", 0)
    assert again.text == first.text and again.cached
    assert json.loads(cache.read_text())


def test_docx_is_readable(tmp_path):
    from docx import Document

    path = tmp_path / "note.docx"
    doc = Document()
    doc.add_paragraph("Le protocole AXI utilise cinq canaux.")
    doc.save(path)
    assert "cinq canaux" in read_document(path)


def test_gui_compares_all_techniques_without_freezing(qtbot, faithful):
    from promptlab.gui import MainWindow

    win = MainWindow(client=faithful)
    qtbot.addWidget(win)
    win.source.setPlainText(load_corpus()[0].text)
    win.n_words.setValue(30)
    win.compare_btn.click()
    assert not win.compare_btn.isEnabled()  # traitement en arrière-plan
    qtbot.waitUntil(lambda: win.table.rowCount() == len(TECHNIQUES), timeout=5000)
    qtbot.waitUntil(win.compare_btn.isEnabled, timeout=5000)
    assert win.summary.toPlainText()
    assert win.table.item(0, 6).text() == "OK"
    assert win.status.text() == "Terminé"
