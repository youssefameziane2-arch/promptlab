"""Interface Qt (PySide6).

La génération tourne dans un ``QThread`` : l'interface reste réactive et reçoit
progression et résultats par signaux (programmation événementielle Qt).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .documents import read_document
from .llm import LLMClient, make_client
from .metrics import evaluate
from .summarizer import Summarizer
from .techniques import TECHNIQUES


class Worker(QObject):
    progress = Signal(str)
    result = Signal(object, dict)      # SummaryResult, métriques
    finished = Signal()
    failed = Signal(str)

    def __init__(self, client: LLMClient, text: str, techniques: list[str], n_words: int,
                 repair: bool) -> None:
        super().__init__()
        self.client, self.text, self.techniques = client, text, techniques
        self.n_words, self.repair = n_words, repair

    def run(self) -> None:
        try:
            summarizer = Summarizer(self.client, repair=self.repair)
            for t in self.techniques:
                self.progress.emit(f"{TECHNIQUES[t].label} en cours…")
                res = summarizer.summarize(self.text, t, self.n_words)
                self.result.emit(res, evaluate(res.summary, self.text, self.n_words))
        except Exception as exc:  # erreur réseau, modèle absent…
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    COLUMNS = ["Technique", "Mots", "Longueur", "Fidélité", "Tentatives", "Temps (s)", "Vérifs"]

    def __init__(self, client: LLMClient | None = None) -> None:
        super().__init__()
        self.setWindowTitle("PromptLab — résumé de documents techniques")
        self.client_override = client
        self._thread: QThread | None = None

        # --- panneau gauche : source et réglages
        self.source = QPlainTextEdit(objectName="source")
        self.source.setPlaceholderText("Collez un texte ou ouvrez un fichier…")
        self.open_btn = QPushButton("Ouvrir… (txt, docx, pdf)", objectName="openBtn")
        self.model = QLineEdit("ollama:qwen2.5:3b", objectName="model")
        self.technique = QComboBox(objectName="technique")
        for key, t in TECHNIQUES.items():
            self.technique.addItem(t.label, key)
        self.technique.setCurrentIndex(list(TECHNIQUES).index("few_shot"))
        self.n_words = QSpinBox(objectName="nWords", minimum=10, maximum=300, value=50)
        self.repair = QCheckBox("Vérifier et réparer la sortie", objectName="repair", checked=True)
        self.run_btn = QPushButton("Résumer", objectName="runBtn")
        self.compare_btn = QPushButton("Comparer les 5 techniques", objectName="compareBtn")
        form = QFormLayout()
        form.addRow("Modèle", self.model)
        form.addRow("Technique", self.technique)
        form.addRow("Longueur (mots)", self.n_words)
        form.addRow("", self.repair)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(self.source, 1)
        lv.addWidget(self.open_btn)
        lv.addLayout(form)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.compare_btn)
        lv.addLayout(buttons)

        # --- panneau droit : résultat, tentatives, comparaison
        self.summary = QPlainTextEdit(objectName="summary", readOnly=True)
        self.attempts = QListWidget(objectName="attempts")
        self.table = QTableWidget(0, len(self.COLUMNS), objectName="table")
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.status = QLabel("Prêt", objectName="status")
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel("Résumé"))
        rv.addWidget(self.summary, 2)
        rv.addWidget(QLabel("Tentatives (vérifications échouées)"))
        rv.addWidget(self.attempts, 1)
        rv.addWidget(QLabel("Comparaison"))
        rv.addWidget(self.table, 2)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([520, 680])
        central = QWidget()
        cv = QVBoxLayout(central)
        cv.addWidget(splitter)
        cv.addWidget(self.status)
        self.setCentralWidget(central)

        self.open_btn.clicked.connect(self._open)
        self.run_btn.clicked.connect(lambda: self._start([self.technique.currentData()]))
        self.compare_btn.clicked.connect(lambda: self._start(list(TECHNIQUES)))

    # ------------------------------------------------------------------ actions
    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir un document", "",
                                              "Documents (*.txt *.md *.docx *.pdf)")
        if path:
            self.source.setPlainText(read_document(path))
            self.status.setText(f"Chargé : {Path(path).name}")

    def _client(self) -> LLMClient:
        return self.client_override or make_client(self.model.text().strip())

    def _start(self, techniques: list[str]) -> None:
        text = self.source.toPlainText().strip()
        if not text:
            self.status.setText("Aucun texte à résumer")
            return
        self.table.setRowCount(0)
        self.attempts.clear()
        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = Worker(self._client(), text, techniques, self.n_words.value(),
                              self.repair.isChecked())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.result.connect(self._show)
        # Slots = méthodes de la fenêtre : exécutés dans le thread de l'interface (connexion en file).
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _on_failed(self, message: str) -> None:
        self.status.setText(f"Erreur : {message}")

    def _on_finished(self) -> None:
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        for w in (self.run_btn, self.compare_btn, self.open_btn):
            w.setEnabled(not busy)
        if not busy and self.status.text().endswith("…"):
            self.status.setText("Terminé")

    def _show(self, res, metrics: dict) -> None:
        self.summary.setPlainText(res.summary)
        for i, a in enumerate(res.attempts, 1):
            issues = ", ".join(a.failed) or "OK"
            self.attempts.addItem(f"[{TECHNIQUES[res.technique].label}] tentative {i} : {issues}")
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [TECHNIQUES[res.technique].label, str(res.word_count),
                  f"{100 * metrics['length_accuracy']:.0f} %", f"{100 * metrics['support']:.0f} %",
                  str(len(res.attempts)), f"{res.seconds:.1f}",
                  "OK" if res.passed else ("tronqué" if res.truncated else "à revoir")]
        for col, v in enumerate(values):
            self.table.setItem(row, col, QTableWidgetItem(v))


def main() -> None:  # pragma: no cover
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 760)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
