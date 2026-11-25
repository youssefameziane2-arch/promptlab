from promptlab.baselines import BASELINES
from promptlab.documents import load_corpus
from promptlab.metrics import evaluate, key_facts, rouge_l, support
from promptlab.stats import mean_ci, paired_diff
from promptlab.text import count_words


def test_rouge_l_bounds():
    assert rouge_l("le routeur calcule le chemin", "le routeur calcule le chemin") == 1.0
    assert rouge_l("abc", "xyz") == 0.0


def test_support_detects_unsupported_content():
    src = "OSPF calcule les chemins avec Dijkstra dans chaque zone."
    assert support("OSPF calcule les chemins avec Dijkstra.", src) == 1.0
    assert support("BGP échange des préfixes entre systèmes autonomes.", src) < 0.3


def test_key_facts_accepts_alternatives_and_accents():
    facts = [["zone 0", "backbone"], ["dijkstra"], ["routeur désigné", "dr"]]
    assert key_facts("La zone de BACKBONE et l'algorithme de Dijkstra.", facts) == 2 / 3


def test_mean_ci_contains_mean():
    m, lo, hi = mean_ci([0.2, 0.4, 0.6, 0.8])
    assert lo <= m <= hi


def test_paired_bootstrap_detects_a_real_difference():
    a = [0.8, 0.82, 0.79, 0.85, 0.81, 0.83]
    b = [0.6, 0.61, 0.58, 0.66, 0.6, 0.62]
    d = paired_diff(a, b)
    assert d["significant"] and d["p_better"] == 1.0
    assert not paired_diff(a, a)["significant"]


def test_corpus_and_baselines():
    corpus = load_corpus()
    assert len(corpus) == 6
    for doc in corpus:
        for fn in BASELINES.values():
            summary = fn(doc.text, 60)
            assert 25 <= count_words(summary) <= 70
            assert evaluate(summary, doc.text, 60)["support"] == 1.0  # extractif = fidèle
