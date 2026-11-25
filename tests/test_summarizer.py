from promptlab.llm import ScriptedClient
from promptlab.summarizer import Summarizer, checks, clean, truncate
from promptlab.techniques import TECHNIQUES
from promptlab.text import count_words

SOURCE = ("Le protocole MQTT utilise un broker. Les clients publient sur des sujets. "
          "Trois niveaux de qualité de service existent. Le port standard est 1883. ") * 3


def test_clean_removes_chatty_wrapping():
    raw = ("<think>je réfléchis</think>**Voici le résumé du texte :** mqtt relie des clients "
           "via un broker.\n\n1. Question : quel est le port ?")
    assert clean(raw) == "Mqtt relie des clients via un broker."


def test_checks_flag_invented_numbers_and_length():
    failed = {c.name for c in checks("MQTT utilise le port 8080.", SOURCE, 20, 0.15) if not c.ok}
    assert failed == {"length", "no_invented_numbers"}


def test_repair_loop_feeds_back_the_exact_problem():
    answers = iter(["mot " * 80 + "fin.", "MQTT relie des clients à un broker sur le port 1883 "
                    "avec trois niveaux de qualité de service garantis."])
    client = ScriptedClient(lambda s, p, seed: next(answers))
    res = Summarizer(client, tolerance=0.3).summarize(SOURCE, "zero_shot", 18)
    assert res.passed and len(res.attempts) == 2
    assert res.attempts[0].failed == ["length"]
    assert "81 mots au lieu de 18" in client.prompts[1]  # retour précis, pas une simple relance


def test_fallback_truncates_at_sentence_boundary():
    long = "Première phrase assez longue pour compter. " * 10
    client = ScriptedClient(lambda s, p, seed: long)
    res = Summarizer(client, max_attempts=2).summarize(SOURCE, "few_shot", 20)
    assert res.truncated and res.summary.endswith(".")
    assert count_words(res.summary) <= 23
    assert len(res.attempts) == 2


def test_without_repair_a_single_attempt_is_made():
    client = ScriptedClient(lambda s, p, seed: "trop court.")
    res = Summarizer(client, repair=False).summarize(SOURCE, "role", 40)
    assert len(res.attempts) == 1 and not res.passed


def test_truncate_never_cuts_mid_word():
    assert truncate("Un deux trois quatre cinq six sept huit neuf dix onze", 5).endswith("…")


def test_every_technique_builds_a_distinct_prompt():
    pairs = {(TECHNIQUES[k].system(), TECHNIQUES[k].prompt("TEXTE", 50)) for k in TECHNIQUES}
    assert len(pairs) == 5  # role prompting diffère par le message système
    assert all("50 mots" in p and "TEXTE" in p for _, p in pairs)
    assert "Exemples" in TECHNIQUES["few_shot"].prompt("x", 50)
    assert "ingénieur" in TECHNIQUES["role"].system()


def test_fallback_prefers_truncated_first_attempt_over_worse_rewrites():
    """Régression trouvée par le banc : la réécriture trop courte ne doit pas l'emporter."""
    long_ok = "Le protocole MQTT relie des clients à un broker central. " * 4
    answers = iter([long_ok, "MQTT broker.", "MQTT."])
    client = ScriptedClient(lambda s, p, seed: next(answers))
    res = Summarizer(client, tolerance=0.15).summarize(SOURCE, "zero_shot", 20)
    assert res.truncated and res.passed
    assert res.summary.startswith("Le protocole MQTT")
