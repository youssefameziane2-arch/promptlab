import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from promptlab.llm import ScriptedClient  # noqa: E402
from promptlab.text import words  # noqa: E402


def n_words_from_prompt(prompt: str) -> int:
    import re

    return int(re.findall(r"en (\d+) mots", prompt)[-1])


def faithful_responder(system, prompt, seed):
    """Faux LLM : recopie le début du texte source à la bonne longueur."""
    n = n_words_from_prompt(prompt)
    source = prompt.split("Texte :\n")[-1].split("\nRésumé :")[0]
    return " ".join(words(source)[: n - 1]) + "."


@pytest.fixture
def faithful():
    return ScriptedClient(faithful_responder)
