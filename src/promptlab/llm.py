"""Clients LLM : Ollama (local), API compatible OpenAI, rejeu et client scripté.

Chaque réponse réelle peut être mise en cache (``cache_path``) puis rejouée :
le banc d'évaluation devient reproductible et exécutable en CI sans GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Completion:
    text: str
    seconds: float
    cached: bool = False


def cache_key(model: str, system: str, prompt: str, seed: int) -> str:
    raw = "\x00".join([model, system, prompt, str(seed)])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


class _Cache:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.data: dict[str, dict] = {}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, value: dict) -> None:
        if not self.path:
            return
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


class LLMClient:
    """Base : gère le cache ; les sous-classes implémentent ``_call``."""

    name = "llm"
    model = ""

    def __init__(self, cache_path: str | Path | None = None, temperature: float = 0.2) -> None:
        self.cache = _Cache(cache_path)
        self.temperature = temperature
        self.calls = 0

    def complete(self, system: str, prompt: str, seed: int = 0) -> Completion:
        key = cache_key(self.name, system, prompt, seed)
        hit = self.cache.get(key)
        if hit is not None:
            return Completion(hit["text"], hit["seconds"], cached=True)
        self.calls += 1
        start = time.perf_counter()
        text = self._call(system, prompt, seed)
        seconds = time.perf_counter() - start
        self.cache.put(key, {"text": text, "seconds": round(seconds, 3)})
        return Completion(text, seconds)

    def _call(self, system: str, prompt: str, seed: int) -> str:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _post(url: str, payload: dict, headers: dict | None = None, timeout: int = 300) -> dict:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read())


class OllamaClient(LLMClient):
    def __init__(self, model: str = "qwen2.5:3b", host: str | None = None, **kw) -> None:
        super().__init__(**kw)
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.name = f"ollama:{model}"

    def _call(self, system: str, prompt: str, seed: int) -> str:
        data = self._post(f"{self.host}/api/chat", {
            "model": self.model, "stream": False,
            "options": {"temperature": self.temperature, "seed": seed},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}]})
        return data["message"]["content"]


class OpenAICompatClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", **kw) -> None:
        super().__init__(**kw)
        self.model = model
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.name = f"openai:{model}"

    def _call(self, system: str, prompt: str, seed: int) -> str:
        data = self._post(f"{self.base}/chat/completions", {
            "model": self.model, "temperature": self.temperature, "seed": seed,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}]},
            headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
        return data["choices"][0]["message"]["content"]


class ReplayClient(LLMClient):
    """Rejoue un cache existant ; toute question absente du cache est une erreur."""

    def __init__(self, cache_path: str | Path, model_name: str) -> None:
        super().__init__(cache_path)
        self.name = model_name

    def _call(self, system: str, prompt: str, seed: int) -> str:
        raise KeyError("prompt absent du cache : relancer le banc avec le vrai modèle")


class ScriptedClient(LLMClient):
    """Client de test : ``responder(system, prompt, seed) -> str``."""

    def __init__(self, responder, name: str = "scripted") -> None:
        super().__init__(None)
        self.responder = responder
        self.name = name
        self.prompts: list[str] = []

    def _call(self, system: str, prompt: str, seed: int) -> str:
        self.prompts.append(prompt)
        return self.responder(system, prompt, seed)


def make_client(spec: str, cache_path: str | Path | None = None) -> LLMClient:
    backend, _, model = spec.partition(":")
    if backend == "ollama":
        return OllamaClient(model or "qwen2.5:3b", cache_path=cache_path)
    if backend == "openai":
        return OpenAICompatClient(model or "gpt-4o-mini", cache_path=cache_path)
    if backend == "replay":
        if not cache_path:
            raise ValueError("replay needs --cache")
        return ReplayClient(cache_path, model)
    raise ValueError(f"unknown backend {spec!r}")
