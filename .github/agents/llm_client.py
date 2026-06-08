"""
llm_client.py — GitHub Models-first / Ollama-fallback LLM interface.

Primary:  GitHub Models API (https://models.github.ai/inference)
          Authenticated via GITHUB_TOKEN — the token GitHub auto-injects
          into every Actions run. No external API key or billing account needed.
          Requires `models: read` permission in the workflow's permissions block.

Fallback: Local Ollama server (http://localhost:11434)
          Used automatically when running on a self-hosted Mac runner where
          Ollama is installed. Free, no network egress, GPU-accelerated if available.

The GitHub Models API is OpenAI-compatible, so the same payload format works
for both providers. Model IDs differ; see GH_MODEL / OLLAMA_MODEL env vars.
"""
from __future__ import annotations

import json
import os
import sys

import requests

# --------------------------------------------------------------------------- #
# Configuration (override via environment variables)
# --------------------------------------------------------------------------- #
# GitHub Models
GH_TOKEN        = os.environ.get("GITHUB_TOKEN", "")
GH_MODEL        = os.environ.get("GH_MODEL", "gpt-4o-mini")          # cheap + fast
GH_ENDPOINT     = "https://models.github.ai/inference/chat/completions"

# Local Ollama (self-hosted runner fallback)
OLLAMA_URL      = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def _call_github_models(prompt: str, system: str = "") -> str:
    """
    Call GitHub Models API using GITHUB_TOKEN.
    OpenAI-compatible chat completions endpoint.
    Raises on HTTP error or missing token.
    """
    if not GH_TOKEN:
        raise RuntimeError(
            "GITHUB_TOKEN is not set. "
            "Ensure the workflow has 'models: read' permission."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        GH_ENDPOINT,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"model": GH_MODEL, "messages": messages},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_ollama(prompt: str, system: str = "") -> str:
    """
    Call a local Ollama server (self-hosted runner mode).
    Raises on connection error or timeout.
    """
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"]


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
def call(prompt: str, system: str = "", prefer_local: bool = False) -> str:
    """
    Generate a response using the best available provider.

    Args:
        prompt:       User-turn text.
        system:       System / instruction text.
        prefer_local: When True, try Ollama first (self-hosted runner mode).
                      When False (default), use GitHub Models (ubuntu-latest mode).

    Returns:
        Response string from whichever provider succeeds first.
    """
    if prefer_local:
        try:
            return _call_ollama(prompt, system)
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            print(
                f"[llm_client] Ollama unavailable ({exc}); falling back to GitHub Models.",
                file=sys.stderr,
            )

    # GitHub Models (default path — works in every standard Actions runner)
    return _call_github_models(prompt, system)
