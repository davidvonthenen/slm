# Copyright (c) 2026 David vonThenen. All rights reserved.
# Restricted distribution. Unauthorized copying or hosting of this file via any medium
# is strictly prohibited. Proprietary and confidential.

#!/usr/bin/env python3

"""OpenAI-compatible client for the hybrid RAG REST endpoint."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from openai import OpenAI

def _use_openai() -> bool:
    """Check if the OpenAI-compatible endpoint should be used."""
    use_openai = os.getenv("USE_OPENAI", "False")
    return use_openai == "True" or use_openai == "true" or use_openai == "TRUE" or use_openai == "1"

def _resolve_base_url() -> str:
    """Resolve the OpenAI-compatible base URL for the RAG agent service."""

    use_openai = _use_openai()

    # if use_openai is True, then return default openai base URL
    if use_openai:
        return "https://api.openai.com/v1"

    host = os.getenv("EXAMPLE_HOST")
    port = os.getenv("EXAMPLE_PORT")
    if not host or not port:
        host = host or "127.0.0.1"
        port = port or "8000"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{port}/v1"


def _resolve_api_key() -> str:
    """Resolve the API key for the RAG server client."""
    return os.getenv("OPENAI_API_KEY", "local-agent")


def _resolve_model() -> str:
    """Resolve the model name for the request payload."""
    if _use_openai():
        return os.getenv("OPENAI_MODEL", "gpt-5.4")
    return os.getenv("OPENAI_MODEL", "local-llm")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the RAG client."""

    parser = argparse.ArgumentParser(description="Query the hybrid RAG OpenAI-compatible endpoint.")
    parser.add_argument("--question", required=True, help="Question to ask the RAG server.")
    parser.add_argument("--model", default=None, help="Override the model name in the request payload.")
    parser.add_argument("--base-url", default=None, help="Override the API base URL for the RAG server.")
    parser.add_argument("--api-key", default=None, help="Override the API key for the RAG server.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Send a chat completion request to the RAG server and print the answer."""

    args = parse_args(argv)
    base_url = args.base_url or _resolve_base_url()
    api_key = args.api_key or _resolve_api_key()
    model = args.model or _resolve_model()

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.question}],
        temperature=0.0,
    )

    content = response.choices[0].message.content if response.choices else ""

    print("\n")
    print("Question:")
    print(args.question)
    print("\n")
    print("Answer:")
    print(content)
    print("\n")


if __name__ == "__main__":
    main()
