"""DuckDuckGo-first keyless internet-search client."""
from __future__ import annotations

from typing import Any, Dict, List

from .config import Settings


# Call backends one at a time. The DDGS package currently has an open bug where
# one failing backend can discard results returned by another backend when a
# comma-delimited backend list is used.
_SEARCH_BACKENDS = ("duckduckgo", "brave", "mojeek", "wikipedia")


class DuckDuckGoSearchClient:
    """Search DuckDuckGo first, then use keyless DDGS fallbacks if blocked."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        if client is None:
            from ddgs import DDGS

            client = DDGS(timeout=int(settings.web_search_timeout))
        self.client = client

    def search(self, question: str) -> List[Dict[str, Any]]:
        """Return normalized results for the exact question."""
        question = question.strip()
        if not question:
            raise ValueError("Search question must not be empty")

        errors: list[str] = []

        for backend in _SEARCH_BACKENDS:
            try:
                raw_results = self.client.text(
                    question,
                    region=self.settings.duckduckgo_region,
                    safesearch="moderate",
                    max_results=self.settings.web_search_max_results,
                    backend=backend,
                )
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
                continue

            results = self._normalize(raw_results, backend)
            if results:
                return results

            errors.append(f"{backend}: no results")

        details = "; ".join(errors)
        raise RuntimeError(f"No keyless web-search backend returned results. {details}")

    @staticmethod
    def _normalize(
        raw_results: Any,
        backend: str,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for item in raw_results or []:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title") or "").strip()
            url = str(item.get("href") or item.get("url") or "").strip()
            content = str(item.get("body") or item.get("content") or "").strip()

            if not title and not url and not content:
                continue

            provider = "duckduckgo" if backend == "duckduckgo" else f"ddgs/{backend}"
            results.append(
                {
                    "provider": provider,
                    "title": title,
                    "url": url,
                    "content": content,
                    "score": item.get("score"),
                }
            )

        return results


__all__ = ["DuckDuckGoSearchClient"]
