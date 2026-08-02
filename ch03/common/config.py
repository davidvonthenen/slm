# Copyright (c) 2026 David vonThenen. All rights reserved.
# Restricted distribution. Unauthorized copying or hosting of this file via any medium
# is strictly prohibited. Proprietary and confidential.

"""Environment-based settings shared by ingestion and retrieval."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings for OpenSearch, embeddings, and RAG retrieval, etc."""

    opensearch_host: str
    opensearch_port: int
    opensearch_user: str
    opensearch_password: str
    opensearch_use_ssl: bool
    opensearch_verify_certs: bool
    opensearch_index: str
    embedding_model: str
    embedding_device: str | None
    rag_top_k: int
    rag_num_candidates: int
    web_search_max_results: int = 5
    web_search_timeout: float = 15.0
    tavily_api_key: str = ""
    duckduckgo_region: str = "wt-wt"



def load_settings() -> Settings:
    """Load settings from environment variables with local defaults."""

    device = os.getenv("EMBEDDING_DEVICE")
    return Settings(
        # OPENSEARCH
        opensearch_host=os.getenv("OPENSEARCH_HOST", "127.0.0.1"),
        opensearch_port=int(os.getenv("OPENSEARCH_PORT", "9200")),
        opensearch_user=os.getenv("OPENSEARCH_USER", "admin"),
        opensearch_password=os.getenv("OPENSEARCH_PASSWORD", "admin"),
        opensearch_use_ssl=_read_bool("OPENSEARCH_USE_SSL", False),
        opensearch_verify_certs=_read_bool("OPENSEARCH_VERIFY_CERTS", False),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", "bbc-vector-chunks"),

        # EMBEDDINGS
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        embedding_device=device if device else None,

        # RAG
        rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        rag_num_candidates=int(os.getenv("RAG_NUM_CANDIDATES", "25")),

        # TAVILY OR DUCKDUCKGO
        web_search_max_results=_get_int("WEB_SEARCH_MAX_RESULTS", Settings.web_search_max_results),
        web_search_timeout=_get_float("WEB_SEARCH_TIMEOUT", Settings.web_search_timeout),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        duckduckgo_region=os.getenv("DUCKDUCKGO_REGION", Settings.duckduckgo_region).strip(),
    )

__all__ = ["Settings", "load_settings"]
