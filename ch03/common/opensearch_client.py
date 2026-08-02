# Copyright (c) 2026 David vonThenen. All rights reserved.
# Restricted distribution. Unauthorized copying or hosting of this file via any medium
# is strictly prohibited. Proprietary and confidential.

"""OpenSearch client and vector-index creation helpers."""
from __future__ import annotations

from opensearchpy import OpenSearch

from common.config import Settings


def create_client(settings: Settings) -> OpenSearch:
    """Create a client for the configured OpenSearch node."""

    http_auth = None
    if settings.opensearch_user:
        http_auth = (settings.opensearch_user, settings.opensearch_password)

    return OpenSearch(
        hosts=[
            {
                "host": settings.opensearch_host,
                "port": settings.opensearch_port,
            }
        ],
        http_auth=http_auth,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        ssl_show_warn=False,
    )


def ensure_index(settings: Settings, embedding_dimension: int) -> None:
    """Create the vector index when it does not already exist."""

    client = create_client(settings)
    if client.indices.exists(index=settings.opensearch_index):
        return

    body = {
        "settings": {
            "index": {
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "path": {"type": "keyword"},
                "title": {"type": "keyword"},
                "category": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
            }
        },
    }
    client.indices.create(index=settings.opensearch_index, body=body)
