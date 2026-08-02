# Copyright (c) 2026 David vonThenen. All rights reserved.
# Restricted distribution. Unauthorized copying or hosting of this file via any medium
# is strictly prohibited. Proprietary and confidential.

"""Sentence-transformer embedding helpers."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from common.config import Settings


class EmbeddingModel:
    """Load one embedding model for ingestion and query encoding."""

    def __init__(self, settings: Settings) -> None:
        kwargs = {}
        if settings.embedding_device:
            kwargs["device"] = settings.embedding_device

        self.model = SentenceTransformer(settings.embedding_model, **kwargs)
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("The embedding model did not report its vector dimension.")
        self.dimension = int(dimension)

    def encode(self, texts: str | Iterable[str]) -> np.ndarray:
        """Encode one string or a sequence with normalized vectors."""

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def to_list(vector: object) -> list[float]:
    """Convert a NumPy or tensor-like vector into plain Python floats."""

    if hasattr(vector, "tolist"):
        values = vector.tolist()
    else:
        values = list(vector)  # type: ignore[arg-type]
    return [float(value) for value in values]
