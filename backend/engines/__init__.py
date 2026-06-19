"""
Engine selection.

Picks the TTS backend based on the ``TTS_ENGINE`` env var:

    TTS_ENGINE=mlx     -> Apple MLX   (local dev on Apple Silicon)
    TTS_ENGINE=torch   -> PyTorch CPU (hosted, e.g. Hugging Face Spaces)
    TTS_ENGINE=auto    -> MLX if importable, else PyTorch  (default)

The rest of the app talks only to the abstract :class:`BaseEngine`, so swapping
backends never touches request-handling code.
"""

from __future__ import annotations

import logging
import os

from .base import (  # re-exported for convenience
    SAMPLE_RATE,
    VALID_VOICE_IDS,
    VOICES,
    BaseEngine,
    Token,
    synthesize,
)

log = logging.getLogger("kokoro")

__all__ = [
    "SAMPLE_RATE",
    "VALID_VOICE_IDS",
    "VOICES",
    "BaseEngine",
    "Token",
    "synthesize",
    "get_engine",
]


def _mlx_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("mlx") is not None


def get_engine() -> BaseEngine:
    choice = os.getenv("TTS_ENGINE", "auto").lower()

    if choice == "auto":
        choice = "mlx" if _mlx_available() else "torch"

    if choice == "mlx":
        from .mlx_engine import MlxEngine

        log.info("Using MLX engine (Apple Silicon).")
        return MlxEngine()
    if choice == "torch":
        from .torch_engine import TorchEngine

        log.info("Using PyTorch engine (CPU).")
        return TorchEngine()

    raise ValueError(f"Unknown TTS_ENGINE={choice!r} (expected mlx | torch | auto)")
