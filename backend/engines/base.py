"""
Engine abstraction for Narra's TTS.

The same app runs in two places with two different Kokoro backends:

  * **MlxEngine**   — Apple MLX, fast on your Mac (local dev / the original app).
  * **TorchEngine** — hexgrad's PyTorch ``kokoro``, runs on plain CPU so it can be
                      hosted for free on Hugging Face Spaces.

Both must produce the *exact same* output shape — a waveform plus a per-word
timeline (start/end seconds) — because Narra's read-along word highlighting is
the whole product. This module owns everything that is identical between the two
engines (the voice catalog, the paragraph-aware orchestration, the timeline
shaping) so each concrete engine only has to implement the raw inference call.
"""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np

SAMPLE_RATE = 24_000

# Kokoro voices. The first letter of the id encodes language/accent
# (a = American, b = British English); the second encodes gender (f / m).
# This catalog is engine-agnostic — both backends serve the same voices.
VOICES = [
    {"id": "af_heart",   "name": "Heart",   "accent": "American", "gender": "Female"},
    {"id": "af_bella",   "name": "Bella",   "accent": "American", "gender": "Female"},
    {"id": "af_nicole",  "name": "Nicole",  "accent": "American", "gender": "Female"},
    {"id": "af_aoede",   "name": "Aoede",   "accent": "American", "gender": "Female"},
    {"id": "af_kore",    "name": "Kore",    "accent": "American", "gender": "Female"},
    {"id": "af_sarah",   "name": "Sarah",   "accent": "American", "gender": "Female"},
    {"id": "af_nova",    "name": "Nova",    "accent": "American", "gender": "Female"},
    {"id": "af_sky",     "name": "Sky",     "accent": "American", "gender": "Female"},
    {"id": "am_adam",    "name": "Adam",    "accent": "American", "gender": "Male"},
    {"id": "am_michael", "name": "Michael", "accent": "American", "gender": "Male"},
    {"id": "am_echo",    "name": "Echo",    "accent": "American", "gender": "Male"},
    {"id": "am_eric",    "name": "Eric",    "accent": "American", "gender": "Male"},
    {"id": "am_fenrir",  "name": "Fenrir",  "accent": "American", "gender": "Male"},
    {"id": "am_liam",    "name": "Liam",    "accent": "American", "gender": "Male"},
    {"id": "am_onyx",    "name": "Onyx",    "accent": "American", "gender": "Male"},
    {"id": "am_puck",    "name": "Puck",    "accent": "American", "gender": "Male"},
    {"id": "bf_emma",    "name": "Emma",    "accent": "British",  "gender": "Female"},
    {"id": "bf_isabella","name": "Isabella","accent": "British",  "gender": "Female"},
    {"id": "bf_alice",   "name": "Alice",   "accent": "British",  "gender": "Female"},
    {"id": "bf_lily",    "name": "Lily",    "accent": "British",  "gender": "Female"},
    {"id": "bm_george",  "name": "George",  "accent": "British",  "gender": "Male"},
    {"id": "bm_fable",   "name": "Fable",   "accent": "British",  "gender": "Male"},
    {"id": "bm_lewis",   "name": "Lewis",   "accent": "British",  "gender": "Male"},
    {"id": "bm_daniel",  "name": "Daniel",  "accent": "British",  "gender": "Male"},
]
VALID_VOICE_IDS = {v["id"] for v in VOICES}


@dataclass
class Token:
    """One Kokoro token with its (segment-relative) timing.

    ``start`` / ``end`` are seconds within the segment the token came from, or
    ``None`` for tokens the model didn't time (rare punctuation). The shared
    orchestrator shifts them into a continuous document timeline.
    """

    text: str
    whitespace: str
    start: float | None
    end: float | None


class BaseEngine(ABC):
    """Common surface every TTS backend implements.

    Concrete engines only provide ``infer`` (raw, segment-level synthesis) and
    ``warmup``. The paragraph splitting and timeline assembly live in the shared
    :func:`synthesize` function below so both engines stay byte-for-byte
    compatible from the client's point of view.
    """

    name: str = "base"

    def __init__(self) -> None:
        # Kokoro pipelines are not thread-safe; serialize generation so
        # concurrent requests can't corrupt model state.
        self.lock = threading.Lock()

    @abstractmethod
    def infer(self, text: str, voice: str, speed: float) -> Iterator[tuple[np.ndarray, list[Token]]]:
        """Synthesize one text segment.

        Yields ``(waveform, tokens)`` per pipeline result, where ``waveform`` is
        a 1-D float32 array at :data:`SAMPLE_RATE` and ``tokens`` carry
        segment-relative timings. Must hold ``self.lock`` while touching the model.
        """

    @abstractmethod
    def warmup(self, progress: dict) -> None:
        """Pay one-time load/compile costs up front, updating ``progress`` in place."""


def synthesize(engine: BaseEngine, text: str, voice: str, speed: float) -> tuple[np.ndarray, list[dict]]:
    """Run an engine over a full document and build one continuous timeline.

    Kokoro splits on newlines and discards them, which flattens the document. We
    split here while *keeping* the separators, feed each paragraph to the engine
    individually, and re-insert the line breaks as structural tokens so the
    reader preserves the original paragraph formatting. Each segment's timestamps
    are relative to that segment, so we shift them by the running audio offset to
    build one timeline matching the concatenated waveform.

    Returns ``(waveform, tokens)`` where each token dict is
    ``{"t": text, "ws": trailing_whitespace, "s": start_s, "e": end_s}`` — the
    exact shape the frontend's read-along highlighting consumes.
    """
    chunks: list[np.ndarray] = []
    tokens: list[dict] = []
    offset = 0.0  # seconds of audio emitted so far

    parts = re.split(r"(\n+)", text)

    for part in parts:
        if part == "":
            continue
        if part.strip() == "":
            # a run of newlines — a structural break, no audio
            tokens.append({"t": "", "ws": part, "s": None, "e": None})
            continue
        for wav, seg_tokens in engine.infer(part, voice, speed):
            if wav is None or len(wav) == 0:
                continue
            for tok in seg_tokens:
                tokens.append(
                    {
                        "t": tok.text,
                        "ws": tok.whitespace,
                        "s": round(offset + tok.start, 3) if tok.start is not None else None,
                        "e": round(offset + tok.end, 3) if tok.end is not None else None,
                    }
                )
            chunks.append(wav)
            offset += len(wav) / SAMPLE_RATE

    if not chunks:
        raise RuntimeError("Model produced no audio")
    return np.concatenate(chunks), tokens
