"""
PyTorch backend — Kokoro-82M via hexgrad's ``kokoro`` package.

This is the **hosted** path (Hugging Face Spaces, free CPU tier). We deliberately
use the official PyTorch ``kokoro`` package rather than ``kokoro-onnx`` because it
runs the same *misaki* G2P pipeline and therefore exposes per-word timestamps
(``MToken.start_ts`` / ``end_ts``) — which Narra's read-along highlighting depends
on. ``kokoro-onnx`` returns audio only and would silently break the highlighting.

Kokoro-82M is tiny (82M params) and runs at roughly real-time on CPU, which is
what makes free hosting viable. No GPU required.

System dependency: ``espeak-ng`` must be installed (misaki falls back to it for
out-of-dictionary words). The HF Spaces Dockerfile installs it.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from .base import VOICES, BaseEngine, Token

log = logging.getLogger("kokoro")

REPO_ID = "hexgrad/Kokoro-82M"


class TorchEngine(BaseEngine):
    name = "torch"

    def __init__(self) -> None:
        super().__init__()
        # One KPipeline per language code ('a' American, 'b' British). Built
        # lazily and cached — constructing a pipeline loads the model weights.
        self._pipelines: dict[str, object] = {}

    def _pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            # Imported lazily so a Mac dev box that never sets TTS_ENGINE=torch
            # doesn't need torch/kokoro installed at all.
            from kokoro import KPipeline

            log.info("Building Kokoro KPipeline for lang '%s'...", lang_code)
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=REPO_ID)
        return self._pipelines[lang_code]

    def infer(self, text: str, voice: str, speed: float) -> Iterator[tuple[np.ndarray, list[Token]]]:
        lang_code = voice[0]  # 'a' American / 'b' British English
        with self.lock:
            pipeline = self._pipeline(lang_code)
            # We already split the document into paragraphs upstream, so disable
            # Kokoro's own newline splitting (split_pattern=None) to get one
            # result per segment with a clean, contiguous timeline.
            for res in pipeline(text, voice=voice, speed=speed, split_pattern=None):
                if res.audio is None:
                    continue
                # res.audio is a 1-D torch FloatTensor at 24 kHz.
                wav = res.audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
                toks = [
                    Token(t.text, t.whitespace, t.start_ts, t.end_ts)
                    for t in (res.tokens or [])
                ]
                yield wav, toks

    def warmup(self, progress: dict) -> None:
        """Build both English pipelines and pre-load every voice pack.

        On CPU the first inference per accent also pays the misaki/spaCy import
        and any espeak warmup, so we do it up front in the background thread.
        """
        for lang, seed_voice in (("a", "af_heart"), ("b", "bf_emma")):
            pipeline = self._pipeline(lang)
            with self.lock:
                for _ in pipeline("Ready.", voice=seed_voice, speed=1.0, split_pattern=None):
                    pass
        progress["model"] = True
        progress["engine"] = True
        log.info("Engine warm. Pre-loading voice packs...")

        for v in VOICES:
            try:
                pipeline = self._pipeline(v["id"][0])
                with self.lock:
                    # load_voice fetches and caches the voice tensor so the first
                    # real request that uses it doesn't pay the download.
                    pipeline.load_voice(v["id"])
                progress["voices_ready"] += 1
            except Exception:
                log.exception("Failed to preload voice %s", v["id"])

        progress["ready"] = True
        log.info(
            "Warmup complete: %d/%d voices ready.",
            progress["voices_ready"], progress["voices_total"],
        )
