"""
MLX backend — Kokoro-82M via Apple MLX.

This is the original, fast path used for local development on Apple Silicon. It
is *not* usable on Hugging Face Spaces (MLX requires an Apple GPU), so the hosted
SaaS uses :mod:`backend.engines.torch_engine` instead. Both implement the same
:class:`~backend.engines.base.BaseEngine` interface.
"""

from __future__ import annotations

import logging
from typing import Iterator

import mlx.core as mx
import numpy as np

from mlx_audio.tts.utils import load_model

from .base import VOICES, BaseEngine, Token

log = logging.getLogger("kokoro")

MODEL_REPO = "mlx-community/Kokoro-82M-bf16"


def _patch_kokoro_sinegen() -> None:
    """Work around a length-mismatch bug in mlx-audio's iSTFTNet vocoder.

    SineGen._f02sine() upsamples F0 via a downsample-then-upsample interpolation
    whose rounding can drift the output length by a few hundred samples, while
    _f02uv() preserves the input length exactly. The two are then multiplied/
    broadcast together, which crashes with a shape mismatch on certain (often
    longer) inputs. We crop/pad the sine output back to the input length so the
    source and U/V signals always line up.
    """
    from mlx_audio.tts.models.kokoro import istftnet

    if getattr(istftnet.SineGen, "_f02sine_len_patched", False):
        return

    original = istftnet.SineGen._f02sine

    def _f02sine_aligned(self, f0_values):
        sines = original(self, f0_values)
        target = f0_values.shape[1]
        cur = sines.shape[1]
        if cur > target:
            sines = sines[:, :target, :]
        elif cur < target:
            sines = mx.pad(sines, ((0, 0), (0, target - cur), (0, 0)))
        return sines

    istftnet.SineGen._f02sine = _f02sine_aligned
    istftnet.SineGen._f02sine_len_patched = True
    log.info("Applied SineGen length-alignment patch.")


class MlxEngine(BaseEngine):
    name = "mlx"

    def __init__(self) -> None:
        super().__init__()
        _patch_kokoro_sinegen()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            with self.lock:
                if self._model is None:
                    log.info("Loading %s (first run downloads ~330 MB)...", MODEL_REPO)
                    self._model = load_model(MODEL_REPO)
                    log.info("Model ready.")
        return self._model

    def infer(self, text: str, voice: str, speed: float) -> Iterator[tuple[np.ndarray, list[Token]]]:
        model = self.model
        lang_code = voice[0]  # 'a' American / 'b' British English
        with self.lock:
            pipeline = model._get_pipeline(lang_code)
            # NB: do NOT clear pipeline.voices — keeping the cache means a voice
            # pack is loaded once and reused, not reloaded (~3 s) every request.
            for res in pipeline(text, voice=voice, speed=speed):
                if res.audio is None:
                    continue
                wav = np.asarray(res.audio, dtype=np.float32).reshape(-1)
                toks = [
                    Token(t.text, t.whitespace, t.start_ts, t.end_ts)
                    for t in (res.tokens or [])
                ]
                yield wav, toks

    def warmup(self, progress: dict) -> None:
        """Load the model, build both English pipelines, and pre-load every voice."""
        model = self.model
        progress["model"] = True

        # One tiny generation per accent compiles the MLX/Metal kernels for both
        # the American and British phoneme paths (and triggers the slow misaki/
        # spaCy import).
        with self.lock:
            for lang, seed_voice in (("a", "af_heart"), ("b", "bf_emma")):
                pipe = model._get_pipeline(lang)
                for _ in pipe("Ready.", voice=seed_voice, speed=1.0):
                    pass
        progress["engine"] = True
        log.info("Engine warm. Pre-loading voice packs...")

        for v in VOICES:
            try:
                pipe = model._get_pipeline(v["id"][0])
                with self.lock:
                    pipe.load_voice(v["id"])
                progress["voices_ready"] += 1
            except Exception:
                log.exception("Failed to preload voice %s", v["id"])

        progress["ready"] = True
        log.info(
            "Warmup complete: %d/%d voices ready.",
            progress["voices_ready"], progress["voices_total"],
        )
