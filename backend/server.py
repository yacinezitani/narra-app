"""
Kokoro-MLX TTS server.

A thin FastAPI wrapper around the Kokoro-82M model running locally via MLX
(Apple Silicon). The model is loaded once at startup and reused for every
request. Audio is returned as in-memory WAV bytes — no temp files on disk.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import threading
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from mlx_audio.tts.utils import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("kokoro")


def _patch_kokoro_sinegen() -> None:
    """Work around a length-mismatch bug in mlx-audio's iSTFTNet vocoder.

    SineGen._f02sine() upsamples F0 via a downsample-then-upsample interpolation
    whose rounding can drift the output length by a few hundred samples, while
    _f02uv() preserves the input length exactly. The two are then multiplied/
    broadcast together (`noise_amp * normal(sine_waves.shape)`), which crashes
    with e.g. "Shapes (1,140400,1) and (1,140700,9) cannot be broadcast" on
    certain (often longer) inputs. We crop/pad the sine output back to the input
    length so the source and U/V signals always line up.
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


_patch_kokoro_sinegen()

MODEL_REPO = "mlx-community/Kokoro-82M-bf16"
SAMPLE_RATE = 24_000
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

# Kokoro voices. The first letter of the id encodes the language/accent:
#   a = American English   b = British English
# The second letter encodes gender:  f = female   m = male
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

app = FastAPI(title="Kokoro-MLX TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The model is not thread-safe; serialize generation behind a lock so concurrent
# requests don't corrupt MLX state.
_model = None
_model_lock = threading.Lock()

# Warmup progress, polled by the client so it can show "getting ready".
warmup = {"model": False, "engine": False, "voices_ready": 0, "voices_total": len(VOICES), "ready": False}


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                log.info("Loading %s (first run downloads ~330 MB)...", MODEL_REPO)
                _model = load_model(MODEL_REPO)
                log.info("Model ready.")
    return _model


def _prewarm() -> None:
    """Make the first real request fast.

    Loading the model, importing misaki/spaCy, compiling the Metal kernels on the
    first inference, and fetching each voice pack are all one-time costs. We pay
    them up front in a background thread so the user never waits on a cold path.
    """
    try:
        model = get_model()
        warmup["model"] = True

        # Build the English pipelines (this triggers the slow spaCy/misaki import)
        # and run one tiny generation per accent to compile the MLX kernels for
        # both the American and British phoneme paths.
        with _model_lock:
            for lang, seed_voice in (("a", "af_heart"), ("b", "bf_emma")):
                pipe = model._get_pipeline(lang)
                for _ in pipe("Ready.", voice=seed_voice, speed=1.0):
                    pass
        warmup["engine"] = True
        log.info("Engine warm. Pre-loading voice packs...")

        # Pre-load every voice pack so switching voices is instant.
        for v in VOICES:
            try:
                pipe = model._get_pipeline(v["id"][0])
                with _model_lock:
                    pipe.load_voice(v["id"])
                warmup["voices_ready"] += 1
            except Exception:
                log.exception("Failed to preload voice %s", v["id"])

        warmup["ready"] = True
        log.info("Warmup complete: %d/%d voices ready.", warmup["voices_ready"], warmup["voices_total"])
    except Exception:
        log.exception("Warmup failed (requests will still work, just colder).")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = "af_heart"
    speed: float = Field(1.0, ge=0.5, le=2.0)


def synthesize(text: str, voice: str, speed: float) -> tuple[np.ndarray, list[dict]]:
    """Run Kokoro and return (waveform, token timeline).

    We drive the pipeline directly (rather than ``model.generate``) because the
    pipeline yields per-segment tokens carrying word-level start/end timestamps,
    which the high-level wrapper discards. Each segment's timestamps are relative
    to that segment, so we shift them by the running audio offset to build one
    continuous timeline matching the concatenated waveform.
    """
    model = get_model()
    lang_code = voice[0]  # 'a' American / 'b' British English
    chunks: list[np.ndarray] = []
    tokens: list[dict] = []
    offset = 0.0  # seconds of audio emitted so far

    # Kokoro splits on newlines and discards them, which flattens the document.
    # We split here while keeping the separators, feed each paragraph to the
    # pipeline individually, and re-insert the line breaks as structural tokens
    # so the reader preserves the original paragraph formatting.
    parts = re.split(r"(\n+)", text)

    with _model_lock:
        pipeline = model._get_pipeline(lang_code)
        # NB: do NOT clear pipeline.voices here — keeping the cache means a voice
        # pack is loaded once and reused, instead of reloaded (~3s) every request.
        for part in parts:
            if part == "":
                continue
            if part.strip() == "":
                # a run of newlines — a structural break, no audio
                tokens.append({"t": "", "ws": part, "s": None, "e": None})
                continue
            for res in pipeline(part, voice=voice, speed=speed):
                audio = res.audio
                if audio is None:
                    continue
                wav = np.asarray(audio, dtype=np.float32).reshape(-1)
                for tok in res.tokens or []:
                    start = tok.start_ts
                    end = tok.end_ts
                    tokens.append(
                        {
                            "t": tok.text,
                            "ws": tok.whitespace,
                            # null when the model didn't time this token (rare punctuation)
                            "s": round(offset + start, 3) if start is not None else None,
                            "e": round(offset + end, 3) if end is not None else None,
                        }
                    )
                chunks.append(wav)
                offset += len(wav) / SAMPLE_RATE

    if not chunks:
        raise HTTPException(500, "Model produced no audio")
    return np.concatenate(chunks), tokens


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_REPO, "loaded": _model is not None, "warmup": warmup}


@app.on_event("startup")
def _start_warmup():
    threading.Thread(target=_prewarm, name="prewarm", daemon=True).start()


@app.get("/api/voices")
def voices():
    return {"voices": VOICES}


@app.post("/api/tts")
def tts(req: TTSRequest):
    """Generate speech and return audio plus a word-level timeline.

    Response JSON:
        {
          "audio":       base64-encoded WAV (24 kHz PCM16),
          "sample_rate": 24000,
          "duration":    float seconds,
          "tokens":      [{ "t": text, "ws": trailing_ws, "s": start, "e": end }]
        }
    Audio and timeline are returned together so the client renders both atomically.
    """
    if req.voice not in VALID_VOICE_IDS:
        raise HTTPException(400, f"Unknown voice '{req.voice}'")
    try:
        wav, tokens = synthesize(req.text.strip(), req.voice, req.speed)
    except HTTPException:
        raise
    except Exception as exc:  # surface model errors to the client cleanly
        log.exception("Generation failed")
        raise HTTPException(500, str(exc))

    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return {
        "audio": base64.b64encode(buf.getvalue()).decode("ascii"),
        "sample_rate": SAMPLE_RATE,
        "duration": round(len(wav) / SAMPLE_RATE, 3),
        "tokens": tokens,
    }


@app.post("/api/transcode")
async def transcode(request: Request):
    """Convert the already-generated WAV (raw request body) to MP3.

    Transcoding the exact audio the client holds — rather than re-synthesizing —
    means the download matches what the user heard, with no extra model run.
    """
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "Empty audio body")
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        raise HTTPException(400, "Could not read audio (expected WAV)")

    out = io.BytesIO()
    sf.write(out, data, sr, format="MP3")
    return Response(
        content=out.getvalue(),
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'attachment; filename="narra.mp3"'},
    )


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/logo.svg")
def logo():
    return FileResponse(FRONTEND / "logo.svg", media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
