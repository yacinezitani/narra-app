"""
Narra TTS server.

A thin FastAPI wrapper around Kokoro-82M. The actual model runs behind a
pluggable engine (Apple MLX locally, PyTorch CPU when hosted) selected by the
``TTS_ENGINE`` env var — see :mod:`backend.engines`. The engine is loaded once at
startup and reused for every request; audio is returned as in-memory WAV bytes,
no temp files on disk.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
from pathlib import Path

import soundfile as sf
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.auth import enforce_quota, get_user_id
from backend.billing import BILLING_ENABLED
from backend.billing import router as billing_router
from backend.ratelimit import enforce_rate_limit, is_owner
from backend.engines import (
    SAMPLE_RATE,
    VALID_VOICE_IDS,
    VOICES,
    get_engine,
    synthesize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("kokoro")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

# CORS: in production set ALLOWED_ORIGINS to your frontend URL(s), comma-separated
# (e.g. "https://narra.pages.dev"). Defaults to "*" for local dev.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="Narra TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Stripe billing routes only when configured (open local dev skips them).
if BILLING_ENABLED:
    app.include_router(billing_router)

# The chosen backend (MLX / PyTorch). Constructed once; thread-safe internally.
engine = get_engine()

# Warmup progress, polled by the client so it can show "getting ready".
warmup = {"model": False, "engine": False, "voices_ready": 0, "voices_total": len(VOICES), "ready": False}


def _prewarm() -> None:
    """Make the first real request fast by paying load/compile costs up front."""
    try:
        engine.warmup(warmup)
    except Exception:
        log.exception("Warmup failed (requests will still work, just colder).")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = "af_heart"
    speed: float = Field(1.0, ge=0.5, le=2.0)


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": engine.name, "warmup": warmup}


@app.on_event("startup")
def _start_warmup():
    threading.Thread(target=_prewarm, name="prewarm", daemon=True).start()


@app.get("/api/voices")
def voices():
    return {"voices": VOICES}


@app.post("/api/tts")
async def tts(
    req: TTSRequest,
    request: Request,
    user_id: str | None = Depends(get_user_id),
    x_narra_key: str | None = Header(default=None),
):
    """Generate speech and return audio plus a word-level timeline.

    Response JSON:
        {
          "audio":       base64-encoded WAV (24 kHz PCM16),
          "sample_rate": 24000,
          "duration":    float seconds,
          "tokens":      [{ "t": text, "ws": trailing_ws, "s": start, "e": end }]
        }
    Audio and timeline are returned together so the client renders both atomically.

    Protection layers, applied before any synthesis so we never spend compute on a
    rejected request:
      1. Owner bypass — a request with the X-Narra-Key owner secret skips all limits.
      2. Rate limit  — per-IP sliding window (protects the free backend from abuse).
      3. Quota       — per-user plan cap, only when Supabase auth is enabled.
    """
    if req.voice not in VALID_VOICE_IDS:
        raise HTTPException(400, f"Unknown voice '{req.voice}'")

    owner = is_owner(x_narra_key)
    enforce_rate_limit(request, owner)

    text = req.text.strip()
    if not owner:
        await enforce_quota(user_id, len(text))

    try:
        wav, tokens = synthesize(engine, text, req.voice, req.speed)
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


@app.get("/config.js")
def config_js():
    # Serves the same runtime config locally that Vercel/Netlify serve statically.
    return FileResponse(FRONTEND / "config.js", media_type="application/javascript")


@app.get("/logo.svg")
def logo():
    return FileResponse(FRONTEND / "logo.svg", media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn

    # HOST/PORT are configurable so the same entrypoint works locally
    # (127.0.0.1:8000) and on hosts that inject a port (HF Spaces uses 7860).
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
