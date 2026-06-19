# Hugging Face Spaces — Docker SDK image for Narra's TTS backend (PyTorch CPU).
#
# HF Spaces runs this container and routes traffic to port 7860. The PyTorch
# Kokoro engine runs on CPU (no GPU needed), which is what makes the free tier
# viable. See deploy/hf-space/README.md for the Space setup.

FROM python:3.12-slim

# espeak-ng: misaki's G2P fallback for out-of-dictionary words. libsndfile1:
# soundfile's backend for reading/writing WAV/MP3. ffmpeg: MP3 transcoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces best practice: run as a non-root user and keep all caches under it.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    # Model/voice downloads cache here (writable by the non-root user).
    HF_HOME=/home/user/.cache/huggingface \
    # Force the hosted backend regardless of what's installed.
    TTS_ENGINE=torch \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /home/user/app

# Install deps first so Docker layer-caches them across code changes.
COPY --chown=user requirements-hf.txt .
RUN pip install --no-cache-dir --user -r requirements-hf.txt

# Copy the app. The build context is the repo root (see README), so these paths
# are relative to it.
COPY --chown=user backend ./backend
COPY --chown=user frontend ./frontend

EXPOSE 7860

# uvicorn reads HOST/PORT from the env set above.
CMD ["python", "-m", "backend.server"]
