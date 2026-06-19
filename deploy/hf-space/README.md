---
title: Narra TTS
emoji: 🗣️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Narra TTS — Hugging Face Space (backend)

This folder holds everything needed to host Narra's TTS backend on a **free HF
Spaces CPU** instance. The model is Kokoro-82M run via PyTorch on CPU (no GPU
required), which keeps it on the free tier.

> The YAML front matter above is the Space "card" config. When you create the
> Space, this README becomes the Space's `README.md` and HF reads `sdk: docker`
> + `app_port: 7860` from it.

## What gets deployed

A Space is its own git repo. Its **root** must contain:

```
Dockerfile            (deploy/hf-space/Dockerfile)
requirements-hf.txt   (deploy/hf-space/requirements-hf.txt)
README.md             (this file, with the front matter)
backend/              (the app — copied from the repo)
frontend/             (the UI — copied from the repo)
```

## One-time setup

1. Create a Space: https://huggingface.co/new-space → SDK **Docker** → **CPU basic (free)**.
2. Clone it locally:
   ```bash
   git clone https://huggingface.co/spaces/<you>/narra-tts space && cd space
   ```
3. Assemble the Space contents from this repo (run from the repo root):
   ```bash
   ./deploy/hf-space/sync.sh ../space
   ```
4. Commit & push — the Space builds the Docker image and goes live:
   ```bash
   cd ../space && git add -A && git commit -m "Deploy Narra TTS" && git push
   ```

## Config (Space → Settings → Variables)

| Variable          | Value                              | Why |
|-------------------|------------------------------------|-----|
| `TTS_ENGINE`      | `torch`                            | Force the CPU backend (also set in the Dockerfile). |
| `ALLOWED_ORIGINS` | `https://<your-frontend-domain>`   | Lock CORS to your real frontend before launch. |

## Notes & caveats

- **First request is slow** (cold model load + warmup). The `/api/health`
  endpoint reports warmup progress so the frontend can show "getting ready".
- **CPU is slower than your Mac's MLX** — expect a few seconds per paragraph.
  Synthesize per section for long documents, not whole books at once.
- **Free CPU has limited RAM/concurrency.** Fine for a beta; move TTS to a
  pay-per-use GPU host (Modal/RunPod) once paid traffic arrives.
- The word-timing read-along works here because we use the PyTorch `kokoro`
  package (same misaki pipeline as MLX), **not** `kokoro-onnx`.
