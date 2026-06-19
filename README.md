<div align="center">

<img src="frontend/logo.svg" width="88" height="88" alt="Narra logo" />

# Narra

**Read-along text-to-speech that runs 100% on your Mac.**

Type or paste any text, press play, and watch every word highlight in perfect sync
with natural neural speech — powered by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
on Apple [MLX](https://github.com/ml-explore/mlx). No cloud, no API keys, no limits.

![platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)
![python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![runs](https://img.shields.io/badge/runs-100%25%20local-2ea44f)
![license](https://img.shields.io/badge/license-MIT-555)

</div>

---

## ✨ Features

- **🎯 Karaoke word highlighting** — the active word lights up and prior words shade as
  the audio plays, auto-scrolling to keep pace. Timing comes from Kokoro's own
  alignment (`pred_dur`), so it's exact, not guessed.
- **👆 Click-to-seek** — click any word to jump the audio straight to it.
- **🎛️ Pro transport** — play/pause, ±10 s skip, a scrubbable timeline with sentence
  markers, and a download-to-MP3 button.
- **🗣️ Voice picker with previews** — 24 American/British voices, each with a ▶ button
  to audition a sample before choosing.
- **⏩ Speed control** — preset chips (0.75×–2×) **plus** a slider for any custom rate,
  applied live without re-synthesizing.
- **🌗 Light & dark mode** — one-click toggle, follows your system theme by default.
- **💾 Remembers your session** — your text, voice, and speed are saved locally and
  restored on reload.
- **⚡ Fast** — voices and engine pre-warm at startup; warm synthesis runs several
  times faster than real-time (a paragraph in ~2 s).
- **🔒 Private** — your text never leaves your machine.

## 🧰 Requirements

- A **Mac with Apple Silicon** (M1 or newer) — MLX uses the Apple GPU.
- **Python 3.12**
- ~1 GB free disk (model + voice packs, downloaded once and cached)

## 🚀 Quick start

```bash
git clone https://github.com/<you>/narra.git
cd narra
./run.sh
```

`run.sh` creates a virtualenv, installs dependencies, and starts the server.
The first launch downloads the model (~330 MB). Then open:

```
http://127.0.0.1:8000
```

Type some text, hit **Generate & Read**, and watch it light up. Keyboard:
<kbd>Space</kbd> play/pause · <kbd>←</kbd>/<kbd>→</kbd> seek · <kbd>⌘</kbd>+<kbd>Enter</kbd> generate.

<details>
<summary>Manual setup (without <code>run.sh</code>)</summary>

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```
</details>

## 🏗️ How it works

```
Browser (frontend/index.html)
   │  POST /api/tts  { text, voice, speed }
   ▼
FastAPI (backend/server.py)
   │  Kokoro-82M via MLX, pipeline driven directly so we keep word tokens
   │   · concatenates per-segment audio
   │   · shifts each segment's word timestamps by the running offset
   ▼
JSON { audio (base64 WAV), duration, tokens: [{ t, ws, s, e }] }
   ▼  played + highlighted word-by-word in the browser
```

The model and all 24 voice packs are loaded **once** in a background warmup thread at
startup, then cached — so your first click is already fast and switching voices is
near-instant.

## 📡 API

| Method | Path          | Returns                                                              |
|--------|---------------|---------------------------------------------------------------------|
| `GET`  | `/api/health` | `{ status, model, loaded, warmup }`                                  |
| `GET`  | `/api/voices` | `{ voices: [{ id, name, accent, gender }] }`                        |
| `POST` | `/api/tts`    | `{ audio, sample_rate, duration, tokens }`                          |

`tokens` is the spoken text in order; each is `{ t: word, ws: trailing space,
s: start_sec, e: end_sec }` (`s`/`e` are `null` for untimed punctuation).

```bash
curl -X POST http://127.0.0.1:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from Narra","voice":"af_heart","speed":1.0}' \
  | jq '{duration, first: .tokens[0]}'
```

## 🎙️ Voices

24 curated voices; the id encodes accent + gender — `a`/`b` = American/British,
`f`/`m` = female/male (e.g. `af_heart` = American female). Defaults to `af_heart`.

## ☁️ Hosting — can I put this on Vercel / Netlify?

**The static frontend, yes. The backend, no — and that's by design.**

Narra's speed and privacy come from **MLX, which runs only on Apple Silicon**. Vercel
and Netlify run Linux/x86 serverless functions with short timeouts and no persistent
in-memory model, so they physically cannot host the Kokoro/MLX backend. Narra is built
to run **locally on your Mac**, which is also what makes it fully private and free.

If you ever want a public URL, you have two routes:

1. **Static UI on Vercel + your Mac as backend** — deploy `frontend/` to Vercel/Netlify
   and expose your local server with a tunnel (e.g. `cloudflared tunnel --url
   http://localhost:8000`), then point the UI at that URL. Your Mac must stay running.
2. **Full cloud** — swap MLX for a Linux/GPU-compatible Kokoro build (ONNX, or a host
   like Modal / Replicate / HF Spaces). Always-on, but a larger change and likely paid GPU.

For most people, **running it locally is the best experience.**

## 🩹 Notes & troubleshooting

- **First run is slow / downloads a lot** — that's the one-time model + voice download.
  Subsequent runs are cached and fast.
- **`misaki` errors** — Kokoro needs the `misaki[en]` grapheme-to-phoneme engine; it's
  pinned in `requirements.txt`.
- An upstream iSTFTNet length-mismatch bug (crashes on some longer inputs) is fixed by a
  small startup monkeypatch (`_patch_kokoro_sinegen`) kept in our code, so the vendored
  package stays untouched.

## 📦 Project layout

```
narra/
├── backend/server.py     FastAPI app: model, warmup, /api/tts, /api/voices
├── frontend/
│   ├── index.html        Single-page reader (no build step)
│   └── logo.svg          App icon
├── requirements.txt
├── run.sh                One-command setup + launch
└── README.md
```

## 🙏 Credits

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad — the TTS model.
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — Kokoro on MLX.
- [misaki](https://github.com/hexgrad/misaki) — grapheme-to-phoneme.
- [MLX](https://github.com/ml-explore/mlx) by Apple.

## 📄 License

MIT — see [LICENSE](LICENSE). Model and voice weights are subject to their own licenses.
