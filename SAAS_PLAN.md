# Narra SaaS — Build Plan

**Product:** A read-along reading assistant. Import your PDFs / EPUBs / articles, hear
them in natural neural speech with exact word-by-word highlighting. A cheaper, open-voice
alternative to Speechify.

**Wedge / moat:** Narra's exact `pred_dur` word-sync read-along + free open Kokoro voices,
undercutting Speechify ($139/yr).

**Target users:** Readers with dyslexia/ADHD/ESL, students, professionals reading long
docs. Later: schools (accessibility budgets).

---

## Current state (starting point)

| Layer | Today | Problem for SaaS |
|---|---|---|
| TTS engine | Kokoro-82M via Apple **MLX** (`mlx-audio`) | **Mac-only.** Won't run on cloud GPUs. Hard blocker. |
| Backend | FastAPI, `backend/server.py` (~316 lines) | No auth, no DB, no quota, single-process |
| Frontend | One file `frontend/index.html` (~854 lines), vanilla JS | No accounts, localStorage only, no import |
| Data | None (localStorage) | No cross-device sync, no library |
| Billing | None | — |

---

## Phase 0 — De-risk the engine (the make-or-break blocker)

> Do this FIRST. If hosted Kokoro is too slow/expensive, the whole business model changes.

- [ ] Port synthesis off MLX to a **cloud-GPU-capable** Kokoro build:
      `kokoro` (PyTorch) or **`kokoro-onnx`** (ONNX Runtime, runs on CPU *and* GPU, cheapest).
- [ ] **Critical:** preserve the word-timestamp output (`tokens: [{t, ws, s, e}]`). The
      read-along is the whole product — confirm the PyTorch/ONNX path still exposes
      per-word durations. If it doesn't, that's the #1 risk to solve before anything else.
- [ ] Abstract the engine behind an interface so `server.py` doesn't care which backend
      runs (`MlxEngine` for local dev on your Mac, `OnnxEngine` for the server).
- [ ] Benchmark cost: measure seconds-of-audio per GPU-second on a cheap host
      (Modal / RunPod / Fly.io GPU). Derive your per-1000-char cost → sets pricing & free tier.
- [ ] Add a **job queue** (so 50 concurrent users don't crash one box): start with
      FastAPI `BackgroundTasks` + Redis/RQ, or Modal's built-in autoscaling.

**Exit criteria:** synthesize text on a Linux GPU host, word timings intact, known $/char.

---

## Phase 1 — SaaS foundation (accounts, quota, billing)

Stack: **Supabase** (Postgres + Auth + Storage, already available) + **Stripe**.

- [ ] **Auth:** Supabase email/OAuth (Google). Frontend gets a session JWT; backend
      verifies it on every `/api/tts` call.
- [ ] **DB schema (Supabase):**
  - `profiles` (user_id, plan, chars_used_this_period, period_resets_at)
  - `documents` (id, user_id, title, source_type, text, created_at)
  - `reading_state` (document_id, user_id, char_offset, last_voice, last_speed)
- [ ] **Quota enforcement:** middleware counts characters per request, blocks when the
      free cap is hit, returns a clean "upgrade" response. **This protects you from GPU
      bankruptcy** — must exist before any public launch.
- [ ] **Billing:** Stripe Checkout + customer portal; webhook → flip `profiles.plan`.
- [ ] **Tiers:**
  - Free: ~15k chars/mo, 3 voices, 1 device
  - Pro ~$8–10/mo: unlimited*, all voices, import, sync, MP3 export
- [ ] Rate limiting per IP/user.

**Exit criteria:** a stranger can sign up, hit the free cap, pay, and get unblocked.

---

## Phase 2 — The retention feature: read THEIR content

This is what makes people stay instead of bouncing. Build in this order:

- [ ] **Paste a URL → read the article** (Speechify's killer feature). Server-side fetch
      + readability extraction (`trafilatura` / `readability-lxml`).
- [ ] **Upload PDF** → extract text (`pymupdf`). Store in `documents`, store file in
      Supabase Storage.
- [ ] **Upload EPUB / DOCX** (`ebooklib`, `python-docx`).
- [ ] **Library UI:** list of saved documents, resume where you left off.
- [ ] Long-document handling: chunk text, synthesize on demand per section (don't render
      a whole book at once — kills GPU cost and latency).

**Exit criteria:** drop in a PDF or paste a blog URL, get instant read-along, come back
tomorrow and resume.

---

## Phase 3 — Stickiness: sync + mobile

The habit (and the lock-in) forms on the phone during the commute.

- [ ] Cross-device sync of library, reading position, voice/speed (already in DB → just
      wire the frontend to read/write it instead of localStorage).
- [ ] **PWA** (installable, offline shell) before a native app — fastest path to mobile.
- [ ] Background/lock-screen audio playback on mobile (Media Session API).
- [ ] MP3 export (you already have this) gated to Pro.

**Exit criteria:** start a doc on desktop, finish it on your phone, same position.

---

## Phase 4 — Growth & lock-in

- [ ] **Browser extension:** "Read this page with Narra" → biggest organic acquisition channel.
- [ ] Highlights & notes on synced text (accumulated value = switching cost).
- [ ] Reading stats / streaks.
- [ ] Accessibility polish: dyslexia font, bionic-reading toggle, adjustable spacing.
- [ ] Edu/Teams plan (seats, admin) — schools have real accessibility budgets.

---

## Suggested final stack

- **Frontend:** keep vanilla for MVP; migrate to a framework only when the library/auth
  UI gets complex. PWA wrapper for mobile.
- **Backend:** FastAPI (keep) + engine abstraction + job queue.
- **TTS host:** Modal or RunPod GPU (autoscaling, pay-per-use — matches usage-based cost).
- **Data/Auth/Storage:** Supabase.
- **Billing:** Stripe.

## Key risks (watch these)

1. **Word timings may not survive the MLX→ONNX port** — verify in Phase 0, it's the moat.
2. **GPU cost per user** — meter from day one or the free tier eats your runway.
3. **Voice quality vs ElevenLabs** — don't compete on creator voiceover; stay in reading.
4. **Speechify is the incumbent** — win on price + open voices + simplicity, not features.

## Recommended first milestone (MVP to charge money)

Phase 0 (engine on a GPU host) → Phase 1 (auth + quota + Stripe) → Phase 2 first item
(URL → read-along). That's a payable Speechify-lite. Everything else is iteration.
