# Deploying Narra

Narra deploys as **two separate pieces** — they cannot live on the same host:

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Frontend (static)          │  HTTPS  │  Backend / TTS (FastAPI)     │
│  Vercel or Netlify          │ ──────▶ │  Hugging Face Spaces (CPU)   │
│  serves frontend/           │         │  runs Kokoro via PyTorch     │
└─────────────────────────────┘         └──────────────┬───────────────┘
                                                        │ service-role
                              ┌─────────────────────────▼───────────────┐
                              │ Supabase: auth · profiles · quota · docs │
                              └──────────────────────────────────────────┘
                                         Stripe: Pro subscription
```

**Why split?** The TTS backend loads an ML model into memory and runs multi-second
inferences — serverless platforms (Vercel/Netlify functions) can't host that. They
host the static frontend; Hugging Face Spaces hosts the model.

---

## 1. Backend → Hugging Face Spaces

See [deploy/hf-space/README.md](deploy/hf-space/README.md). In short:

1. Create a **Docker**, **CPU basic (free)** Space.
2. From the repo root: `./deploy/hf-space/sync.sh <cloned-space-dir>`, then commit & push.
3. Set Space → Settings → Variables:
   - `TTS_ENGINE=torch`
   - `ALLOWED_ORIGINS=https://<your-frontend-domain>`
   - `SUPABASE_URL`, `SUPABASE_JWT_SECRET`, `SUPABASE_SERVICE_ROLE_KEY`
   - (optional, for billing) `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`, `STRIPE_WEBHOOK_SECRET`, `APP_BASE_URL`

Your Space URL looks like `https://<user>-<space>.hf.space`.

## 2. Frontend → Vercel or Netlify

Both are pre-configured to publish the static `frontend/` folder with **no build step**
([vercel.json](vercel.json) / [netlify.toml](netlify.toml)).

1. Edit [frontend/config.js](frontend/config.js): set
   `window.NARRA_API_BASE = "https://<your-space>.hf.space";`
2. Import the repo into Vercel **or** Netlify — it auto-detects the config. Deploy.
3. Put the resulting frontend URL into the backend's `ALLOWED_ORIGINS` (step 1).

> Local dev is unchanged: `./run.sh` serves the frontend from FastAPI with
> `NARRA_API_BASE=""` (same origin), and auth/quota/billing stay off until the
> `SUPABASE_*` / `STRIPE_*` env vars are set.

## 3. Supabase (already provisioned)

Project **narra** (`jthbiebytblycgjtqpvs`) — schema is applied. Grab the
**JWT secret** and **service-role key** from Settings → API for the backend env.
The frontend uses the **publishable** key only.

## 4. Stripe (when you're ready to charge)

Create a recurring **Pro** product/price, a webhook pointing at
`https://<your-space>.hf.space/api/billing/webhook`, then set the four `STRIPE_*`
env vars on the Space. Billing routes mount automatically once configured.
