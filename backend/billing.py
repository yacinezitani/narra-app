"""
Stripe billing — Checkout + webhooks for the Narra Pro subscription.

Flow:

  1. A signed-in user clicks "Upgrade" -> frontend calls POST /api/billing/checkout.
     We create (or reuse) a Stripe Customer linked to their Supabase user id and
     return a Stripe Checkout URL. The browser redirects there to pay.

  2. Stripe processes the payment and calls our webhook (POST /api/billing/webhook)
     with the *signed* event. We verify the signature, then flip the user's
     ``profiles.plan`` in Supabase to 'pro' (or back to 'free' on cancellation).
     The webhook — never the redirect — is the source of truth for entitlement,
     because the redirect can be skipped/forged but a signed webhook cannot.

  3. A user manages/cancels via the Stripe Customer Portal
     (POST /api/billing/portal returns its URL).

This module is **opt-in**: if STRIPE_SECRET_KEY isn't set, the router is simply
not mounted, so local dev runs without Stripe. All secrets come from env vars —
nothing is hardcoded.

Required env:
    STRIPE_SECRET_KEY        sk_live_... / sk_test_...
    STRIPE_PRICE_ID          price_...  (the recurring Pro price)
    STRIPE_WEBHOOK_SECRET    whsec_...  (from the webhook endpoint config)
    APP_BASE_URL             https://your-frontend   (for success/cancel redirects)
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY   (to update the profile)
"""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request

from backend.auth import get_user_id

log = logging.getLogger("kokoro")

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

BILLING_ENABLED = bool(STRIPE_SECRET_KEY and STRIPE_PRICE_ID)

# The router is only mounted by server.py when BILLING_ENABLED is true.
router = APIRouter(prefix="/api/billing", tags=["billing"])

if BILLING_ENABLED:
    import stripe  # imported lazily so the dep is optional in local dev

    stripe.api_key = STRIPE_SECRET_KEY
    log.info("Stripe billing ENABLED.")
else:
    log.info("Stripe billing DISABLED (set STRIPE_SECRET_KEY + STRIPE_PRICE_ID to enable).")


# ---------------------------------------------------------------------------
# Supabase helpers (service-role: bypasses RLS, server-trusted only).
# ---------------------------------------------------------------------------
def _sb_headers() -> dict:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


async def _get_or_create_customer(user_id: str, email: str | None) -> str:
    """Return the Stripe customer id for a user, creating + persisting it once.

    We store the id on ``profiles.stripe_customer_id`` so repeat checkouts and
    webhook lookups map cleanly between Stripe and Supabase users.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        # Look up an existing customer id on the profile.
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=_sb_headers(),
            params={"id": f"eq.{user_id}", "select": "stripe_customer_id"},
        )
        resp.raise_for_status()
        rows = resp.json()
        if rows and rows[0].get("stripe_customer_id"):
            return rows[0]["stripe_customer_id"]

        # None yet — create one in Stripe, tagging it with our user id so the
        # webhook can map back even if the DB write below ever races.
        customer = stripe.Customer.create(email=email, metadata={"supabase_user_id": user_id})

        # Persist it on the profile.
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=_sb_headers(),
            params={"id": f"eq.{user_id}"},
            json={"stripe_customer_id": customer.id},
        )
        return customer.id


async def _set_plan(customer_id: str, plan: str) -> None:
    """Flip a customer's plan in Supabase, keyed by their Stripe customer id."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.patch(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers=_sb_headers(),
            params={"stripe_customer_id": f"eq.{customer_id}"},
            json={"plan": plan},
        )
        resp.raise_for_status()
    log.info("Set plan=%s for stripe customer %s", plan, customer_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/checkout")
async def create_checkout(
    user_id: str | None = Depends(get_user_id),
    x_user_email: str | None = Header(default=None),
):
    """Create a Stripe Checkout session and return its URL for the browser to open."""
    if user_id is None:
        raise HTTPException(401, "Sign in to upgrade")

    customer_id = await _get_or_create_customer(user_id, x_user_email)
    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        # Echo the user id in metadata as a belt-and-suspenders mapping.
        subscription_data={"metadata": {"supabase_user_id": user_id}},
        success_url=f"{APP_BASE_URL}/?upgraded=1",
        cancel_url=f"{APP_BASE_URL}/?canceled=1",
    )
    return {"url": session.url}


@router.post("/portal")
async def customer_portal(user_id: str | None = Depends(get_user_id)):
    """Return a Stripe Customer Portal URL so the user can manage/cancel billing."""
    if user_id is None:
        raise HTTPException(401, "Sign in to manage billing")

    customer_id = await _get_or_create_customer(user_id, None)
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=f"{APP_BASE_URL}/",
    )
    return {"url": session.url}


@router.post("/webhook")
async def webhook(request: Request, stripe_signature: str | None = Header(default=None)):
    """Handle Stripe events. The signed webhook is the source of truth for plan state.

    We only act on subscription lifecycle events:
      * checkout.session.completed / customer.subscription.updated (active) -> 'pro'
      * customer.subscription.deleted (or status not active)               -> 'free'
    """
    payload = await request.body()
    try:
        # Signature verification is what makes the webhook trustworthy — it proves
        # the event really came from Stripe and wasn't forged by a caller.
        event = stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_WEBHOOK_SECRET)
    except Exception as exc:
        log.warning("Stripe webhook signature verification failed: %s", exc)
        raise HTTPException(400, "Invalid signature")

    etype = event["type"]
    obj = event["data"]["object"]

    if etype == "checkout.session.completed":
        await _set_plan(obj["customer"], "pro")

    elif etype == "customer.subscription.updated":
        # 'active'/'trialing' keep Pro; anything else (past_due, canceled) drops it.
        plan = "pro" if obj["status"] in ("active", "trialing") else "free"
        await _set_plan(obj["customer"], plan)

    elif etype == "customer.subscription.deleted":
        await _set_plan(obj["customer"], "free")

    # Acknowledge everything else so Stripe stops retrying.
    return {"received": True}
