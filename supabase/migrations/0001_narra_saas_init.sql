-- Narra SaaS — initial schema
--
-- Tables: profiles (plan + usage), documents (the reading library),
-- reading_state (resume position per doc), usage_events (audit ledger).
-- Plus Row Level Security so each user only ever sees their own rows, a
-- signup trigger that auto-creates a profile, and a SECURITY DEFINER quota
-- function the backend calls to meter + enforce usage atomically.

-- ---------------------------------------------------------------------------
-- profiles: one row per auth user. Holds plan + the rolling usage window.
-- ---------------------------------------------------------------------------
create table if not exists public.profiles (
    id                  uuid primary key references auth.users (id) on delete cascade,
    plan                text        not null default 'free',          -- 'free' | 'pro'
    chars_used          integer     not null default 0,               -- chars used in the current window
    period_started_at   timestamptz not null default now(),          -- start of the rolling 30-day window
    stripe_customer_id  text,                                         -- set on first checkout
    created_at          timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- documents: the user's reading library (pasted text, imported URL/PDF/EPUB).
-- ---------------------------------------------------------------------------
create table if not exists public.documents (
    id          uuid        primary key default gen_random_uuid(),
    user_id     uuid        not null references auth.users (id) on delete cascade,
    title       text        not null default 'Untitled',
    source_type text        not null default 'paste',                 -- 'paste' | 'url' | 'pdf' | 'epub' | 'docx'
    source_url  text,                                                 -- original URL/filename, if any
    content     text        not null,                                 -- extracted plain text
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);
create index if not exists documents_user_id_idx on public.documents (user_id, updated_at desc);

-- ---------------------------------------------------------------------------
-- reading_state: where the user left off + their last voice/speed, per doc.
-- ---------------------------------------------------------------------------
create table if not exists public.reading_state (
    document_id uuid        primary key references public.documents (id) on delete cascade,
    user_id     uuid        not null references auth.users (id) on delete cascade,
    char_offset integer     not null default 0,                       -- resume position in content
    voice       text        not null default 'af_heart',
    speed       real        not null default 1.0,
    updated_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- usage_events: append-only ledger for analytics / abuse investigation.
-- ---------------------------------------------------------------------------
create table if not exists public.usage_events (
    id         bigint generated always as identity primary key,
    user_id    uuid        not null references auth.users (id) on delete cascade,
    chars      integer     not null,
    created_at timestamptz not null default now()
);
create index if not exists usage_events_user_id_idx on public.usage_events (user_id, created_at desc);

-- ---------------------------------------------------------------------------
-- Row Level Security: a user can only touch rows that belong to them.
-- The backend uses the service_role key, which bypasses RLS for trusted ops
-- (quota metering); the client uses the anon key and is fully constrained here.
-- ---------------------------------------------------------------------------
alter table public.profiles      enable row level security;
alter table public.documents     enable row level security;
alter table public.reading_state enable row level security;
alter table public.usage_events  enable row level security;

create policy "own profile"        on public.profiles      for all using (auth.uid() = id)      with check (auth.uid() = id);
create policy "own documents"      on public.documents     for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own reading_state"  on public.reading_state for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
-- usage_events is written by the backend (service_role) and read-only to the user.
create policy "read own usage"     on public.usage_events  for select using (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- Auto-create a profile when a new auth user signs up.
-- ---------------------------------------------------------------------------
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
    insert into public.profiles (id) values (new.id)
    on conflict (id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();

-- ---------------------------------------------------------------------------
-- consume_quota: atomically meter + enforce usage.
--
-- Called by the backend (service_role) before each synthesis. Resets the
-- rolling window if 30 days have passed, then allows the request only if it
-- keeps the user under their plan's monthly character cap. Returns the decision
-- plus the post-charge counters so the API can tell the client what's left.
--
-- Caps live here (DB-side) so they can't be bypassed by a tampered client.
-- ---------------------------------------------------------------------------
create or replace function public.consume_quota(p_user uuid, p_chars integer)
returns table (allowed boolean, plan text, chars_used integer, char_limit integer)
language plpgsql
security definer set search_path = public
as $$
declare
    v_plan       text;
    v_used       integer;
    v_started    timestamptz;
    v_limit      integer;
begin
    select pr.plan, pr.chars_used, pr.period_started_at
      into v_plan, v_used, v_started
      from public.profiles pr
     where pr.id = p_user
       for update;                                  -- row lock: serialize concurrent requests

    if not found then
        -- No profile yet (e.g. trigger race); create one on the fly.
        insert into public.profiles (id) values (p_user)
        on conflict (id) do nothing;
        v_plan := 'free'; v_used := 0; v_started := now();
    end if;

    -- Roll the 30-day window over if it has expired.
    if v_started < now() - interval '30 days' then
        v_used := 0;
        v_started := now();
    end if;

    -- Plan caps (characters / 30-day window). Pro is effectively unlimited but
    -- still capped to guard against runaway cost.
    v_limit := case v_plan
        when 'pro' then 2000000   -- ~2M chars/mo
        else            15000     -- free tier
    end;

    if v_used + p_chars > v_limit then
        -- Reject without charging; persist any window reset that happened above.
        update public.profiles
           set chars_used = v_used, period_started_at = v_started
         where id = p_user;
        return query select false, v_plan, v_used, v_limit;
        return;
    end if;

    -- Charge it.
    update public.profiles
       set chars_used = v_used + p_chars, period_started_at = v_started
     where id = p_user;
    insert into public.usage_events (user_id, chars) values (p_user, p_chars);

    return query select true, v_plan, v_used + p_chars, v_limit;
end;
$$;

-- ---------------------------------------------------------------------------
-- Harden the SECURITY DEFINER functions: they must NOT be callable from the
-- client. consume_quota is invoked only by the trusted backend (service_role);
-- handle_new_user is a signup trigger and should never be reachable via REST RPC.
-- Without this, a client could POST /rest/v1/rpc/consume_quota with another
-- user's id to inflate their usage. (Caught by Supabase's security advisor.)
-- ---------------------------------------------------------------------------
revoke execute on function public.consume_quota(uuid, integer) from anon, authenticated, public;
revoke execute on function public.handle_new_user() from anon, authenticated, public;
