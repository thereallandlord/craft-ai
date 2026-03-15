-- Craft AI v4 — SaaS Auth: multi-provider auth, usage tracking, subscriptions
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New Query)

-- ============================================================
-- 1. auth_accounts — unified identity hub
-- ============================================================
CREATE TABLE IF NOT EXISTS auth_accounts (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email           TEXT UNIQUE,
    telegram_id     BIGINT UNIQUE,
    google_id       TEXT UNIQUE,
    display_name    TEXT,
    avatar_url      TEXT DEFAULT '',
    plan            TEXT DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'business')),
    is_club_member  BOOLEAN DEFAULT false,
    club_checked_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_auth_accounts_email ON auth_accounts(email);
CREATE INDEX IF NOT EXISTS idx_auth_accounts_telegram_id ON auth_accounts(telegram_id);
CREATE INDEX IF NOT EXISTS idx_auth_accounts_plan ON auth_accounts(plan);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_auth_accounts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_auth_accounts_updated_at
    BEFORE UPDATE ON auth_accounts
    FOR EACH ROW
    EXECUTE FUNCTION update_auth_accounts_updated_at();

-- ============================================================
-- 2. Add auth_id FK to existing users table
-- ============================================================
ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_id UUID REFERENCES auth_accounts(id);
CREATE INDEX IF NOT EXISTS idx_users_auth_id ON users(auth_id);

-- ============================================================
-- 3. Backfill: create auth_accounts for existing Telegram users
-- ============================================================
INSERT INTO auth_accounts (id, telegram_id, display_name, avatar_url, plan, is_club_member)
SELECT
    gen_random_uuid(),
    CAST(user_id AS BIGINT),
    first_name,
    COALESCE(photo_url, ''),
    COALESCE(plan, 'free'),
    false
FROM users
WHERE CAST(user_id AS BIGINT) NOT IN (SELECT telegram_id FROM auth_accounts WHERE telegram_id IS NOT NULL)
ON CONFLICT (telegram_id) DO NOTHING;

-- Link users to their auth_accounts
UPDATE users
SET auth_id = auth_accounts.id
FROM auth_accounts
WHERE CAST(users.user_id AS BIGINT) = auth_accounts.telegram_id
  AND users.auth_id IS NULL;

-- ============================================================
-- 4. usage_tracking — daily feature usage per user
-- ============================================================
CREATE TABLE IF NOT EXISTS usage_tracking (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    auth_id     UUID REFERENCES auth_accounts(id) ON DELETE CASCADE,
    anon_token  TEXT,
    action_type TEXT NOT NULL,
    date        DATE NOT NULL DEFAULT CURRENT_DATE,
    count       INT DEFAULT 0,
    UNIQUE (auth_id, action_type, date),
    UNIQUE (anon_token, action_type, date)
);

CREATE INDEX IF NOT EXISTS idx_usage_tracking_auth_date ON usage_tracking(auth_id, date);
CREATE INDEX IF NOT EXISTS idx_usage_tracking_anon_date ON usage_tracking(anon_token, date);

-- ============================================================
-- 5. subscriptions — payment/plan tracking (for lava.top, Phase 2)
-- ============================================================
CREATE TABLE IF NOT EXISTS subscriptions (
    id                      UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    auth_id                 UUID NOT NULL REFERENCES auth_accounts(id) ON DELETE CASCADE,
    plan                    TEXT NOT NULL CHECK (plan IN ('pro', 'business')),
    status                  TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'cancelled', 'past_due')),
    lava_subscription_id    TEXT UNIQUE,
    amount                  INT,
    currency                TEXT DEFAULT 'RUB',
    period_start            TIMESTAMPTZ,
    period_end              TIMESTAMPTZ,
    created_at              TIMESTAMPTZ DEFAULT now(),
    updated_at              TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_auth_id ON subscriptions(auth_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_subscriptions_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_subscriptions_updated_at
    BEFORE UPDATE ON subscriptions
    FOR EACH ROW
    EXECUTE FUNCTION update_subscriptions_updated_at();

-- ============================================================
-- 6. referral_codes + referrals (future, empty for now)
-- ============================================================
CREATE TABLE IF NOT EXISTS referral_codes (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    auth_id     UUID NOT NULL REFERENCES auth_accounts(id) ON DELETE CASCADE,
    code        TEXT UNIQUE NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS referrals (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    referrer_id UUID NOT NULL REFERENCES auth_accounts(id) ON DELETE CASCADE,
    referred_id UUID NOT NULL REFERENCES auth_accounts(id) ON DELETE CASCADE,
    code_used   TEXT REFERENCES referral_codes(code),
    status      TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'converted', 'rewarded')),
    created_at  TIMESTAMPTZ DEFAULT now()
);
