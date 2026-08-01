-- ============================================================
-- Remote Supabase Schema Sync Migration
-- Run this ONCE in the Supabase SQL Editor to bring the remote
-- schema up to date with the local SQLite schema.
-- ============================================================

-- ── agents: add missing columns ──────────────────────────────
ALTER TABLE agents ADD COLUMN IF NOT EXISTS token_name    TEXT        DEFAULT NULL;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS last_used_at  TIMESTAMP   DEFAULT NULL;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS budget_usd    REAL        DEFAULT NULL;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS budget_tokens INTEGER     DEFAULT NULL;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS budget_reset_at TEXT      DEFAULT NULL;

-- ── model_connections: add missing columns ───────────────────
ALTER TABLE model_connections ADD COLUMN IF NOT EXISTS token_name             TEXT        DEFAULT NULL;
ALTER TABLE model_connections ADD COLUMN IF NOT EXISTS last_used_at           TIMESTAMP   DEFAULT NULL;
ALTER TABLE model_connections ADD COLUMN IF NOT EXISTS total_tokens_consumed  INTEGER     DEFAULT 0;
ALTER TABLE model_connections ADD COLUMN IF NOT EXISTS total_cost             REAL        DEFAULT 0.0;
ALTER TABLE model_connections ADD COLUMN IF NOT EXISTS budget_usd             REAL        DEFAULT NULL;
ALTER TABLE model_connections ADD COLUMN IF NOT EXISTS budget_tokens          INTEGER     DEFAULT NULL;
ALTER TABLE model_connections ADD COLUMN IF NOT EXISTS budget_reset_at        TEXT        DEFAULT NULL;

-- ── auth_accounts: ensure unique constraints exist ───────────
-- (These may already exist — IF NOT EXISTS handles it safely)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_auth_provider_user'
    ) THEN
        ALTER TABLE auth_accounts
            ADD CONSTRAINT uq_auth_provider_user
            UNIQUE (provider, provider_user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_auth_user_provider'
    ) THEN
        ALTER TABLE auth_accounts
            ADD CONSTRAINT uq_auth_user_provider
            UNIQUE (user_id, provider);
    END IF;
END $$;
