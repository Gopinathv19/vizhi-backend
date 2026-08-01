-- Migration: add_budget_enforcement
-- Adds optional per-agent and per-model-token spending limits.
--
-- budget_usd      – maximum cumulative USD cost allowed (NULL = unlimited)
-- budget_tokens   – maximum cumulative total tokens allowed (NULL = unlimited)
-- budget_reset_at – when the budget window resets (NULL = never / lifetime)
--
-- Spent-so-far figures are derived live from the queries/responses tables
-- so they stay accurate even if rows are backfilled.

-- ── agents ──────────────────────────────────────────────────────────────
ALTER TABLE agents ADD COLUMN budget_usd        REAL    DEFAULT NULL;
ALTER TABLE agents ADD COLUMN budget_tokens     INTEGER DEFAULT NULL;
ALTER TABLE agents ADD COLUMN budget_reset_at   TEXT    DEFAULT NULL;

-- ── model_connections ────────────────────────────────────────────────────
ALTER TABLE model_connections ADD COLUMN budget_usd        REAL    DEFAULT NULL;
ALTER TABLE model_connections ADD COLUMN budget_tokens     INTEGER DEFAULT NULL;
ALTER TABLE model_connections ADD COLUMN budget_reset_at   TEXT    DEFAULT NULL;
