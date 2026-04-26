CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    username        TEXT        NOT NULL UNIQUE,
    hashed_password TEXT        NOT NULL,
    role            TEXT        NOT NULL DEFAULT 'user',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS persons (
    id           SERIAL PRIMARY KEY,
    real_name    TEXT        NOT NULL UNIQUE,
    anonymous_id TEXT        NOT NULL UNIQUE,
    person_type  TEXT        NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS daily_revenue (
    id            SERIAL PRIMARY KEY,
    source_uid    TEXT          NOT NULL UNIQUE,
    content_hash  TEXT          NOT NULL,
    date          DATE          NOT NULL,
    total_revenue NUMERIC(12,2) NOT NULL,
    card_revenue  NUMERIC(12,2),
    cash_revenue  NUMERIC(12,2),
    created_at    TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS expenses (
    id           SERIAL PRIMARY KEY,
    source_uid   TEXT          NOT NULL UNIQUE,
    content_hash TEXT          NOT NULL,
    date         DATE          NOT NULL,
    category     TEXT          NOT NULL,
    amount       NUMERIC(12,2) NOT NULL,
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS amortization (
    id              SERIAL PRIMARY KEY,
    source_uid      TEXT          NOT NULL UNIQUE,
    content_hash    TEXT          NOT NULL,
    date            DATE          NOT NULL,
    asset_name      TEXT          NOT NULL,
    total_amount    NUMERIC(12,2) NOT NULL,
    duration_months INTEGER       NOT NULL,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS specialist_capacity (
    id              SERIAL PRIMARY KEY,
    source_uid      TEXT         NOT NULL UNIQUE,
    content_hash    TEXT         NOT NULL,
    date            DATE         NOT NULL,
    person          TEXT         NOT NULL REFERENCES persons(anonymous_id) ON DELETE RESTRICT,
    available_hours NUMERIC(6,2) NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS specialist_activity (
    id            SERIAL PRIMARY KEY,
    source_uid    TEXT        NOT NULL UNIQUE,
    content_hash  TEXT        NOT NULL,
    date          DATE        NOT NULL,
    person        TEXT        NOT NULL REFERENCES persons(anonymous_id) ON DELETE RESTRICT,
    units         INTEGER,
    activity_type TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS specialist_payouts (
    id                SERIAL PRIMARY KEY,
    source_uid        TEXT          NOT NULL UNIQUE,
    content_hash      TEXT          NOT NULL,
    date              DATE          NOT NULL,
    person            TEXT          NOT NULL REFERENCES persons(anonymous_id) ON DELETE RESTRICT,
    payout_amount     NUMERIC(12,2) NOT NULL,
    generated_revenue NUMERIC(12,2),
    created_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ   NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS monthly_metrics (
    id           SERIAL PRIMARY KEY,
    metric_uid   TEXT           NOT NULL UNIQUE,
    month        DATE           NOT NULL,
    metric_name  TEXT           NOT NULL,
    metric_value NUMERIC(14,4),
    person       TEXT           REFERENCES persons(anonymous_id) ON DELETE SET NULL,
    category     TEXT,
    created_at   TIMESTAMPTZ    NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ    NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS chat_messages (
    id         SERIAL PRIMARY KEY,
    session_id TEXT        NOT NULL,
    role       TEXT        NOT NULL,
    content    TEXT        NOT NULL,
    username   TEXT        NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);


CREATE INDEX IF NOT EXISTS idx_daily_revenue_date        ON daily_revenue (date);
CREATE INDEX IF NOT EXISTS idx_expenses_date             ON expenses (date);
CREATE INDEX IF NOT EXISTS idx_amortization_date         ON amortization (date);
CREATE INDEX IF NOT EXISTS idx_spec_capacity_date        ON specialist_capacity (date);
CREATE INDEX IF NOT EXISTS idx_spec_capacity_person      ON specialist_capacity (person);
CREATE INDEX IF NOT EXISTS idx_spec_activity_date        ON specialist_activity (date);
CREATE INDEX IF NOT EXISTS idx_spec_activity_person      ON specialist_activity (person);
CREATE INDEX IF NOT EXISTS idx_spec_payouts_date         ON specialist_payouts (date);
CREATE INDEX IF NOT EXISTS idx_spec_payouts_person       ON specialist_payouts (person);
CREATE INDEX IF NOT EXISTS idx_monthly_metrics_month     ON monthly_metrics (month);
CREATE INDEX IF NOT EXISTS idx_monthly_metrics_person    ON monthly_metrics (person);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session     ON chat_messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_username    ON chat_messages (username);


CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'daily_revenue', 'expenses', 'amortization',
        'specialist_capacity', 'specialist_activity', 'specialist_payouts',
        'monthly_metrics'
    ]
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_updated_at ON %I;
             CREATE TRIGGER trg_updated_at
             BEFORE UPDATE ON %I
             FOR EACH ROW EXECUTE FUNCTION update_updated_at();',
            t, t
        );
    END LOOP;
END;
$$;
