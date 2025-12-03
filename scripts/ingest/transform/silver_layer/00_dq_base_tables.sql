CREATE SCHEMA IF NOT EXISTS silver;

CREATE TABLE IF NOT EXISTS silver.dq_summary (
    id          bigserial PRIMARY KEY,
    table_name  text        NOT NULL,
    rule_name   text        NOT NULL,
    issue_type  text        NOT NULL, -- 'NULL_CHECK', 'FORMAT_CHECK'
    severity    text        NOT NULL, -- 'ERROR', 'WARNING'
    total_rows  bigint      NOT NULL,
    failed_rows bigint      NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

TRUNCATE TABLE silver.dq_summary;