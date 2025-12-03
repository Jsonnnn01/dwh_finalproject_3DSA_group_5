CREATE OR REPLACE VIEW silver.silver_dq_summary AS
SELECT
    table_name,
    rule_name,
    issue_type,
    severity,
    total_rows,
    failed_rows,
    CASE
        WHEN total_rows = 0 THEN 0
        ELSE failed_rows::numeric / total_rows::numeric
    END AS failed_ratio,
    created_at
FROM silver.dq_summary;