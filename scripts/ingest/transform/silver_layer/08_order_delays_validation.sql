CREATE TABLE IF NOT EXISTS silver.valid_order_delays_raw
AS SELECT * FROM public.order_delays_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.invalid_order_delays_raw
AS SELECT * FROM public.order_delays_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_order_delays_raw;
TRUNCATE TABLE silver.invalid_order_delays_raw;

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (
            "delay in days" IS NULL
            OR TRIM("delay in days"::text) = ''
            OR REGEXP_REPLACE("delay in days"::text, '[^0-9]', '', 'g') = ''
        ) AS bad_delay
    FROM public.order_delays_raw
)
INSERT INTO silver.valid_order_delays_raw
SELECT *
FROM flagged
WHERE NOT (bad_order_id OR bad_delay);

INSERT INTO silver.invalid_order_delays_raw
SELECT *
FROM flagged
WHERE (bad_order_id OR bad_delay);

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'order_delays_raw',
    'order_id_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE order_id IS NULL OR TRIM(order_id) = '')
FROM public.order_delays_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'order_delays_raw',
    'delay_in_days_numeric',
    'FORMAT_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE "delay in days" IS NULL
           OR TRIM("delay in days"::text) = ''
           OR REGEXP_REPLACE("delay in days"::text, '[^0-9]', '', 'g') = ''
    )
FROM public.order_delays_raw;