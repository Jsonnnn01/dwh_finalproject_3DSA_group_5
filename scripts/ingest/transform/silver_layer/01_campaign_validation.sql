CREATE TABLE IF NOT EXISTS silver.valid_campaign_data_raw
AS SELECT * FROM public.campaign_data_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.invalid_campaign_data_raw
AS SELECT * FROM public.campaign_data_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_campaign_data_raw;
TRUNCATE TABLE silver.invalid_campaign_data_raw;

WITH flagged AS (
    SELECT
        *,
        (campaign_id IS NULL OR TRIM(campaign_id) = '') AS bad_campaign_id,
        (campaign_name IS NULL OR TRIM(campaign_name) = '') AS bad_campaign_name,
        (
            discount IS NULL
            OR TRIM(discount) = ''
            OR REGEXP_REPLACE(discount, '[^0-9]', '', 'g') = ''
        ) AS bad_discount
    FROM public.campaign_data_raw
)
INSERT INTO silver.valid_campaign_data_raw
SELECT *
FROM flagged
WHERE NOT (bad_campaign_id OR bad_campaign_name OR bad_discount);

INSERT INTO silver.invalid_campaign_data_raw
SELECT *
FROM flagged
WHERE (bad_campaign_id OR bad_campaign_name OR bad_discount);

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'campaign_data_raw',
    'campaign_id_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE campaign_id IS NULL OR TRIM(campaign_id) = '')
FROM public.campaign_data_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'campaign_data_raw',
    'campaign_name_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE campaign_name IS NULL OR TRIM(campaign_name) = '')
FROM public.campaign_data_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'campaign_data_raw',
    'discount_has_digits',
    'FORMAT_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE discount IS NULL
           OR TRIM(discount) = ''
           OR REGEXP_REPLACE(discount, '[^0-9]', '', 'g') = ''
    )
FROM public.campaign_data_raw;