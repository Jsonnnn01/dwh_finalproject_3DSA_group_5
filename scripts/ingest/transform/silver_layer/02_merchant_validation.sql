CREATE TABLE IF NOT EXISTS silver.valid_merchant_data_raw
AS SELECT * FROM public.merchant_data_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.invalid_merchant_data_raw
AS SELECT * FROM public.merchant_data_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_merchant_data_raw;
TRUNCATE TABLE silver.invalid_merchant_data_raw;

WITH flagged AS (
    SELECT
        *,
        (merchant_id IS NULL OR TRIM(merchant_id) = '') AS bad_merchant_id,
        (name IS NULL OR TRIM(name) = '') AS bad_name,
        (country IS NULL OR TRIM(country) = '') AS bad_country
    FROM public.merchant_data_raw
)
INSERT INTO silver.valid_merchant_data_raw
SELECT *
FROM flagged
WHERE NOT (bad_merchant_id OR bad_name OR bad_country);

INSERT INTO silver.invalid_merchant_data_raw
SELECT *
FROM flagged
WHERE (bad_merchant_id OR bad_name OR bad_country);

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'merchant_data_raw',
    'merchant_id_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE merchant_id IS NULL OR TRIM(merchant_id) = '')
FROM public.merchant_data_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'merchant_data_raw',
    'name_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE name IS NULL OR TRIM(name) = '')
FROM public.merchant_data_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'merchant_data_raw',
    'country_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE country IS NULL OR TRIM(country) = '')
FROM public.merchant_data_raw;