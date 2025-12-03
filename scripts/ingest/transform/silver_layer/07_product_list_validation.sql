CREATE TABLE IF NOT EXISTS silver.valid_product_list_raw
AS SELECT * FROM public.product_list_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.invalid_product_list_raw
AS SELECT * FROM public.product_list_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_product_list_raw;
TRUNCATE TABLE silver.invalid_product_list_raw;

WITH flagged AS (
    SELECT
        *,
        (product_id IS NULL OR TRIM(product_id) = '') AS bad_product_id,
        (product_name IS NULL OR TRIM(product_name) = '') AS bad_product_name,
        (price IS NULL) AS bad_price
    FROM public.product_list_raw
)
INSERT INTO silver.valid_product_list_raw
SELECT *
FROM flagged
WHERE NOT (bad_product_id OR bad_product_name OR bad_price);

INSERT INTO silver.invalid_product_list_raw
SELECT *
FROM flagged
WHERE (bad_product_id OR bad_product_name OR bad_price);

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'product_list_raw',
    'product_id_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE product_id IS NULL OR TRIM(product_id) = '')
FROM public.product_list_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'product_list_raw',
    'product_name_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE product_name IS NULL OR TRIM(product_name) = '')
FROM public.product_list_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'product_list_raw',
    'price_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE price IS NULL)
FROM public.product_list_raw;