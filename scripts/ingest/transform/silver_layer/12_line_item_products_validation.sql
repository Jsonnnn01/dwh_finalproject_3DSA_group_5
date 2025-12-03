CREATE TABLE IF NOT EXISTS silver.valid_line_item_products1_raw
AS SELECT * FROM public.line_item_products1_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_line_item_products1_raw
AS SELECT * FROM public.line_item_products1_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.valid_line_item_products2_raw
AS SELECT * FROM public.line_item_products2_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_line_item_products2_raw
AS SELECT * FROM public.line_item_products2_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.valid_line_item_products3_raw
AS SELECT * FROM public.line_item_products3_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_line_item_products3_raw
AS SELECT * FROM public.line_item_products3_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_line_item_products1_raw;
TRUNCATE TABLE silver.invalid_line_item_products1_raw;
TRUNCATE TABLE silver.valid_line_item_products2_raw;
TRUNCATE TABLE silver.invalid_line_item_products2_raw;
TRUNCATE TABLE silver.valid_line_item_products3_raw;
TRUNCATE TABLE silver.invalid_line_item_products3_raw;

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (product_id IS NULL OR TRIM(product_id) = '') AS bad_product_id,
        (product_name IS NULL OR TRIM(product_name) = '') AS bad_product_name
    FROM public.line_item_products1_raw
)
INSERT INTO silver.valid_line_item_products1_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_product_id OR bad_product_name);

INSERT INTO silver.invalid_line_item_products1_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_product_id OR bad_product_name);

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (product_id IS NULL OR TRIM(product_id) = '') AS bad_product_id,
        (product_name IS NULL OR TRIM(product_name) = '') AS bad_product_name
    FROM public.line_item_products2_raw
)
INSERT INTO silver.valid_line_item_products2_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_product_id OR bad_product_name);

INSERT INTO silver.invalid_line_item_products2_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_product_id OR bad_product_name);

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (product_id IS NULL OR TRIM(product_id) = '') AS bad_product_id,
        (product_name IS NULL OR TRIM(product_name) = '') AS bad_product_name
    FROM public.line_item_products3_raw
)
INSERT INTO silver.valid_line_item_products3_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_product_id OR bad_product_name);

INSERT INTO silver.invalid_line_item_products3_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_product_id OR bad_product_name);

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'line_item_products1_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR product_id IS NULL OR TRIM(product_id) = ''
           OR product_name IS NULL OR TRIM(product_name) = ''
    )
FROM public.line_item_products1_raw;

INSERT INTO silver.dq_summary
(table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'line_item_products2_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR product_id IS NULL OR TRIM(product_id) = ''
           OR product_name IS NULL OR TRIM(product_name) = ''
    )
FROM public.line_item_products2_raw;

INSERT INTO silver.dq_summary
(table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'line_item_products3_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR product_id IS NULL OR TRIM(product_id) = ''
           OR product_name IS NULL OR TRIM(product_name) = ''
    )
FROM public.line_item_products3_raw;