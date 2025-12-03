CREATE TABLE IF NOT EXISTS silver.valid_line_item_prices1_raw
AS SELECT * FROM public.line_item_prices1_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_line_item_prices1_raw
AS SELECT * FROM public.line_item_prices1_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.valid_line_item_prices2_raw
AS SELECT * FROM public.line_item_prices2_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_line_item_prices2_raw
AS SELECT * FROM public.line_item_prices2_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.valid_line_item_prices3_raw
AS SELECT * FROM public.line_item_prices3_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_line_item_prices3_raw
AS SELECT * FROM public.line_item_prices3_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_line_item_prices1_raw;
TRUNCATE TABLE silver.invalid_line_item_prices1_raw;
TRUNCATE TABLE silver.valid_line_item_prices2_raw;
TRUNCATE TABLE silver.invalid_line_item_prices2_raw;
TRUNCATE TABLE silver.valid_line_item_prices3_raw;
TRUNCATE TABLE silver.invalid_line_item_prices3_raw;

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (price IS NULL) AS bad_price,
        (quantity IS NULL OR TRIM(quantity) = '') AS bad_quantity
    FROM public.line_item_prices1_raw
)
INSERT INTO silver.valid_line_item_prices1_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_price OR bad_quantity);

INSERT INTO silver.invalid_line_item_prices1_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_price OR bad_quantity);

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (price IS NULL) AS bad_price,
        (quantity IS NULL OR TRIM(quantity) = '') AS bad_quantity
    FROM public.line_item_prices2_raw
)
INSERT INTO silver.valid_line_item_prices2_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_price OR bad_quantity);

INSERT INTO silver.invalid_line_item_prices2_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_price OR bad_quantity);

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (price IS NULL) AS bad_price,
        (quantity IS NULL OR TRIM(quantity) = '') AS bad_quantity
    FROM public.line_item_prices3_raw
)
INSERT INTO silver.valid_line_item_prices3_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_price OR bad_quantity);

INSERT INTO silver.invalid_line_item_prices3_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_price OR bad_quantity);

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'line_item_prices1_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR price IS NULL
           OR quantity IS NULL OR TRIM(quantity) = ''
    )
FROM public.line_item_prices1_raw;

INSERT INTO silver.dq_summary
(table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'line_item_prices2_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR price IS NULL
           OR quantity IS NULL OR TRIM(quantity) = ''
    )
FROM public.line_item_prices2_raw;

INSERT INTO silver.dq_summary
(table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'line_item_prices3_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR price IS NULL
           OR quantity IS NULL OR TRIM(quantity) = ''
    )
FROM public.line_item_prices3_raw;