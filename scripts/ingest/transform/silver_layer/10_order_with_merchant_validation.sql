CREATE TABLE IF NOT EXISTS silver.valid_order_with_merchant_data1_raw
AS SELECT * FROM public.order_with_merchant_data1_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_order_with_merchant_data1_raw
AS SELECT * FROM public.order_with_merchant_data1_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.valid_order_with_merchant_data2_raw
AS SELECT * FROM public.order_with_merchant_data2_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_order_with_merchant_data2_raw
AS SELECT * FROM public.order_with_merchant_data2_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.valid_order_with_merchant_data3_raw
AS SELECT * FROM public.order_with_merchant_data3_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_order_with_merchant_data3_raw
AS SELECT * FROM public.order_with_merchant_data3_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_order_with_merchant_data1_raw;
TRUNCATE TABLE silver.invalid_order_with_merchant_data1_raw;
TRUNCATE TABLE silver.valid_order_with_merchant_data2_raw;
TRUNCATE TABLE silver.invalid_order_with_merchant_data2_raw;
TRUNCATE TABLE silver.valid_order_with_merchant_data3_raw;
TRUNCATE TABLE silver.invalid_order_with_merchant_data3_raw;

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (merchant_id IS NULL OR TRIM(merchant_id) = '') AS bad_merchant_id,
        (staff_id IS NULL OR TRIM(staff_id) = '') AS bad_staff_id
    FROM public.order_with_merchant_data1_raw
)
INSERT INTO silver.valid_order_with_merchant_data1_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_merchant_id OR bad_staff_id);

INSERT INTO silver.invalid_order_with_merchant_data1_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_merchant_id OR bad_staff_id);

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (merchant_id IS NULL OR TRIM(merchant_id) = '') AS bad_merchant_id,
        (staff_id IS NULL OR TRIM(staff_id) = '') AS bad_staff_id
    FROM public.order_with_merchant_data2_raw
)
INSERT INTO silver.valid_order_with_merchant_data2_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_merchant_id OR bad_staff_id);

INSERT INTO silver.invalid_order_with_merchant_data2_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_merchant_id OR bad_staff_id);

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (merchant_id IS NULL OR TRIM(merchant_id) = '') AS bad_merchant_id,
        (staff_id IS NULL OR TRIM(staff_id) = '') AS bad_staff_id
    FROM public.order_with_merchant_data3_raw
)
INSERT INTO silver.valid_order_with_merchant_data3_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_merchant_id OR bad_staff_id);

INSERT INTO silver.invalid_order_with_merchant_data3_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_merchant_id OR bad_staff_id);

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'order_with_merchant_data1_raw',
    'required_ids_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR merchant_id IS NULL OR TRIM(merchant_id) = ''
           OR staff_id IS NULL OR TRIM(staff_id) = ''
    )
FROM public.order_with_merchant_data1_raw;

INSERT INTO silver.dq_summary
(table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'order_with_merchant_data2_raw',
    'required_ids_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR merchant_id IS NULL OR TRIM(merchant_id) = ''
           OR staff_id IS NULL OR TRIM(staff_id) = ''
    )
FROM public.order_with_merchant_data2_raw;

INSERT INTO silver.dq_summary
(table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'order_with_merchant_data3_raw',
    'required_ids_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR merchant_id IS NULL OR TRIM(merchant_id) = ''
           OR staff_id IS NULL OR TRIM(staff_id) = ''
    )
FROM public.order_with_merchant_data3_raw;