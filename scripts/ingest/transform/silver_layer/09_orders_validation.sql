-- create valid/invalid tables for each raw orders table

CREATE TABLE IF NOT EXISTS silver.valid_order_data_20200101_20200701_raw
AS SELECT * FROM public.order_data_20200101_20200701_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_order_data_20200101_20200701_raw
AS SELECT * FROM public.order_data_20200101_20200701_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.valid_order_data_20200701_20211001_raw
AS SELECT * FROM public.order_data_20200701_20211001_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_order_data_20200701_20211001_raw
AS SELECT * FROM public.order_data_20200701_20211001_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.valid_order_data_20211001_20220101_raw
AS SELECT * FROM public.order_data_20211001_20220101_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_order_data_20211001_20220101_raw
AS SELECT * FROM public.order_data_20211001_20220101_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.valid_order_data_20220101_20221201_raw
AS SELECT * FROM public.order_data_20220101_20221201_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_order_data_20220101_20221201_raw
AS SELECT * FROM public.order_data_20220101_20221201_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.valid_order_data_20221201_20230601_raw
AS SELECT * FROM public.order_data_20221201_20230601_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_order_data_20221201_20230601_raw
AS SELECT * FROM public.order_data_20221201_20230601_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.valid_order_data_20230601_20240101_raw
AS SELECT * FROM public.order_data_20230601_20240101_raw WITH NO DATA;
CREATE TABLE IF NOT EXISTS silver.invalid_order_data_20230601_20240101_raw
AS SELECT * FROM public.order_data_20230601_20240101_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_order_data_20200101_20200701_raw;
TRUNCATE TABLE silver.invalid_order_data_20200101_20200701_raw;
TRUNCATE TABLE silver.valid_order_data_20200701_20211001_raw;
TRUNCATE TABLE silver.invalid_order_data_20200701_20211001_raw;
TRUNCATE TABLE silver.valid_order_data_20211001_20220101_raw;
TRUNCATE TABLE silver.invalid_order_data_20211001_20220101_raw;
TRUNCATE TABLE silver.valid_order_data_20220101_20221201_raw;
TRUNCATE TABLE silver.invalid_order_data_20220101_20221201_raw;
TRUNCATE TABLE silver.valid_order_data_20221201_20230601_raw;
TRUNCATE TABLE silver.invalid_order_data_20221201_20230601_raw;
TRUNCATE TABLE silver.valid_order_data_20230601_20240101_raw;
TRUNCATE TABLE silver.invalid_order_data_20230601_20240101_raw;

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (user_id IS NULL OR TRIM(user_id) = '') AS bad_user_id,
        ("estimated arrival" IS NULL OR TRIM("estimated arrival") = '') AS bad_est_arrival,
        (transaction_date IS NULL OR TRIM(transaction_date) = '') AS bad_txn_date
)
INSERT INTO silver.valid_order_data_20200101_20200701_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date)
FROM public.order_data_20200101_20200701_raw flagged;

INSERT INTO silver.invalid_order_data_20200101_20200701_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date)
FROM public.order_data_20200101_20200701_raw flagged;

-- repeat the flagged pattern per table:

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (user_id IS NULL OR TRIM(user_id) = '') AS bad_user_id,
        ("estimated arrival" IS NULL OR TRIM("estimated arrival") = '') AS bad_est_arrival,
        (transaction_date IS NULL OR TRIM(transaction_date) = '') AS bad_txn_date
    FROM public.order_data_20200701_20211001_raw
)
INSERT INTO silver.valid_order_data_20200701_20211001_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date);

INSERT INTO silver.invalid_order_data_20200701_20211001_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date);

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (user_id IS NULL OR TRIM(user_id) = '') AS bad_user_id,
        ("estimated arrival" IS NULL OR TRIM("estimated arrival") = '') AS bad_est_arrival,
        (transaction_date IS NULL OR TRIM(transaction_date) = '') AS bad_txn_date
    FROM public.order_data_20211001_20220101_raw
)
INSERT INTO silver.valid_order_data_20211001_20220101_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date);

INSERT INTO silver.invalid_order_data_20211001_20220101_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date);

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (user_id IS NULL OR TRIM(user_id) = '') AS bad_user_id,
        ("estimated arrival" IS NULL OR TRIM("estimated arrival") = '') AS bad_est_arrival,
        (transaction_date IS NULL OR TRIM(transaction_date) = '') AS bad_txn_date
    FROM public.order_data_20220101_20221201_raw
)
INSERT INTO silver.valid_order_data_20220101_20221201_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date);

INSERT INTO silver.invalid_order_data_20220101_20221201_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date);

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (user_id IS NULL OR TRIM(user_id) = '') AS bad_user_id,
        ("estimated arrival" IS NULL OR TRIM("estimated arrival") = '') AS bad_est_arrival,
        (transaction_date IS NULL OR TRIM(transaction_date) = '') AS bad_txn_date
    FROM public.order_data_20221201_20230601_raw
)
INSERT INTO silver.valid_order_data_20221201_20230601_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date);

INSERT INTO silver.invalid_order_data_20221201_20230601_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date);

WITH flagged AS (
    SELECT
        *,
        (order_id IS NULL OR TRIM(order_id) = '') AS bad_order_id,
        (user_id IS NULL OR TRIM(user_id) = '') AS bad_user_id,
        ("estimated arrival" IS NULL OR TRIM("estimated arrival") = '') AS bad_est_arrival,
        (transaction_date IS NULL OR TRIM(transaction_date) = '') AS bad_txn_date
    FROM public.order_data_20230601_20240101_raw
)
INSERT INTO silver.valid_order_data_20230601_20240101_raw
SELECT * FROM flagged
WHERE NOT (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date);

INSERT INTO silver.invalid_order_data_20230601_20240101_raw
SELECT * FROM flagged
WHERE (bad_order_id OR bad_user_id OR bad_est_arrival OR bad_txn_date);

-- DQ summary per orders table (same rules, just aggregated)

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'order_data_20200101_20200701_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR user_id IS NULL OR TRIM(user_id) = ''
           OR "estimated arrival" IS NULL OR TRIM("estimated arrival") = ''
           OR transaction_date IS NULL OR TRIM(transaction_date) = ''
    )
FROM public.order_data_20200101_20200701_raw;

INSERT INTO silver.dq_summary
(table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'order_data_20200701_20211001_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR user_id IS NULL OR TRIM(user_id) = ''
           OR "estimated arrival" IS NULL OR TRIM("estimated arrival") = ''
           OR transaction_date IS NULL OR TRIM(transaction_date) = ''
    )
FROM public.order_data_20200701_20211001_raw;

INSERT INTO silver.dq_summary
(table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'order_data_20211001_20220101_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR user_id IS NULL OR TRIM(user_id) = ''
           OR "estimated arrival" IS NULL OR TRIM("estimated arrival") = ''
           OR transaction_date IS NULL OR TRIM(transaction_date) = ''
    )
FROM public.order_data_20211001_20220101_raw;

INSERT INTO silver.dq_summary
(table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'order_data_20220101_20221201_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR user_id IS NULL OR TRIM(user_id) = ''
           OR "estimated arrival" IS NULL OR TRIM("estimated arrival") = ''
           OR transaction_date IS NULL OR TRIM(transaction_date) = ''
    )
FROM public.order_data_20220101_20221201_raw;

INSERT INTO silver.dq_summary
(table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'order_data_20221201_20230601_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR user_id IS NULL OR TRIM(user_id) = ''
           OR "estimated arrival" IS NULL OR TRIM("estimated arrival") = ''
           OR transaction_date IS NULL OR TRIM(transaction_date) = ''
    )
FROM public.order_data_20221201_20230601_raw;

INSERT INTO silver.dq_summary
(table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'order_data_20230601_20240101_raw',
    'required_fields_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (
        WHERE order_id IS NULL OR TRIM(order_id) = ''
           OR user_id IS NULL OR TRIM(user_id) = ''
           OR "estimated arrival" IS NULL OR TRIM("estimated arrival") = ''
           OR transaction_date IS NULL OR TRIM(transaction_date) = ''
    )
FROM public.order_data_20230601_20240101_raw;