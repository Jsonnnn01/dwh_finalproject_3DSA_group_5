CREATE TABLE IF NOT EXISTS silver.valid_user_credit_card_raw
AS SELECT * FROM public.user_credit_card_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.invalid_user_credit_card_raw
AS SELECT * FROM public.user_credit_card_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_user_credit_card_raw;
TRUNCATE TABLE silver.invalid_user_credit_card_raw;

WITH flagged AS (
    SELECT
        *,
        (user_id IS NULL OR TRIM(user_id) = '') AS bad_user_id,
        (credit_card_number IS NULL) AS bad_cc_number,
        (issuing_bank IS NULL OR TRIM(issuing_bank) = '') AS bad_bank
    FROM public.user_credit_card_raw
)
INSERT INTO silver.valid_user_credit_card_raw
SELECT *
FROM flagged
WHERE NOT (bad_user_id OR bad_cc_number OR bad_bank);

INSERT INTO silver.invalid_user_credit_card_raw
SELECT *
FROM flagged
WHERE (bad_user_id OR bad_cc_number OR bad_bank);

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'user_credit_card_raw',
    'user_id_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE user_id IS NULL OR TRIM(user_id) = '')
FROM public.user_credit_card_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'user_credit_card_raw',
    'credit_card_number_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE credit_card_number IS NULL)
FROM public.user_credit_card_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'user_credit_card_raw',
    'issuing_bank_not_null',
    'NULL_CHECK',
    'WARNING',
    COUNT(*),
    COUNT(*) FILTER (WHERE issuing_bank IS NULL OR TRIM(issuing_bank) = '')
FROM public.user_credit_card_raw;