CREATE TABLE IF NOT EXISTS silver.valid_user_data_raw
AS SELECT * FROM public.user_data_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.invalid_user_data_raw
AS SELECT * FROM public.user_data_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_user_data_raw;
TRUNCATE TABLE silver.invalid_user_data_raw;

WITH flagged AS (
    SELECT
        *,
        (user_id IS NULL OR TRIM(user_id) = '') AS bad_user_id,
        (birthdate IS NULL OR TRIM(birthdate) = '') AS bad_birthdate,
        (gender IS NULL OR TRIM(gender) = '') AS bad_gender
    FROM public.user_data_raw
)
INSERT INTO silver.valid_user_data_raw
SELECT *
FROM flagged
WHERE NOT (bad_user_id OR bad_birthdate OR bad_gender);

INSERT INTO silver.invalid_user_data_raw
SELECT *
FROM flagged
WHERE (bad_user_id OR bad_birthdate OR bad_gender);

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'user_data_raw',
    'user_id_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE user_id IS NULL OR TRIM(user_id) = '')
FROM public.user_data_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'user_data_raw',
    'birthdate_not_null',
    'NULL_CHECK',
    'WARNING',
    COUNT(*),
    COUNT(*) FILTER (WHERE birthdate IS NULL OR TRIM(birthdate) = '')
FROM public.user_data_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'user_data_raw',
    'gender_not_null',
    'NULL_CHECK',
    'WARNING',
    COUNT(*),
    COUNT(*) FILTER (WHERE gender IS NULL OR TRIM(gender) = '')
FROM public.user_data_raw;