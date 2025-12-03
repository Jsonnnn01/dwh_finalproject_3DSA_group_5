CREATE TABLE IF NOT EXISTS silver.valid_user_job_raw
AS SELECT * FROM public.user_job_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.invalid_user_job_raw
AS SELECT * FROM public.user_job_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_user_job_raw;
TRUNCATE TABLE silver.invalid_user_job_raw;

WITH flagged AS (
    SELECT
        *,
        (user_id IS NULL OR TRIM(user_id) = '') AS bad_user_id,
        (job_title IS NULL OR TRIM(job_title) = '') AS bad_job_title
    FROM public.user_job_raw
)
INSERT INTO silver.valid_user_job_raw
SELECT *
FROM flagged
WHERE NOT (bad_user_id OR bad_job_title);

INSERT INTO silver.invalid_user_job_raw
SELECT *
FROM flagged
WHERE (bad_user_id OR bad_job_title);

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'user_job_raw',
    'user_id_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE user_id IS NULL OR TRIM(user_id) = '')
FROM public.user_job_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'user_job_raw',
    'job_title_not_null',
    'NULL_CHECK',
    'WARNING',
    COUNT(*),
    COUNT(*) FILTER (WHERE job_title IS NULL OR TRIM(job_title) = '')
FROM public.user_job_raw;