CREATE TABLE IF NOT EXISTS silver.valid_staff_data_raw
AS SELECT * FROM public.staff_data_raw WITH NO DATA;

CREATE TABLE IF NOT EXISTS silver.invalid_staff_data_raw
AS SELECT * FROM public.staff_data_raw WITH NO DATA;

TRUNCATE TABLE silver.valid_staff_data_raw;
TRUNCATE TABLE silver.invalid_staff_data_raw;

WITH flagged AS (
    SELECT
        *,
        (staff_id IS NULL OR TRIM(staff_id) = '') AS bad_staff_id,
        (name IS NULL OR TRIM(name) = '') AS bad_name,
        (job_level IS NULL OR TRIM(job_level) = '') AS bad_job_level
    FROM public.staff_data_raw
)
INSERT INTO silver.valid_staff_data_raw
SELECT *
FROM flagged
WHERE NOT (bad_staff_id OR bad_name OR bad_job_level);

INSERT INTO silver.invalid_staff_data_raw
SELECT *
FROM flagged
WHERE (bad_staff_id OR bad_name OR bad_job_level);

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'staff_data_raw',
    'staff_id_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE staff_id IS NULL OR TRIM(staff_id) = '')
FROM public.staff_data_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'staff_data_raw',
    'name_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE name IS NULL OR TRIM(name) = '')
FROM public.staff_data_raw;

INSERT INTO silver.dq_summary (table_name, rule_name, issue_type, severity, total_rows, failed_rows)
SELECT
    'staff_data_raw',
    'job_level_not_null',
    'NULL_CHECK',
    'ERROR',
    COUNT(*),
    COUNT(*) FILTER (WHERE job_level IS NULL OR TRIM(job_level) = '')
FROM public.staff_data_raw;