BEGIN;

WITH last_run AS (
  SELECT run_id FROM _this_run LIMIT 1
),
error_totals AS (
  SELECT SUM(failed_rows) AS total_errors
  FROM silver.dq_summary ds
  JOIN last_run lr ON ds.run_id = lr.run_id
  WHERE ds.severity = 'ERROR'
)
SELECT
  CASE WHEN COALESCE(total_errors,0) > 0 THEN
    RAISE EXCEPTION 'DQ FAILED: % ERROR rows found. Blocking silver build.', total_errors
  ELSE
    (INSERT INTO silver.dq_summary (run_id, table_name, rule_name, issue_type, severity, total_rows, failed_rows, failed_ratio, created_at)
     SELECT lr.run_id, 'validation_guard', 'no_errors', 'CHECK', 'INFO', 0, 0, 0.0, now() FROM last_run lr)
  END
FROM error_totals;

COMMIT;
