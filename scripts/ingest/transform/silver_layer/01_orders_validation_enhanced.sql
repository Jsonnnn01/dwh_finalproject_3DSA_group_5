BEGIN;

-- 1) Prepare target tables (include run_id + captured_at)
CREATE TABLE IF NOT EXISTS silver.valid_orders_raw AS
SELECT NULL::uuid::uuid AS run_id, * FROM (SELECT NULL::text) AS _null LIMIT 0;
ALTER TABLE silver.valid_orders_raw DROP CONSTRAINT IF EXISTS valid_orders_raw_pkey;
-- structure will be created more explicitly below if needed

CREATE TABLE IF NOT EXISTS silver.invalid_orders_raw AS
SELECT NULL::uuid AS run_id, * FROM (SELECT NULL::text) AS _null LIMIT 0;

-- Truncate to make run idempotent (will reinsert with new run_id)
TRUNCATE TABLE silver.valid_orders_raw;
TRUNCATE TABLE silver.invalid_orders_raw;

-- Pull current run_id
WITH run AS (
  SELECT run_id FROM _this_run LIMIT 1
),

-- 2) Ingest raw orders with normalized order_id and dedupe
raw_prep AS (
  SELECT
    o.*,
    run.run_id,
    trim(lower(o.order_id)) AS order_id_clean,
    coalesce(o.source_ts, now()) AS source_ts_parsed
  FROM public.orders_raw o
  CROSS JOIN run
),

-- flag duplicates by order_id_clean; keep latest by source_ts_parsed if duplicates
dedup AS (
  SELECT *, 
    row_number() OVER (PARTITION BY order_id_clean ORDER BY source_ts_parsed DESC NULLS LAST) AS rn,
    count(*) OVER (PARTITION BY order_id_clean) AS dup_count
  FROM raw_prep
),

-- 3) Basic column checks + format checks
flagged AS (
  SELECT
    d.*,
    (order_id_clean IS NULL OR order_id_clean = '') AS bad_order_id,
    (user_id IS NULL) AS bad_user_id,
    (merchant_id IS NULL) AS bad_merchant_id,
    (staff_id IS NULL) AS bad_staff_id,
    (transaction_date IS NULL) AS bad_transaction_date,
    (quantity IS NULL OR regexp_replace(quantity::text, '[^0-9.]', '', 'g') = '') AS bad_quantity_format
  FROM dedup d
  WHERE rn = 1  -- only keep deduped candidates here
)

-- 4) Insert valids and invalids
INSERT INTO silver.valid_orders_raw
SELECT
  run_id,
  order_id_clean AS order_id,
  transaction_date,
  (CASE WHEN quantity IS NULL THEN NULL ELSE (regexp_replace(quantity::text, '[^0-9]', '', 'g'))::INT END) AS quantity_clean,
  user_id,
  product_id,
  campaign_id,
  merchant_id,
  staff_id,
  source_ts_parsed,
  now() AS validated_at
FROM flagged f
WHERE NOT (bad_order_id OR bad_user_id OR bad_merchant_id OR bad_staff_id OR bad_transaction_date OR bad_quantity_format);

-- write invalid rows (include reason columns)
INSERT INTO silver.invalid_orders_raw
SELECT
  run_id,
  order_id,
  transaction_date,
  quantity,
  user_id,
  product_id,
  campaign_id,
  merchant_id,
  staff_id,
  source_ts_parsed,
  now() AS invalidated_at,
  -- generate a short reason
  concat_ws('; ',
    CASE WHEN bad_order_id THEN 'missing_order_id' ELSE NULL END,
    CASE WHEN bad_user_id THEN 'missing_user_id' ELSE NULL END,
    CASE WHEN bad_merchant_id THEN 'missing_merchant_id' ELSE NULL END,
    CASE WHEN bad_staff_id THEN 'missing_staff_id' ELSE NULL END,
    CASE WHEN bad_transaction_date THEN 'missing_transaction_date' ELSE NULL END,
    CASE WHEN bad_quantity_format THEN 'bad_quantity_format' ELSE NULL END
  ) AS dq_reasons
FROM flagged f
WHERE (bad_order_id OR bad_user_id OR bad_merchant_id OR bad_staff_id OR bad_transaction_date OR bad_quantity_format);

-- 5) Referential integrity checks: ensure referenced keys exist in their valid_* tables.
--    Any valid_orders_raw rows that reference missing parents should be moved to invalid_orders_raw.
WITH run AS (SELECT run_id FROM _this_run LIMIT 1),
missing_refs AS (
  SELECT v.*
  FROM silver.valid_orders_raw v
  LEFT JOIN silver.valid_users u ON v.user_id = u.user_id
  LEFT JOIN silver.valid_merchants m ON v.merchant_id = m.merchant_id
  LEFT JOIN silver.valid_staff s ON v.staff_id = s.staff_id
  CROSS JOIN run
  WHERE u.user_id IS NULL OR m.merchant_id IS NULL OR s.staff_id IS NULL
)
-- move them to invalids
INSERT INTO silver.invalid_orders_raw
SELECT
  run_id,
  order_id,
  transaction_date,
  quantity_clean::text AS quantity,
  user_id,
  product_id,
  campaign_id,
  merchant_id,
  staff_id,
  source_ts_parsed,
  now() AS invalidated_at,
  concat_ws('; ',
    CASE WHEN user_id IS NULL OR (SELECT user_id FROM silver.valid_users WHERE user_id = missing_refs.user_id) IS NULL THEN 'missing_user_fk' ELSE NULL END,
    CASE WHEN merchant_id IS NULL OR (SELECT merchant_id FROM silver.valid_merchants WHERE merchant_id = missing_refs.merchant_id) IS NULL THEN 'missing_merchant_fk' ELSE NULL END,
    CASE WHEN staff_id IS NULL OR (SELECT staff_id FROM silver.valid_staff WHERE staff_id = missing_refs.staff_id) IS NULL THEN 'missing_staff_fk' ELSE NULL END
  ) AS dq_reasons
FROM missing_refs;

-- remove those moved rows from valid_orders_raw
DELETE FROM silver.valid_orders_raw v
USING missing_refs m
WHERE v.order_id = m.order_id;

-- 6) Record DQ summary rows for this run
WITH run AS (SELECT run_id FROM _this_run LIMIT 1),
totals AS (
  SELECT 'orders'::text AS table_name,
         'missing_order_id' AS rule_name,
         'NULL_CHECK' AS issue_type,
         'ERROR' AS severity,
         (SELECT count(*) FROM public.orders_raw) AS total_rows,
         (SELECT count(*) FROM silver.invalid_orders_raw WHERE dq_reasons ILIKE '%missing_order_id%') AS failed_rows
),
other AS (
  SELECT 'orders'::text AS table_name,
         'missing_fk' AS rule_name,
         'FK_CHECK' AS issue_type,
         'ERROR' AS severity,
         (SELECT count(*) FROM public.orders_raw) AS total_rows,
         (SELECT count(*) FROM silver.invalid_orders_raw WHERE dq_reasons ILIKE '%missing_%fk%') AS failed_rows
)
INSERT INTO silver.dq_summary (run_id, table_name, rule_name, issue_type, severity, total_rows, failed_rows, failed_ratio, created_at)
SELECT run.run_id, t.table_name, t.rule_name, t.issue_type, t.severity, t.total_rows, t.failed_rows,
       CASE WHEN t.total_rows = 0 THEN 0 ELSE (t.failed_rows::double precision / t.total_rows) END, now()
FROM run CROSS JOIN (
  SELECT * FROM totals
  UNION ALL
  SELECT * FROM other
) t;

-- 7) Save up to 50 sample invalid rows per distinct dq reason for triage
WITH run AS (SELECT run_id FROM _this_run LIMIT 1)
INSERT INTO silver.invalid_samples (run_id, table_name, rule_name, example_row)
SELECT run.run_id, 'orders' AS table_name, dq_rule, to_jsonb(i)
FROM (
  SELECT i.*, unnest(string_to_array(dq_reasons, ';')) AS dq_rule,
         row_number() OVER (PARTITION BY unnest(string_to_array(dq_reasons, ';')) ORDER BY invalidated_at DESC) AS rn
  FROM silver.invalid_orders_raw i
) i
WHERE i.rn <= 50;

COMMIT;
