BEGIN;

-- Prepare target tables
CREATE TABLE IF NOT EXISTS silver.valid_line_items_raw AS SELECT NULL::uuid AS run_id, * FROM (SELECT NULL::text) AS _null LIMIT 0;
CREATE TABLE IF NOT EXISTS silver.invalid_line_items_raw AS SELECT NULL::uuid AS run_id, * FROM (SELECT NULL::text) AS _null LIMIT 0;

TRUNCATE TABLE silver.valid_line_items_raw;
TRUNCATE TABLE silver.invalid_line_items_raw;

-- pull run_id
WITH run AS (SELECT run_id FROM _this_run LIMIT 1),

raw_prep AS (
  SELECT
    li.*,
    run.run_id,
    trim(lower(li.order_id)) AS order_id_clean,
    trim(li.product_id::text) AS product_id_raw,
    -- normalize price: remove currency symbols, commas, keep numerics/dot
    NULLIF(regexp_replace(li.line_item_price::text, '[^0-9\.]', '', 'g'), '') AS price_digits,
    -- normalize quantity: remove non-digits
    NULLIF(regexp_replace(coalesce(li.line_item_quantity::text, ''), '[^0-9]', '', 'g'), '') AS quantity_digits,
    coalesce(li.source_ts, now()) AS source_ts_parsed
  FROM public.line_items_raw li
  CROSS JOIN run
),

-- dedupe by line_item_id if present, otherwise by order_id+product_id
dedup AS (
  SELECT rp.*,
    row_number() OVER (PARTITION BY coalesce(rp.line_item_id::text, rp.order_id_clean || '|' || rp.product_id_raw) ORDER BY source_ts_parsed DESC NULLS LAST) AS rn
  FROM raw_prep rp
),

flagged AS (
  SELECT
    d.*,
    (order_id_clean IS NULL OR order_id_clean = '') AS bad_order_id,
    (product_id_raw IS NULL OR product_id_raw = '') AS bad_product_id,
    (price_digits IS NULL) AS bad_price,
    (quantity_digits IS NULL) AS bad_quantity
  FROM dedup d
  WHERE rn = 1
)

-- insert valids
INSERT INTO silver.valid_line_items_raw
SELECT
  run_id,
  COALESCE(line_item_id, (md5(order_id_clean || product_id_raw))::bigint) AS line_item_id,
  product_id_raw::int AS product_id,
  order_id_clean AS order_id,
  price_digits::numeric(12,2) AS line_item_price,
  quantity_digits::int AS line_item_quantity,
  now() AS validated_at
FROM flagged f
WHERE NOT (bad_order_id OR bad_product_id OR bad_price OR bad_quantity);

-- insert invalids with reasons
INSERT INTO silver.invalid_line_items_raw
SELECT
  run_id,
  line_item_id,
  product_id_raw AS product_id,
  order_id,
  line_item_price,
  line_item_quantity,
  source_ts_parsed,
  now() AS invalidated_at,
  concat_ws('; ',
    CASE WHEN bad_order_id THEN 'missing_order_id' ELSE NULL END,
    CASE WHEN bad_product_id THEN 'missing_product_id' ELSE NULL END,
    CASE WHEN bad_price THEN 'bad_price_format' ELSE NULL END,
    CASE WHEN bad_quantity THEN 'bad_quantity' ELSE NULL END
  ) AS dq_reasons
FROM flagged f
WHERE (bad_order_id OR bad_product_id OR bad_price OR bad_quantity);

-- FK check: product existence in valid_products
WITH run AS (SELECT run_id FROM _this_run LIMIT 1),
missing_products AS (
  SELECT v.*
  FROM silver.valid_line_items_raw v
  LEFT JOIN silver.valid_products p ON v.product_id = p.product_id
  WHERE p.product_id IS NULL
)
INSERT INTO silver.invalid_line_items_raw
SELECT
  run_id,
  line_item_id,
  product_id::text,
  order_id,
  line_item_price::text,
  line_item_quantity::text,
  now() AS invalidated_at,
  'missing_product_fk' AS dq_reasons
FROM missing_products;

-- remove moved rows from valids
DELETE FROM silver.valid_line_items_raw v
USING missing_products m
WHERE v.line_item_id = m.line_item_id;

-- DQ summary rows for line_items
WITH run AS (SELECT run_id FROM _this_run LIMIT 1)
INSERT INTO silver.dq_summary (run_id, table_name, rule_name, issue_type, severity, total_rows, failed_rows, failed_ratio, created_at)
SELECT run.run_id, 'line_items' AS table_name,
       rule_name, issue_type, severity,
       (SELECT count(*) FROM public.line_items_raw) AS total_rows,
       failed_rows,
       CASE WHEN (SELECT count(*) FROM public.line_items_raw) = 0 THEN 0 ELSE failed_rows::double precision / (SELECT count(*) FROM public.line_items_raw) END,
       now()
FROM (
  VALUES
    ('missing_order_id','NULL_CHECK','ERROR', (SELECT count(*) FROM silver.invalid_line_items_raw WHERE dq_reasons ILIKE '%missing_order_id%')),
    ('missing_product_id','NULL_CHECK','ERROR', (SELECT count(*) FROM silver.invalid_line_items_raw WHERE dq_reasons ILIKE '%missing_product_id%')),
    ('bad_price_format','FORMAT_CHECK','ERROR', (SELECT count(*) FROM silver.invalid_line_items_raw WHERE dq_reasons ILIKE '%bad_price_format%')),
    ('bad_quantity','FORMAT_CHECK','ERROR', (SELECT count(*) FROM silver.invalid_line_items_raw WHERE dq_reasons ILIKE '%bad_quantity%')),
    ('missing_product_fk','FK_CHECK','ERROR', (SELECT count(*) FROM silver.invalid_line_items_raw WHERE dq_reasons ILIKE '%missing_product_fk%'))
) t(rule_name, issue_type, severity, failed_rows);

-- Save invalid samples
WITH run AS (SELECT run_id FROM _this_run LIMIT 1)
INSERT INTO silver.invalid_samples (run_id, table_name, rule_name, example_row)
SELECT run.run_id, 'line_items', dq_rule, to_jsonb(i)
FROM (
  SELECT i.*, unnest(string_to_array(dq_reasons, ';')) AS dq_rule,
         row_number() OVER (PARTITION BY unnest(string_to_array(dq_reasons, ';')) ORDER BY invalidated_at DESC) AS rn
  FROM silver.invalid_line_items_raw i
) i
WHERE i.rn <= 50;

COMMIT;
