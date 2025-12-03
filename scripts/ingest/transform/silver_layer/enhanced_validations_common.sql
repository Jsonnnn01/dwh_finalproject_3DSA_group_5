BEGIN;

-- 1) Run housekeeping: create run_id generator table (if not exists)
CREATE TABLE IF NOT EXISTS silver.validation_runs (
  run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  started_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  description TEXT
);

-- Insert a row to mark this run and capture run_id
INSERT INTO silver.validation_runs (description) VALUES ('validation run - automated')
RETURNING run_id INTO TEMP TABLE _this_run;

-- get run id into a variable-friendly place using a temp table
-- NOTE: many SQL clients don't support variables; scripts below will join on this temp table _this_run.

-- 2) Create or ensure dq_summary includes run_id
CREATE TABLE IF NOT EXISTS silver.dq_summary (
  run_id UUID NOT NULL,
  table_name TEXT NOT NULL,
  rule_name TEXT NOT NULL,
  issue_type TEXT,
  severity TEXT,
  total_rows BIGINT,
  failed_rows BIGINT,
  failed_ratio DOUBLE PRECISION,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 3) Create invalid samples table to hold a small sample per rule + run
CREATE TABLE IF NOT EXISTS silver.invalid_samples (
  run_id UUID NOT NULL,
  table_name TEXT NOT NULL,
  rule_name TEXT NOT NULL,
  example_row JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 4) Helper: strip_prefix(text, prefix) -> remove prefix case-insensitively
CREATE OR REPLACE FUNCTION silver.strip_prefix(value TEXT, prefix TEXT) RETURNS TEXT AS $$
BEGIN
  IF value IS NULL THEN
    RETURN NULL;
  END IF;
  RETURN regexp_replace(value, '^' || prefix || ' ?', '', 'i');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- 5) Helper: canonical_phone(text) -> digits-only, returns NULL if not 10-11 digits
CREATE OR REPLACE FUNCTION silver.canonical_phone(raw TEXT) RETURNS TEXT AS $$
DECLARE
  digits TEXT;
BEGIN
  IF raw IS NULL THEN
    RETURN NULL;
  END IF;
  digits := regexp_replace(raw, '\D', '', 'g');
  IF length(digits) = 10 OR length(digits) = 11 THEN
    RETURN digits;
  ELSE
    RETURN NULL;
  END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMIT;
