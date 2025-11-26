import argparse
import os
import sys
import logging
import json
from time import time

import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile

from sqlalchemy import create_engine, inspect
import psycopg2
from psycopg2 import sql
from sqlalchemy.engine.url import make_url

LOG_FMT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("ingest")

DEFAULT_CHUNK = 100_000


def make_engine_from_parts(db_url=None, user=None, password=None, host=None, port=None, db=None):
    """Return SQLAlchemy engine. Prefer db_url if provided, otherwise build from components."""
    if db_url:
        return create_engine(db_url)
    if not all([user, password, host, port, db]):
        raise ValueError("Either --db_url or all of --user/--password/--host/--port/--db must be provided")
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def table_exists(engine, table_name, schema=None):
    """Return True if the given table exists in the target DB/schema."""
    try:
        insp = inspect(engine)
        return insp.has_table(table_name, schema=schema)
    except Exception as e:
        logger.debug("table_exists check failed: %s", e)
        # If inspect fails for any reason, be conservative and return False so first write will create table.
        return False


def copy_csv_to_postgres(csv_path, table_name, db_url):
    """
    Use Postgres COPY to efficiently load a local CSV file.
    table_name is safely quoted using psycopg2.sql to avoid injection.
    db_url is a full SQLAlchemy/psycopg2 compatible URL string.
    """
    logger.info("Attempting Postgres COPY for %s -> %s", csv_path, table_name)

    try:
        u = make_url(db_url)
        conn = psycopg2.connect(
            dbname=u.database,
            user=u.username,
            password=u.password,
            host=u.host,
            port=u.port,
        )
    except Exception:
        conn = psycopg2.connect(db_url)

    cur = conn.cursor()
    with open(csv_path, "rb") as f:
        copy_sql = sql.SQL("COPY {} FROM STDIN WITH CSV HEADER").format(sql.Identifier(table_name))
        try:
            cur.copy_expert(copy_sql.as_string(conn), f)
            conn.commit()
            logger.info("COPY succeeded for %s", table_name)
        except Exception:
            conn.rollback()
            logger.exception("COPY failed - rolling back")
            raise
        finally:
            cur.close()
            conn.close()


def ingest_csv_chunks(csv_path, table_name, engine, chunksize=DEFAULT_CHUNK, use_copy=False, dtype_map=None):
    if use_copy:
        try:
            db_url = engine.url.render_as_string(hide_password=False)
            copy_csv_to_postgres(csv_path, table_name, db_url)
            return
        except Exception as e:
            logger.warning("COPY failed (%s) — falling back to chunked inserts", e)

    logger.info("Reading CSV in chunks (chunksize=%d) from %s", chunksize, csv_path)
    it = pd.read_csv(csv_path, iterator=True, chunksize=chunksize)
    rows_total = 0

    # Decide whether table exists before writing any chunks.
    exists = table_exists(engine, table_name)
    logger.debug("Table exists=%s for %s", exists, table_name)

    for chunk in it:
        t0 = time()
        if dtype_map:
            _apply_dtype_map(chunk, dtype_map)

        # Otherwise, always append to preserve Bronze history.
        if not exists:
            # create table
            chunk.to_sql(table_name, con=engine, if_exists="replace", index=False)
            exists = True
            logger.info("Created table %s with first chunk", table_name)
        else:
            chunk.to_sql(table_name, con=engine, if_exists="append", index=False)
        rows_total += len(chunk)
        logger.info("Wrote %d rows (total %d) in %.3fs", len(chunk), rows_total, time() - t0)
    logger.info("CSV ingestion finished. Total rows: %d", rows_total)


def ingest_parquet_batches(parquet_path, table_name, engine, batch_size=DEFAULT_CHUNK, dtype_map=None):
    logger.info("Reading Parquet in batches from %s (batch_size=%d)", parquet_path, batch_size)
    pf = ParquetFile(parquet_path)
    batch_iter = pf.iter_batches(batch_size=batch_size)
    rows_total = 0

    exists = table_exists(engine, table_name)
    logger.debug("Table exists=%s for %s", exists, table_name)

    for batch in batch_iter:
        t0 = time()
        tbl = pa.Table.from_batches([batch])
        df = tbl.to_pandas()
        if dtype_map:
            _apply_dtype_map(df, dtype_map)

        if not exists:
            df.to_sql(table_name, con=engine, if_exists="replace", index=False)
            exists = True
            logger.info("Created table %s with first parquet batch", table_name)
        else:
            df.to_sql(table_name, con=engine, if_exists="append", index=False)
        rows_total += len(df)
        logger.info("Wrote %d rows (total %d) in %.3fs", len(df), rows_total, time() - t0)
    logger.info("Parquet ingestion finished. Total rows: %d", rows_total)


def ingest_json_lines(json_path, table_name, engine, chunksize=DEFAULT_CHUNK, dtype_map=None):
    logger.info("Streaming JSON lines from %s", json_path)
    buffer = []
    rows_total = 0

    exists = table_exists(engine, table_name)
    logger.debug("Table exists=%s for %s", exists, table_name)

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            buffer.append(obj)
            if len(buffer) >= chunksize:
                df = pd.DataFrame(buffer)
                if dtype_map:
                    _apply_dtype_map(df, dtype_map)
                if not exists:
                    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
                    exists = True
                    logger.info("Created table %s with first JSONL chunk", table_name)
                else:
                    df.to_sql(table_name, con=engine, if_exists="append", index=False)
                rows_total += len(df)
                logger.info("Wrote %d rows (total %d)", len(df), rows_total)
                buffer = []
    if buffer:
        df = pd.DataFrame(buffer)
        if dtype_map:
            _apply_dtype_map(df, dtype_map)
        if not exists:
            df.to_sql(table_name, con=engine, if_exists="replace", index=False)
            logger.info("Created table %s with final JSONL chunk", table_name)
        else:
            df.to_sql(table_name, con=engine, if_exists="append", index=False)
        rows_total += len(df)
    logger.info("JSONL ingestion finished. Total rows: %d", rows_total)


def ingest_json_normal(json_path, table_name, engine, dtype_map=None, detect_ndjson=True, chunksize=DEFAULT_CHUNK):
    """
    Ingest normal JSON files (single-object, list-of-objects, dict-of-dicts).
    If detect_ndjson=True, this will heuristically detect ndjson and stream it (delegates to ingest_json_lines).
    """
    logger.info("Reading normal JSON file: %s (detect_ndjson=%s)", json_path, detect_ndjson)

    if detect_ndjson:
        # quick check for ndjson: many newline-delimited JSON objects starting with '{' per line
        with open(json_path, "r", encoding="utf-8") as f:
            preview = f.read(4096)
            if "\n" in preview and preview.strip().count("\n") >= 1 and preview.strip().lstrip().startswith("{"):
                logger.info("Heuristic detected ndjson; delegating to ingest_json_lines")
                ingest_json_lines(json_path, table_name, engine, chunksize=chunksize, dtype_map=dtype_map)
                return

    # Non-ndjson: load whole JSON and normalize
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize into a DataFrame
    if isinstance(data, list):
        df = pd.json_normalize(data)

    elif isinstance(data, dict):
        if all(not isinstance(v, (list, dict)) for v in data.values()):
            df = pd.DataFrame([data])
        else:
            cols = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    cols[k] = pd.Series(v)
                elif isinstance(v, list):
                    cols[k] = pd.Series(v)
                else:
                    cols[k] = pd.Series([v])
            df = pd.DataFrame(cols)
            df = df.reset_index(drop=True)
    else:
        raise ValueError("Unknown JSON structure: top-level type=%s" % type(data))

    if dtype_map:
        _apply_dtype_map(df, dtype_map)

    # Build DataFrame, creates if not exists, appends if exists
    exists = table_exists(engine, table_name)
    if not exists:
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        logger.info("Created table %s with JSON file", table_name)
    else:
        df.to_sql(table_name, con=engine, if_exists="append", index=False)
        logger.info("Appended JSON file to %s", table_name)

    logger.info("Normal JSON ingestion finished. Rows: %d", len(df))


def ingest_excel(excel_path, table_name, engine, dtype_map=None):
    logger.info("Reading Excel file (may load whole file into memory): %s", excel_path)
    df = pd.read_excel(excel_path, sheet_name=0)
    if dtype_map:
        _apply_dtype_map(df, dtype_map)
    if table_exists(engine, table_name):
        df.to_sql(table_name, con=engine, if_exists="append", index=False)
    else:
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    logger.info("Excel ingestion finished. Rows: %d", len(df))


def ingest_pickle(pkl_path, table_name, engine, dtype_map=None):
    logger.info("Reading pickle file (may load whole file into memory): %s", pkl_path)
    df = pd.read_pickle(pkl_path)
    if dtype_map:
        _apply_dtype_map(df, dtype_map)
    if table_exists(engine, table_name):
        df.to_sql(table_name, con=engine, if_exists="append", index=False)
    else:
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    logger.info("Pickle ingestion finished. Rows: %d", len(df))


def ingest_html(html_path, table_name, engine, dtype_map=None):
    logger.info("Parsing HTML tables from %s", html_path)
    tables = pd.read_html(html_path)
    if not tables:
        raise ValueError("No tables found in HTML")
    df = tables[0]
    if dtype_map:
        _apply_dtype_map(df, dtype_map)
    if table_exists(engine, table_name):
        df.to_sql(table_name, con=engine, if_exists="append", index=False)
    else:
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    logger.info("HTML ingestion finished. Rows: %d", len(df))


def _apply_dtype_map(df, dtype_map):
    """Apply dtype_map (column -> pandas dtype or 'datetime') to dataframe in place."""
    for c, t in dtype_map.items():
        if c in df.columns:
            try:
                if isinstance(t, str) and "datetime" in t.lower():
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                else:
                    df[c] = df[c].astype(t)
            except Exception:
                logger.debug("Could not cast %s to %s", c, t)


def main(args):
    source = args.path
    if not os.path.exists(source):
        logger.error("Path does not exist: %s", source)
        sys.exit(2)

    # Build engine
    engine = make_engine_from_parts(
        db_url=args.db_url,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
        db=args.db,
    )

    dtype_map = None
    if args.dtype_json:
        try:
            dtype_map = json.loads(args.dtype_json)
        except json.JSONDecodeError:
            logger.error("Invalid JSON for --dtype_json")
            sys.exit(4)

    s = source.lower()
    if s.endswith(".csv"):
        ingest_csv_chunks(source, args.table_name, engine, chunksize=args.chunksize, use_copy=args.use_copy, dtype_map=dtype_map)
    elif s.endswith(".parquet"):
        ingest_parquet_batches(source, args.table_name, engine, batch_size=args.batch_size, dtype_map=dtype_map)
    elif s.endswith(".jsonl"):
        ingest_json_lines(source, args.table_name, engine, chunksize=args.chunksize, dtype_map=dtype_map)
    elif s.endswith(".json"):
        ingest_json_normal(source, args.table_name, engine, dtype_map=dtype_map, detect_ndjson=True, chunksize=args.chunksize)
    elif s.endswith(".xlsx") or s.endswith(".xls"):
        ingest_excel(source, args.table_name, engine, dtype_map=dtype_map)
    elif s.endswith(".pkl") or s.endswith(".pickle"):
        ingest_pickle(source, args.table_name, engine, dtype_map=dtype_map)
    elif s.endswith(".html") or s.endswith(".htm"):
        ingest_html(source, args.table_name, engine, dtype_map=dtype_map)
    else:
        logger.error("Unsupported file extension for %s", source)
        sys.exit(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest local files into Postgres with chunking/fast copy")
    parser.add_argument("--path", required=True, help="Local path to the file to ingest")
    parser.add_argument("--table_name", required=True, help="Destination Postgres table name")

    # Database connection: provide --db_url or provide the components
    parser.add_argument("--db_url", help="Full DB URL (e.g. postgresql://user:pw@host:5432/dbname). Preferred")
    parser.add_argument("--user", help="DB user (use only if not using --db_url)")
    parser.add_argument("--password", help="DB password (use only if not using --db_url)")
    parser.add_argument("--host", help="DB host (use only if not using --db_url)")
    parser.add_argument("--port", help="DB port (use only if not using --db_url)")
    parser.add_argument("--db", help="Database name (use only if not using --db_url)")

    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK, help="Chunk size for CSV/JSONL")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CHUNK, help="Batch size for Parquet")
    parser.add_argument("--use_copy", action="store_true", help="Use Postgres COPY for CSV (very fast, local file)")
    parser.add_argument("--dtype_json", help='Optional JSON string mapping column->type, e.g. \'{"col1":"int","col2":"datetime"}\'')

    args = parser.parse_args()
    main(args)
