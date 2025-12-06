#!/usr/bin/env python3
"""
Run ingest_data.py for every file discovered by list_files.py.

Usage examples:
  python3 run_ingest_for_all.py --list-cmd "./scripts/utils/list_files.py --base ./data/raw"
  OR (when running in container) use absolute paths and ensure ingest_data.py exists in /app/scripts/ingest/
"""

from __future__ import annotations
import subprocess
import json
import argparse
import shlex
import sys
from typing import List, Dict

def run_list(list_cmd: str) -> List[Dict[str, str]]:
    # run the list command and parse JSON from stdout
    proc = subprocess.run(list_cmd, shell=True, check=True, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    if not stdout:
        return []
    return json.loads(stdout)

def run_ingest_command(python_exe: str, ingest_script_path: str, file_path: str, table_name: str,
                       db_user: str, db_password: str, db_host: str, db_port: str, db_name: str) -> int:
    cmd = [
        python_exe, ingest_script_path,
        "--path", file_path,
        "--table_name", table_name,
        "--user", db_user,
        "--password", db_password,
        "--host", db_host,
        "--port", db_port,
        "--db", db_name,
        "--use_copy"
    ]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd)
    return proc.returncode

def main():
    p = argparse.ArgumentParser(description="Run ingest_data.py for all files returned by list_files")
    p.add_argument("--list-cmd", required=True, help='Command to produce JSON list, e.g. "./scripts/utils/list_files.py --base /data/raw"')
    p.add_argument("--python", default="python3", help="Python executable to use (default: python3)")
    p.add_argument("--ingest-script", default="./scripts/ingest/ingest_data.py", help="Path to ingest_data.py")
    p.add_argument("--db-user", default="postgres")
    p.add_argument("--db-password", default="postgres")
    p.add_argument("--db-host", default="db")
    p.add_argument("--db-port", default="5432")
    p.add_argument("--db-name", default="bronze")
    args = p.parse_args()

    try:
        files = run_list(args.list_cmd)
    except subprocess.CalledProcessError as e:
        print("Failed to run list command:", e, file=sys.stderr)
        sys.exit(2)

    if not files:
        print("No files found by list command.")
        return

    for item in files:
        file_path = item["path"]
        table_name = item.get("table_name", "")
        print("="*80)
        print("Ingesting:", file_path, "->", table_name)
        rc = run_ingest_command(
            args.python, args.ingest_script, file_path, table_name,
            args.db_user, args.db_password, args.db_host, args.db_port, args.db_name
        )
        if rc != 0:
            print(f"INGEST FAILED for {file_path} (exit {rc})", file=sys.stderr)
            sys.exit(rc)
        else:
            print(f"INGEST OK for {file_path}")

if __name__ == "__main__":
    main()
