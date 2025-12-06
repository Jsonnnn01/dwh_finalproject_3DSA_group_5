"""
List files under a base directory and print JSON array of objects:
[{"path": "/abs/path/to/file", "table_name": "file_raw"}, ...]
"""

from __future__ import annotations
import os
import json
import argparse
from typing import List, Dict

def build_table_name(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    # replace spaces with underscore, lowercase
    return name.lower().replace(" ", "_") + "_raw"

def list_files(base_path: str) -> List[Dict[str, str]]:
    out = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.startswith("."):
                continue
            full = os.path.abspath(os.path.join(root, f))
            out.append({
                "path": full,
                "table_name": build_table_name(f)
            })
    return out

def main():
    p = argparse.ArgumentParser(description="List files under base path and output JSON")
    p.add_argument("--base", "-b", default="/data/raw", help="Base directory to walk (default: /data/raw)")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = p.parse_args()

    results = list_files(args.base)
    if args.pretty:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(results, ensure_ascii=False))

if __name__ == "__main__":
    main()
