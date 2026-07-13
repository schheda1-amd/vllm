# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diagnostic: dump the schema of a rocprofv3 results .db so we can find which
table holds the PMC counter values on this build.

Usage:
    python inspect_rocprof_db.py <path-to.db>
    python inspect_rocprof_db.py --glob '/workspace/vllm/bw_*/S8192_B1/FETCH_SIZE/**/*.db'

Prints every table with its row count, then for tables that look like they hold
counters/pmc/events/dispatches: their columns and a few sample rows.
"""

import argparse
import glob
import sqlite3
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("db", nargs="?", help="path to a results .db")
    p.add_argument("--glob", help="glob to locate a .db (first match used)")
    args = p.parse_args()

    db = args.db
    if db is None and args.glob:
        matches = sorted(glob.glob(args.glob, recursive=True))
        db = matches[0] if matches else None
    if not db:
        print("ERROR: provide a db path or --glob", file=sys.stderr)
        sys.exit(1)

    print(f"DB: {db}\n")
    con = sqlite3.connect(db)

    # Include BOTH tables and views (rocpd_* may be exposed as views).
    objs = list(con.execute(
        "SELECT name, type FROM sqlite_master "
        "WHERE type IN ('table','view') ORDER BY type, name"))
    tables = [name for name, _ in objs]
    types = {name: typ for name, typ in objs}

    print("=== ALL TABLES/VIEWS (type name : row count) ===")
    counts = {}
    for t in tables:
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        except sqlite3.Error as e:
            n = f"ERR({e})"
        counts[t] = n
        print(f"  [{types[t]}] {t}: {n}")

    print("\n=== DETAIL for counter/pmc/event/dispatch/agent tables ===")
    hints = ("counter", "pmc", "event", "dispatch", "agent", "info")
    for t in tables:
        if not any(h in t.lower() for h in hints):
            continue
        n = counts.get(t, 0)
        if not isinstance(n, int) or n == 0:
            continue
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
        print(f"\n-- {t} ({n} rows)")
        print(f"   cols: {cols}")
        try:
            for row in con.execute(f'SELECT * FROM "{t}" LIMIT 3'):
                print(f"   row: {tuple(row)}")
        except sqlite3.Error as e:
            print(f"   (sample failed: {e})")

    con.close()


if __name__ == "__main__":
    main()
