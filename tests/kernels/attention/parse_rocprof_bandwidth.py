# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Parse rocprofv3 SQLite result .db files and compute achieved HBM bandwidth.

Decode attention is memory-bound, so achieved bandwidth is the metric of record.
This GPU exposes FETCH_SIZE / WRITE_SIZE (KB read / written at the HBM
interface). Those two are DERIVED counters that together exceed the per-pass
hardware counter budget, so the launcher collects them in SEPARATE rocprofv3
passes. Each pass is its own process and writes one .db per rank.

A cell directory therefore contains:  (passes) x (ranks) db files.
  SPX          : 2 dbs   (FETCH pass, WRITE pass; 1 rank)
  cpx / cpx-bl : 16 dbs  (2 passes x 8 XCD ranks)

Aggregation model (robust; no per-XCD cross-pass identity needed -- note ALL
XCDs on one physical GPU share the same agent uuid, so per-agent keying across
passes is impossible):

    total_bytes = SUM over ALL dbs of (FETCH_or_WRITE_KB * 1024) on GPU agents
    wall_ns     = MAX over ALL dbs of (GPU kernel-busy ns in that db)
    aggregate_BW = total_bytes / wall_ns            # bytes/ns == GB/s

Why this is correct:
  * SPX: fetch-db and write-db each run the identical workload (busy ~= T).
    total = fetch+write bytes; wall = T; BW = (fetch+write)/T. Correct read+
    write bandwidth of the one GPU.
  * CPX: the 8 XCDs run concurrently, each ~= T busy. total = sum over 8 XCDs
    of (fetch+write); wall = max busy = T; BW = total/T = sum of per-XCD BW =
    the aggregate HBM utilization across the physical GPU's XCDs. Higher than
    SPX is the memory-bandwidth argument for CPX partitioning.

Only GPU agents (rocpd_info_agent.type='GPU') are counted; CPU agents excluded.
"""

import argparse
import glob
import os
import sqlite3
import sys


def _obj_exists(con, name):
    # rocpd_* may be a base table OR a view depending on rocprofv3 version.
    row = con.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table','view') AND name=?",
        (name,)).fetchone()
    return row is not None


def _gpu_agent_ids(con):
    """Set of agent_id values whose type is GPU (exclude CPU sockets)."""
    ids = set()
    if _obj_exists(con, "rocpd_info_agent"):
        try:
            for (aid,) in con.execute(
                "SELECT id FROM rocpd_info_agent WHERE upper(type)='GPU'"
            ):
                ids.add(int(aid))
        except sqlite3.OperationalError:
            pass
    return ids


def parse_db(db_path, rd_counter, wr_counter, unit_bytes):
    """Return (bytes, busy_ns, n_gpu_agents_with_work) for ONE db.

    `bytes` = summed (rd/wr counter) * scale over GPU agents in this db (a db
    from a single --pmc pass carries only ONE of the two counters).
    `busy_ns` = summed GPU kernel-dispatch duration in this db.
    """
    con = sqlite3.connect(db_path)
    try:
        if not _obj_exists(con, "rocpd_pmc_event"):
            raise RuntimeError("no rocpd_pmc_event table/view")
        n_events = con.execute(
            "SELECT COUNT(*) FROM rocpd_pmc_event").fetchone()[0]
        if n_events == 0:
            names = [r[0] for r in con.execute(
                "SELECT DISTINCT name FROM rocpd_info_pmc "
                "WHERE name NOT LIKE 'KFD%'")]
            raise RuntimeError(
                "rocpd_pmc_event is EMPTY -- counters not collected "
                f"(non-KFD counters present: {names or '(none)'})")

        gpu_ids = _gpu_agent_ids(con)

        def _gpu_filter(col):
            if not gpu_ids:
                return ""  # no agent info -> don't filter (best effort)
            ids = ",".join(str(i) for i in sorted(gpu_ids))
            return f" AND {col} IN ({ids})"

        # Counter bytes (KB -> bytes) for the counters present in this db.
        placeholders = ",".join("?" for _ in (rd_counter, wr_counter))
        q_cnt = (
            "SELECT SUM(pe.value) "
            "FROM rocpd_pmc_event pe "
            "JOIN rocpd_info_pmc ip ON pe.pmc_id = ip.id "
            f"WHERE ip.name IN ({placeholders})"
            + _gpu_filter("ip.agent_id")
        )
        row = con.execute(q_cnt, (rd_counter, wr_counter)).fetchone()
        kb = float(row[0] or 0.0) if row else 0.0
        scale = 1.0 if unit_bytes else 1024.0
        db_bytes = kb * scale

        # GPU kernel-busy ns in this db. If a db contains multiple GPU agents
        # (XCDs), they run CONCURRENTLY, so this db's wall time is the MAX of
        # per-agent busy sums, not the sum across agents. (On the real runs each
        # rank writes its own db with a single XCD, so this equals that XCD's
        # busy; the max-over-agents form just makes it robust either way.)
        q_busy = (
            'SELECT agent_id, SUM("end" - start) FROM rocpd_kernel_dispatch '
            "WHERE 1=1" + _gpu_filter("agent_id") + " GROUP BY agent_id"
        )
        busy_ns = 0.0
        for _aid, b in con.execute(q_busy):
            busy_ns = max(busy_ns, float(b or 0.0))

        # How many distinct GPU agents actually did work (diagnostic).
        q_na = (
            "SELECT COUNT(DISTINCT agent_id) FROM rocpd_kernel_dispatch "
            "WHERE 1=1" + _gpu_filter("agent_id")
        )
        n_agents = int(con.execute(q_na).fetchone()[0] or 0)

        return db_bytes, busy_ns, n_agents
    finally:
        con.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-glob", required=True,
                   help="glob for ALL result .db of ONE cell (both passes, all "
                        "ranks). e.g. '<cell_dir>/**/*.db'")
    p.add_argument("--mode", required=True)  # spx | cpx | cpx-baseline
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--rd-counter", default="FETCH_SIZE")
    p.add_argument("--wr-counter", default="WRITE_SIZE")
    p.add_argument("--unit-bytes", action="store_true",
                   help="counters already in bytes (default: KB -> *1024)")
    args = p.parse_args()

    dbs = sorted(glob.glob(args.db_glob, recursive=True))
    if not dbs:
        print(f"[bw-parse] ERROR: no db matched {args.db_glob}", file=sys.stderr)
        _append_row(args, 0, 0, float("nan"), float("nan"), float("nan"))
        return

    total_bytes = 0.0
    wall_ns = 0.0
    n_dbs_ok = 0
    max_agents = 0
    for db_path in dbs:
        try:
            db_bytes, busy_ns, n_agents = parse_db(
                db_path, args.rd_counter, args.wr_counter, args.unit_bytes)
        except Exception as e:  # noqa: BLE001
            print(f"[bw-parse] WARN: {db_path}: {e}", file=sys.stderr)
            continue
        total_bytes += db_bytes
        wall_ns = max(wall_ns, busy_ns)   # concurrent -> max, not sum
        max_agents = max(max_agents, n_agents)
        n_dbs_ok += 1

    if n_dbs_ok == 0 or wall_ns <= 0:
        print(f"[bw-parse] ERROR: nothing parsed from {len(dbs)} db(s)",
              file=sys.stderr)
        _append_row(args, 0, 0, float("nan"), float("nan"), float("nan"))
        return

    aggregate_bw = total_bytes / wall_ns  # bytes/ns == GB/s
    _append_row(args, max_agents, n_dbs_ok, aggregate_bw, total_bytes, wall_ns)
    print(f"[bw] mode={args.mode} S={args.seq_len} B={args.batch} "
          f"dbs={n_dbs_ok} gpu_agents={max_agents} "
          f"total={total_bytes/1e9:.3f} GB wall={wall_ns/1e6:.3f} ms "
          f"aggregate={aggregate_bw:.1f} GB/s")


def _append_row(args, n_agents, n_dbs, agg_bw, total_bytes, wall_ns):
    header = ("mode,seq_len,batch,gpu_agents,n_dbs,total_bytes,wall_ns,"
              "aggregate_bw_GBs\n")
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a") as f:
        if write_header:
            f.write(header)
        agg = "" if agg_bw != agg_bw else f"{agg_bw:.4f}"
        tb = "" if total_bytes != total_bytes else f"{total_bytes:.0f}"
        wn = "" if wall_ns != wall_ns else f"{wall_ns:.0f}"
        f.write(f"{args.mode},{args.seq_len},{args.batch},{n_agents},"
                f"{n_dbs},{tb},{wn},{agg}\n")


if __name__ == "__main__":
    main()
