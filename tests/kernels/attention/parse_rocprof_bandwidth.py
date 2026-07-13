# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Parse a rocprofv3 SQLite results .db and compute achieved HBM bandwidth.

rocprofv3 (this build) writes a SQLite database, not a CSV. Relevant tables:
    rocpd_pmc_event(pmc_id, value)          -- one row per (dispatch, counter)
    rocpd_info_pmc(id, agent_id, name)      -- counter definitions (name, agent)
    rocpd_kernel_dispatch(agent_id, start, end)  -- kernel timings

Decode attention is memory-bound, so achieved bandwidth is the metric of record.
We use the memory-traffic counters exposed by this GPU:
    FETCH_SIZE  -- kilobytes read from HBM   (all cache effects accounted for)
    WRITE_SIZE  -- kilobytes written to HBM
    bytes = (FETCH_SIZE + WRITE_SIZE) * 1024
    BW    = bytes / kernel_busy_time         -- GB/s == bytes / ns

Per agent (XCD / rank), then:
    SPX  -> single agent, aggregate == that agent's BW.
    CPX  -> the 8 XCDs run concurrently, aggregate = SUM of per-agent BW.
            Higher CPX aggregate than SPX is the memory-bandwidth argument for
            CPX partitioning.

The counter names are configurable (--rd-counter / --wr-counter) in case a
different build exposes different names; defaults are FETCH_SIZE / WRITE_SIZE
(KB). If a build exposes byte-valued counters instead, pass --unit-bytes.
"""

import argparse
import glob
import os
import sqlite3
import sys


def _table_exists(con, name):
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone()
    return row is not None


def _agent_busy_ns(con):
    """Per-agent kernel busy time (ns) = sum of (end - start) over dispatches."""
    out = {}
    for agent_id, busy in con.execute(
        'SELECT agent_id, SUM("end" - start) '
        "FROM rocpd_kernel_dispatch GROUP BY agent_id"
    ):
        out[int(agent_id)] = float(busy or 0.0)
    return out


def _agent_counter_sums(con, counter_names):
    """Per-agent summed value for each requested counter name.

    Returns {agent_id: {counter_name: value}}.
    Joins pmc events to their definitions (name + agent).
    """
    placeholders = ",".join("?" for _ in counter_names)
    q = (
        "SELECT ip.agent_id, ip.name, SUM(pe.value) "
        "FROM rocpd_pmc_event pe "
        "JOIN rocpd_info_pmc ip ON pe.pmc_id = ip.id "
        f"WHERE ip.name IN ({placeholders}) "
        "GROUP BY ip.agent_id, ip.name"
    )
    out = {}
    for agent_id, name, val in con.execute(q, counter_names):
        out.setdefault(int(agent_id), {})[name] = float(val or 0.0)
    return out


def parse(db_path, rd_counter, wr_counter, unit_bytes):
    con = sqlite3.connect(db_path)
    try:
        if not _table_exists(con, "rocpd_pmc_event"):
            raise RuntimeError("no rocpd_pmc_event table")
        n_events = con.execute(
            "SELECT COUNT(*) FROM rocpd_pmc_event").fetchone()[0]
        if n_events == 0:
            names = [r[0] for r in con.execute(
                "SELECT DISTINCT name FROM rocpd_info_pmc "
                "WHERE name NOT LIKE 'KFD%'")]
            raise RuntimeError(
                "rocpd_pmc_event is EMPTY -- counters were not collected. "
                "Likely --pmc was combined with --kernel-trace (must be a "
                "PMC-only pass) or the counter names are invalid. "
                f"Non-KFD counters present: {names or '(none)'}")

        busy = _agent_busy_ns(con)
        csums = _agent_counter_sums(con, [rd_counter, wr_counter])
        if not csums:
            avail = [r[0] for r in con.execute(
                "SELECT DISTINCT name FROM rocpd_info_pmc "
                "WHERE name NOT LIKE 'KFD%'")]
            raise RuntimeError(
                f"counters {rd_counter}/{wr_counter} not found in this db. "
                f"Available (non-KFD): {avail}")

        unit_scale = 1.0 if unit_bytes else 1024.0  # KB -> bytes by default
        agents = {}
        for agent_id, cs in csums.items():
            rd = cs.get(rd_counter, 0.0)
            wr = cs.get(wr_counter, 0.0)
            agents[agent_id] = {
                "bytes": (rd + wr) * unit_scale,
                "busy_ns": busy.get(agent_id, 0.0),
            }
        return agents
    finally:
        con.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-glob", required=True,
                   help="glob for the rocprofv3 result .db files of ONE cell. "
                        "SPX: matches 1 db; CPX: matches all 8 (one per rank/"
                        "XCD). ALL matches are parsed and merged by agent.")
    p.add_argument("--mode", required=True)  # spx | cpx | cpx-baseline
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--rd-counter", default="FETCH_SIZE")
    p.add_argument("--wr-counter", default="WRITE_SIZE")
    p.add_argument("--unit-bytes", action="store_true",
                   help="counters are already in bytes (default: KB -> *1024)")
    args = p.parse_args()

    dbs = sorted(glob.glob(args.db_glob, recursive=True))
    if not dbs:
        print(f"[bw-parse] ERROR: no db matched {args.db_glob}", file=sys.stderr)
        _append_row(args, 0, [], float("nan"), float("nan"), float("nan"))
        return

    # Merge agents across all per-rank dbs. Each rank's db reports its own
    # agent(s); keyed by (db_index, agent_id) so distinct ranks never collide
    # even if they reuse the same local agent_id numbering.
    agents = {}
    for i, db_path in enumerate(dbs):
        try:
            a = parse(db_path, args.rd_counter, args.wr_counter,
                      args.unit_bytes)
        except Exception as e:  # noqa: BLE001
            print(f"[bw-parse] WARN: {db_path}: {e}", file=sys.stderr)
            continue
        for agent_id, rec in a.items():
            agents[(i, agent_id)] = rec
    if not agents:
        print(f"[bw-parse] ERROR: no counters parsed from {len(dbs)} db(s)",
              file=sys.stderr)
        _append_row(args, 0, [], float("nan"), float("nan"), float("nan"))
        return

    per_agent_bw = []
    total_bytes = 0.0
    for a in sorted(agents.keys()):
        rec = agents[a]
        bw = rec["bytes"] / rec["busy_ns"] if rec["busy_ns"] > 0 else 0.0
        per_agent_bw.append(bw)
        total_bytes += rec["bytes"]

    agg_bw = sum(per_agent_bw)  # SPX: 1 agent; CPX: sum of concurrent XCDs
    total_ns = max((agents[a]["busy_ns"] for a in agents), default=0.0)

    _append_row(args, len(agents), per_agent_bw, agg_bw, total_bytes, total_ns)
    pa = " ".join(f"{b:.1f}" for b in per_agent_bw)
    print(f"[bw] mode={args.mode} S={args.seq_len} B={args.batch} "
          f"agents={len(agents)} aggregate={agg_bw:.1f} GB/s (per-agent: {pa})")


def _append_row(args, num_agents, per_agent_bw, agg_bw, total_bytes, total_ns):
    header = ("mode,seq_len,batch,num_agents,total_bytes,max_busy_ns,"
              "aggregate_bw_GBs,per_agent_bw_GBs\n")
    write_header = not os.path.exists(args.out_csv)
    per_agent_str = ";".join(f"{b:.4f}" for b in per_agent_bw)
    with open(args.out_csv, "a") as f:
        if write_header:
            f.write(header)
        agg = "" if agg_bw != agg_bw else f"{agg_bw:.4f}"
        tb = "" if total_bytes != total_bytes else f"{total_bytes:.0f}"
        tn = "" if total_ns != total_ns else f"{total_ns:.0f}"
        f.write(f"{args.mode},{args.seq_len},{args.batch},{num_agents},"
                f"{tb},{tn},{agg},{per_agent_str}\n")


if __name__ == "__main__":
    main()
