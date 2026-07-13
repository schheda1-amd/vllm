# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Parse a rocprofv3 counter_collection.csv and compute achieved HBM bandwidth.

Decode attention is memory-bound, so achieved bandwidth is the metric of record.
We use the memory-controller (EA) request counters:

    bytes  = (TCC_EA_RDREQ + TCC_EA_WRREQ) * 64      # 64B cache line on CDNA
    BW     = bytes / kernel_busy_time                # GB/s == bytes / ns

Per agent (XCD / rank):
    BW_agent = bytes_agent / busy_ns_agent

Aggregate:
    SPX  -> single agent, so aggregate == that agent's BW.
    CPX  -> the 8 XCDs run concurrently, so aggregate = SUM of per-agent BW.
            Higher aggregate BW than SPX is the memory-bandwidth argument for
            CPX partitioning: more of the HBM is exploited in parallel.

The parser is defensive about rocprofv3 CSV layout (it differs by version):
it locates columns by fuzzy header name and supports BOTH the "long" format
(one row per (dispatch, counter) with Counter_Name/Counter_Value columns) and
the "wide" format (one column per counter). If a needed column is missing it
prints the header it saw so the invocation can be adjusted.

Usage:
    python parse_rocprof_bandwidth.py \
        --counter-csv <dir>/counter_collection.csv \
        --mode spx|cpx --seq-len S --batch B \
        --out-csv <bandwidth.csv>
"""

import argparse
import csv
import sys


RD_COUNTER = "TCC_EA_RDREQ"
WR_COUNTER = "TCC_EA_WRREQ"
LINE_BYTES = 64  # CDNA HBM cache line / EA request granularity


def _find_col(header, *needles):
    """Return index of the first header cell whose lowercased name contains all
    needles (case-insensitive). None if not found."""
    low = [h.strip().lower() for h in header]
    for i, h in enumerate(low):
        if all(n in h for n in needles):
            return i
    return None


def _to_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


def parse(counter_csv):
    """Return dict: agent_id -> {'bytes': float, 'busy_ns': float}."""
    with open(counter_csv, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise RuntimeError(f"empty CSV: {counter_csv}")
    header, data = rows[0], rows[1:]

    # NOTE: use explicit `is not None` chaining -- _find_col can return 0
    # (a valid column index that is falsy), which an `or` chain would skip.
    agent_i = _find_col(header, "agent", "id")
    if agent_i is None:
        agent_i = _find_col(header, "agent")
    if agent_i is None:
        agent_i = _find_col(header, "device")
    if agent_i is None:
        agent_i = _find_col(header, "gpu")
    start_i = _find_col(header, "start", "timestamp")
    end_i = _find_col(header, "end", "timestamp")
    dur_i = _find_col(header, "duration")
    disp_i = _find_col(header, "dispatch", "id")

    cname_i = _find_col(header, "counter", "name")
    cval_i = _find_col(header, "counter", "value")

    agents: dict = {}

    def _agent_key(row):
        if agent_i is not None and agent_i < len(row):
            return row[agent_i].strip()
        return "0"  # single-agent fallback

    def _ensure(a):
        return agents.setdefault(a, {"bytes": 0.0, "busy_ns": 0.0,
                                     "_seen_dispatch": set()})

    def _dispatch_dur(row):
        if dur_i is not None and dur_i < len(row):
            return _to_float(row[dur_i])
        if (start_i is not None and end_i is not None
                and start_i < len(row) and end_i < len(row)):
            return _to_float(row[end_i]) - _to_float(row[start_i])
        return 0.0

    if cname_i is not None and cval_i is not None:
        # --- long format: one row per (dispatch, counter) ---
        rd_i = wr_i = None  # not used
        for row in data:
            if not row or cname_i >= len(row) or cval_i >= len(row):
                continue
            a = _agent_key(row)
            rec = _ensure(a)
            name = row[cname_i].strip()
            val = _to_float(row[cval_i])
            if name == RD_COUNTER or name == WR_COUNTER:
                rec["bytes"] += val * LINE_BYTES
            # accumulate duration once per dispatch (rows repeat per counter)
            if disp_i is not None and disp_i < len(row):
                did = row[disp_i]
                if did not in rec["_seen_dispatch"]:
                    rec["_seen_dispatch"].add(did)
                    rec["busy_ns"] += _dispatch_dur(row)
            else:
                # no dispatch id: best-effort, may double count -> warn later
                rec["busy_ns"] += _dispatch_dur(row)
    else:
        # --- wide format: one column per counter ---
        rd_i = _find_col(header, RD_COUNTER.lower())
        wr_i = _find_col(header, WR_COUNTER.lower())
        if rd_i is None and wr_i is None:
            raise RuntimeError(
                "Could not find TCC_EA_RDREQ/WRREQ columns and no "
                "Counter_Name/Value pair. Header was:\n  " + ",".join(header))
        for row in data:
            if not row:
                continue
            a = _agent_key(row)
            rec = _ensure(a)
            if rd_i is not None and rd_i < len(row):
                rec["bytes"] += _to_float(row[rd_i]) * LINE_BYTES
            if wr_i is not None and wr_i < len(row):
                rec["bytes"] += _to_float(row[wr_i]) * LINE_BYTES
            rec["busy_ns"] += _dispatch_dur(row)

    # strip helper sets
    for rec in agents.values():
        rec.pop("_seen_dispatch", None)
    return agents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--counter-csv", required=True)
    p.add_argument("--mode", required=True)  # spx | cpx | cpx-baseline
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    try:
        agents = parse(args.counter_csv)
    except Exception as e:  # noqa: BLE001
        print(f"[bw-parse] ERROR on {args.counter_csv}: {e}", file=sys.stderr)
        # Still emit a row so the sweep CSV has a placeholder.
        _append_row(args, num_agents=0, per_agent_bw=[], agg_bw=float("nan"),
                    total_bytes=float("nan"), total_ns=float("nan"))
        return

    per_agent_bw = []
    total_bytes = 0.0
    for a in sorted(agents.keys()):
        rec = agents[a]
        bw = rec["bytes"] / rec["busy_ns"] if rec["busy_ns"] > 0 else 0.0
        per_agent_bw.append(bw)
        total_bytes += rec["bytes"]

    # SPX: one agent -> aggregate is that agent. CPX: sum concurrent XCDs.
    agg_bw = sum(per_agent_bw)
    total_ns = max((agents[a]["busy_ns"] for a in agents), default=0.0)

    _append_row(args, num_agents=len(agents), per_agent_bw=per_agent_bw,
                agg_bw=agg_bw, total_bytes=total_bytes, total_ns=total_ns)

    pa = " ".join(f"{b:.1f}" for b in per_agent_bw)
    print(f"[bw] mode={args.mode} S={args.seq_len} B={args.batch} "
          f"agents={len(agents)} aggregate={agg_bw:.1f} GB/s "
          f"(per-agent: {pa})")


def _append_row(args, num_agents, per_agent_bw, agg_bw, total_bytes, total_ns):
    import os
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
