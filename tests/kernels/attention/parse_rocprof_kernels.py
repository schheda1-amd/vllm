# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-kernel GPU time breakdown from rocprofv3 kernel-trace .db files.

Companion to parse_rocprof_bandwidth.py, which answers "what bandwidth did the
attention path achieve" by collapsing everything under a kernel filter into one
number. This one answers the different question "WHICH kernel is taking the
time" -- needed to split a CPX-vs-SPX latency gap across the QKV kernel, the
HIP reduce kernel, and the Triton cross-XCD merge, without modelling anything.

Uses --kernel-trace, NOT --pmc. Three reasons that matter here:
  * PMC replay re-runs dispatches and distorts durations; kernel-trace does not.
  * FETCH_SIZE/WRITE_SIZE need two passes to fit the counter budget. Timing
    needs one.
  * We want EVERY kernel, including the ones a bandwidth filter deliberately
    excludes (the spin-wait barrier, RCCL, cache fills) -- their cost is exactly
    what is in question.

Aggregation
-----------
Ranks on one physical GPU run concurrently, so per-kernel times are reduced
across ranks with MAX, not SUM -- the same convention the CPX harness uses for
step latency (the slowest XCD sets the step time). SUM across ranks would
report 8 XCDs' concurrent work as if it were serial. The mean is printed
alongside so a skewed rank is visible rather than hidden behind the max.

  SPX : 1 db  -> max == mean == the single rank's total
  CPX : 8 dbs -> one per XCD rank
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
import sys
from collections import defaultdict


def _resolve(con, *candidates):
    """First existing table/view name from candidates (incl. suffixed base).

    rocpd_* objects are tables or views depending on the rocprofv3 version, and
    some builds suffix them. Same probe parse_rocprof_bandwidth.py uses.
    """
    have = {
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }
    for c in candidates:
        if c in have:
            return c
    for c in candidates:
        for name in have:
            if name.startswith(c + "_"):
                return name
    return None


def _kernel_name_id_cols(con, ksym):
    if not ksym:
        return None, None
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{ksym}")')]
    name_col = next(
        (
            c
            for c in (
                "kernel_name",
                "formatted_kernel_name",
                "display_name",
                "demangled_kernel_name",
                "name",
            )
            if c in cols
        ),
        None,
    )
    id_col = next((c for c in ("id", "kernel_id") if c in cols), None)
    return name_col, id_col


def _gpu_agent_ids(con):
    ids = set()
    if _resolve(con, "rocpd_info_agent"):
        try:
            for (aid,) in con.execute(
                "SELECT id FROM rocpd_info_agent WHERE upper(type)='GPU'"
            ):
                ids.add(int(aid))
        except sqlite3.OperationalError:
            pass
    return ids


# Collapse C++ template instantiations to something readable: the reduce kernel
# appears as paged_attention_ll4mi_reduce_kernel<__hip_bfloat16, ..., 1> and the
# NPAR_LOOPS/JCHUNK value in that parameter list is precisely what we may end up
# changing, so keep the template args in a --raw dump but strip them by default.
_TEMPLATE_ARGS = re.compile(r"<.*>", re.DOTALL)


def _short_name(name: str) -> str:
    n = name.split("(")[0]
    n = _TEMPLATE_ARGS.sub("", n)
    return n.strip().split("::")[-1] or name


def parse_db(db_path: str, raw: bool):
    """Return {kernel_name: [n_dispatches, total_ns]} for the busiest GPU agent.

    One db can hold several agents (rocprof sees every device the process
    touched). Under torchrun each rank's process drives one XCD, so keying on
    the agent with the most kernel time picks that rank's device and drops
    idle ones.
    """
    con = sqlite3.connect(db_path)
    try:
        ksym = _resolve(con, "kernel_symbols", "rocpd_info_kernel_symbol")
        name_col, id_col = _kernel_name_id_cols(con, ksym)
        if not (ksym and name_col and id_col):
            raise RuntimeError("no resolvable kernel-symbol table")
        if not _resolve(con, "rocpd_kernel_dispatch"):
            raise RuntimeError("no rocpd_kernel_dispatch table/view")

        gpu_ids = _gpu_agent_ids(con)
        agent_filter = ""
        if gpu_ids:
            agent_filter = " WHERE kd.agent_id IN (%s)" % ",".join(
                str(i) for i in sorted(gpu_ids)
            )

        rows = con.execute(
            f'SELECT kd.agent_id, ks."{name_col}", COUNT(*), '
            f'SUM(kd."end" - kd.start) '
            f"FROM rocpd_kernel_dispatch kd "
            f'JOIN {ksym} ks ON kd.kernel_id = ks."{id_col}"'
            + agent_filter
            + f' GROUP BY kd.agent_id, ks."{name_col}"'
        ).fetchall()
        if not rows:
            raise RuntimeError("no kernel dispatches recorded")

        per_agent: dict[int, dict[str, list]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0.0])
        )
        for aid, name, n, total in rows:
            key = name if raw else _short_name(name)
            slot = per_agent[int(aid)][key]
            slot[0] += int(n)
            slot[1] += float(total or 0.0)

        busiest = max(
            per_agent, key=lambda a: sum(v[1] for v in per_agent[a].values())
        )
        return dict(per_agent[busiest])
    finally:
        con.close()


def compare(in_csv: str, base: str, other: str):
    """Print base-vs-other per-kernel time, per cell, from the appended CSV.

    SPX and CPX cannot be profiled in the same boot of the compute partition,
    so the two runs land in this CSV separately and are joined here. A kernel
    present in one mode only (the merge and the barrier exist only under CPX)
    shows a blank on the missing side rather than a fabricated zero-delta.
    """
    import csv as _csv

    # (S, B) -> kernel -> label -> max_ms
    cells: dict[tuple[int, int], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with open(in_csv) as f:
        for row in _csv.DictReader(f):
            key = (int(row["seq_len"]), int(row["batch"]))
            cells[key][row["kernel"]][row["label"]] = float(row["max_ns"]) / 1e6

    for (s, b), kernels in sorted(cells.items()):
        tot = {
            lbl: sum(v.get(lbl, 0.0) for v in kernels.values())
            for lbl in (base, other)
        }
        print()
        print(f"=== S={s} B={b} ===  {base}={tot[base]:.3f} ms  "
              f"{other}={tot[other]:.3f} ms  "
              f"delta={tot[other] - tot[base]:+.3f} ms")
        print(f"{'kernel':<52} {base + '_ms':>10} {other + '_ms':>10} "
              f"{'delta_ms':>10}")
        print("-" * 86)
        # Sort by the delta, largest regression first: the point of this view is
        # which kernel is responsible for the gap, not which is biggest.
        def _delta(item):
            v = item[1]
            return -((v.get(other) or 0.0) - (v.get(base) or 0.0))

        for name, v in sorted(kernels.items(), key=_delta):
            disp = name if len(name) <= 52 else name[:49] + "..."
            a, o = v.get(base), v.get(other)
            sa = f"{a:>10.3f}" if a is not None else f"{'-':>10}"
            so = f"{o:>10.3f}" if o is not None else f"{'-':>10}"
            d = (o or 0.0) - (a or 0.0)
            print(f"{disp:<52} {sa} {so} {d:>+10.3f}")


def main():
    p = argparse.ArgumentParser(
        description="Per-kernel GPU time breakdown from rocprofv3 kernel-trace dbs"
    )
    p.add_argument(
        "--db-glob",
        help="glob for ALL kernel-trace .db of ONE cell (all ranks), "
             "e.g. '<cell_dir>/**/*.db'",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="skip parsing; pivot an existing --in-csv into a per-cell "
             "base-vs-other per-kernel table",
    )
    p.add_argument("--in-csv", default="", help="CSV to read for --compare")
    p.add_argument("--base-label", default="spx")
    p.add_argument("--other-label", default="cpx")
    p.add_argument("--label", default="", help="row label for the CSV (e.g. spx/cpx)")
    p.add_argument("--seq-len", type=int, default=0)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--out-csv", default="")
    p.add_argument(
        "--raw",
        action="store_true",
        help="keep full mangled names incl. template args (shows NPAR_LOOPS)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=15,
        help="print only the N slowest kernels (0 = all)",
    )
    args = p.parse_args()

    if args.compare:
        if not args.in_csv:
            raise SystemExit("--compare needs --in-csv")
        compare(args.in_csv, args.base_label, args.other_label)
        return
    if not args.db_glob:
        raise SystemExit("--db-glob is required (or use --compare --in-csv)")

    dbs = sorted(glob.glob(args.db_glob, recursive=True))
    if not dbs:
        print(f"[kern] ERROR: no db matched {args.db_glob}", file=sys.stderr)
        raise SystemExit(1)

    # per kernel -> list of per-rank totals, and per-rank dispatch counts
    totals: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, list[int]] = defaultdict(list)
    n_ok = 0
    for db in dbs:
        try:
            per_kernel = parse_db(db, args.raw)
        except Exception as e:  # noqa: BLE001
            print(f"[kern] WARN: {db}: {e}", file=sys.stderr)
            continue
        n_ok += 1
        for name, (n, total) in per_kernel.items():
            totals[name].append(total)
            counts[name].append(n)

    if n_ok == 0:
        print(f"[kern] ERROR: nothing parsed from {len(dbs)} db(s)", file=sys.stderr)
        raise SystemExit(1)

    # MAX across ranks: the XCDs are concurrent, so the slowest one sets the
    # step time. Summing would present concurrent work as serial.
    rows = []
    for name, per_rank in totals.items():
        mx = max(per_rank)
        mean = sum(per_rank) / len(per_rank)
        n = max(counts[name])
        rows.append((name, len(per_rank), n, mx, mean, mx / n if n else 0.0))
    rows.sort(key=lambda r: -r[3])

    step_ns = sum(r[3] for r in rows)
    print(
        f"[kern] label={args.label or '(none)'} S={args.seq_len} B={args.batch} "
        f"dbs={n_ok}  GPU-time(max-across-ranks, all kernels)="
        f"{step_ns / 1e6:.3f} ms"
    )
    print(
        f"{'kernel':<52} {'ranks':>5} {'disp':>6} "
        f"{'max_ms':>10} {'mean_ms':>10} {'us/disp':>9} {'%':>6}"
    )
    print("-" * 104)
    shown = rows if args.top <= 0 else rows[: args.top]
    for name, nranks, n, mx, mean, per in shown:
        disp = name if len(name) <= 52 else name[:49] + "..."
        pct = 100.0 * mx / step_ns if step_ns else 0.0
        print(
            f"{disp:<52} {nranks:>5} {n:>6} "
            f"{mx / 1e6:>10.3f} {mean / 1e6:>10.3f} {per / 1e3:>9.2f} {pct:>6.1f}"
        )
    if args.top > 0 and len(rows) > args.top:
        rest = sum(r[3] for r in rows[args.top:])
        print(f"{'(' + str(len(rows) - args.top) + ' more)':<52} "
              f"{'':>5} {'':>6} {rest / 1e6:>10.3f}")

    if args.out_csv:
        write_header = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a") as f:
            if write_header:
                f.write(
                    "label,seq_len,batch,kernel,n_ranks,n_dispatch,"
                    "max_ns,mean_ns,ns_per_dispatch\n"
                )
            for name, nranks, n, mx, mean, per in rows:
                safe = name.replace(",", ";")
                f.write(
                    f"{args.label},{args.seq_len},{args.batch},{safe},"
                    f"{nranks},{n},{mx:.0f},{mean:.0f},{per:.0f}\n"
                )
        print(f"[kern] appended {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
