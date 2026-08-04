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

Aggregation, and why the headline is not a sum
----------------------------------------------
Ranks on one physical GPU run concurrently, so per-kernel times are reduced
across ranks with MAX, not SUM -- the same convention the CPX harness uses for
step latency (the slowest XCD sets the step time). SUM across ranks would
report 8 XCDs' concurrent work as if it were serial.

  SPX : 1 db  -> max == mean == the single rank's total
  CPX : 8 dbs -> one per XCD rank

But *summing those per-kernel maxes* is not the step time either, and getting
that wrong is easy: while rank A runs its QKV kernel, rank B is parked in the
signal-pad barrier waiting for it. Taking the max of QKV across ranks AND the
max of the barrier across ranks counts that wait twice -- once as work, once as
waiting for the same work. So the headline is the max over ranks of each rank's
OWN total busy time, which cannot double-count by construction; the sum of
per-kernel maxes is printed beside it, and a large gap between the two is
itself the rank-skew signal.

For the same reason a "busy excluding spin/collective" line is printed. Time in
``signal_pad_barrier`` / ``ncclDevKernel`` is overwhelmingly *waiting on a
peer*, plus -- for a collective outside the timed loop -- waiting for the
slowest rank to finish the entire benchmark. It is real GPU occupancy and real
wall clock, but it is not this rank's work, so it belongs on its own line
rather than folded into a per-kernel cost comparison.

Per-dispatch median, not just the total
---------------------------------------
Totals mix warmup, one-time setup (KV-cache fill, RNG) and the steady state.
A single end-of-run barrier that spins for 100 ms is one dispatch out of ~46
and would swamp a mean. The median per dispatch is what maps onto the harness's
reported us/step, so it is the column the compare view leads with.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
import sys
from collections import defaultdict

# Collapse C++ template instantiations to something readable: the reduce kernel
# appears as paged_attention_ll4mi_reduce_kernel<__hip_bfloat16, ..., 8> and the
# NPAR_LOOPS value in that parameter list is precisely what we may end up
# changing, so keep the template args in a --raw dump but strip them by default.
#
# Stripping is not cosmetic here. SPX at 131072 instantiates NPAR_LOOPS=8 and
# CPX instantiates NPAR_LOOPS=1, so with the args kept the same kernel lands on
# two different rows of the compare table and reads as "SPX-only" and
# "CPX-only" instead of as the one row whose delta is the whole question.
_TEMPLATE_ARGS = re.compile(r"<.*>", re.DOTALL)

# Leading <length><identifier> component of an Itanium mangled name.
_MANGLED_PART = re.compile(r"(\d+)")


def _demangle(name: str) -> str:
    """Recover the top-level identifier from an Itanium-mangled symbol.

    rocprofv3 builds differ in whether the kernel-symbol table carries a
    demangled column at all; when it does not, every row reads as
    ``_Z35paged_attention_ll4mi_reduce_kernelI...`` and the template-arg strip
    below is a no-op because there is no ``<`` to strip. Rather than shell out
    to c++filt per row, decode just the name prefix -- ``_Z<len><name>`` for a
    free function, ``_ZN<len><ns><len><name>...E`` for a nested one -- which is
    all this tool ever displays. Anything unrecognised is returned unchanged.
    """
    n = name
    if n.endswith(".kd"):  # kernel-descriptor symbol, e.g. Triton's .kd suffix
        n = n[:-3]
    if not n.startswith("_Z"):
        return n
    i = 2
    nested = i < len(n) and n[i] == "N"
    if nested:
        i += 1
    parts = []
    while i < len(n) and n[i].isdigit():
        j = i
        while j < len(n) and n[j].isdigit():
            j += 1
        ln = int(n[i:j])
        parts.append(n[j : j + ln])
        i = j + ln
        if not nested:  # free function: one component, the rest is arg types
            break
    return parts[-1] if parts else n


def _short_name(name: str) -> str:
    n = _demangle(name)
    n = n.split("(")[0]
    n = _TEMPLATE_ARGS.sub("", n)
    return n.strip().split("::")[-1] or name


# Kernels whose duration is dominated by waiting on a peer rather than by work.
# Reported separately from the busy total; see the module docstring.
_SPIN_DEFAULT = r"signal_pad_barrier|ncclDevKernel|ncclKernel|rccl"


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
    """Pick the name column, preferring an already-demangled one.

    Order matters: ``kernel_name`` is mangled in the rocprofv3 builds seen here,
    so probing it first yields ``_Z35...`` for every row and defeats the
    template-arg strip. Try the demangled/formatted columns first and fall back
    to the mangled one, which :func:`_demangle` can still make readable.
    """
    if not ksym:
        return None, None
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{ksym}")')]
    name_col = next(
        (
            c
            for c in (
                "formatted_kernel_name",
                "demangled_kernel_name",
                "display_name",
                "kernel_name",
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


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def parse_db(db_path: str, raw: bool):
    """Return ({kernel: (n, total_ns, median_ns, max_ns)}, rank_busy_ns).

    One db can hold several agents (rocprof sees every device the process
    touched). Under torchrun each rank's process drives one XCD, so keying on
    the agent with the most kernel time picks that rank's device and drops
    idle ones.

    Durations come back per dispatch rather than pre-SUMmed: the median is the
    steady-state per-step cost, and a total alone cannot distinguish "expensive
    every step" from "one 100 ms outlier at the end".
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
            f'SELECT kd.agent_id, ks."{name_col}", (kd."end" - kd.start) '
            f"FROM rocpd_kernel_dispatch kd "
            f'JOIN {ksym} ks ON kd.kernel_id = ks."{id_col}"' + agent_filter
        ).fetchall()
        if not rows:
            raise RuntimeError("no kernel dispatches recorded")

        per_agent: dict[int, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        for aid, name, dur in rows:
            key = name if raw else _short_name(name)
            per_agent[int(aid)][key].append(float(dur or 0.0))

        busiest = max(
            per_agent,
            key=lambda a: sum(sum(v) for v in per_agent[a].values()),
        )
        out = {}
        busy = 0.0
        for name, durs in per_agent[busiest].items():
            out[name] = (len(durs), sum(durs), _median(durs), max(durs))
            busy += sum(durs)
        return out, busy
    finally:
        con.close()


def compare(in_csv: str, base: str, other: str, spin_re: str):
    """Print base-vs-other per-kernel time, per cell, from the appended CSV.

    SPX and CPX cannot be profiled in the same boot of the compute partition,
    so the two runs land in this CSV separately and are joined here. A kernel
    present in one mode only (the merge and the barrier exist only under CPX)
    shows a blank on the missing side rather than a fabricated zero-delta.

    Leads with median us/dispatch, which maps onto the harness's us/step and is
    immune to a single out-of-loop outlier. Spin/collective kernels are listed
    but excluded from the totals line, and marked with a ``*`` -- see the module
    docstring for why folding a peer-wait into a work comparison misleads.
    """
    import csv as _csv

    spin = re.compile(spin_re, re.I) if spin_re else None

    # (S, B) -> kernel -> label -> (med_ns, total_ns)
    cells: dict[tuple[int, int], dict[str, dict[str, tuple]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with open(in_csv) as f:
        for row in _csv.DictReader(f):
            key = (int(row["seq_len"]), int(row["batch"]))
            cells[key][row["kernel"]][row["label"]] = (
                float(row["med_ns_per_dispatch"]),
                float(row["total_max_ns"]),
            )

    for (s, b), kernels in sorted(cells.items()):
        def _is_spin(name):
            return bool(spin and spin.search(name))

        def _med(v, lbl):
            return v[lbl][0] if lbl in v else None

        tot = {
            lbl: sum(
                (_med(v, lbl) or 0.0)
                for n, v in kernels.items()
                if not _is_spin(n)
            )
            for lbl in (base, other)
        }
        print()
        print(
            f"=== S={s} B={b} ===  work-only us/step: "
            f"{base}={tot[base] / 1e3:.1f}  {other}={tot[other] / 1e3:.1f}  "
            f"delta={(tot[other] - tot[base]) / 1e3:+.1f}"
        )
        print(
            f"{'kernel':<46} {base + '_us':>10} {other + '_us':>10} "
            f"{'delta_us':>10} {base + '_ms':>9} {other + '_ms':>9}"
        )
        print("-" * 98)

        # Sort by the delta, largest regression first: the point of this view is
        # which kernel is responsible for the gap, not which is biggest.
        def _delta(item):
            v = item[1]
            return -((_med(v, other) or 0.0) - (_med(v, base) or 0.0))

        for name, v in sorted(kernels.items(), key=_delta):
            mark = "*" if _is_spin(name) else " "
            disp = name if len(name) <= 45 else name[:42] + "..."
            a, o = _med(v, base), _med(v, other)
            sa = f"{a / 1e3:>10.1f}" if a is not None else f"{'-':>10}"
            so = f"{o / 1e3:>10.1f}" if o is not None else f"{'-':>10}"
            d = ((o or 0.0) - (a or 0.0)) / 1e3
            ta = v[base][1] / 1e6 if base in v else None
            to = v[other][1] / 1e6 if other in v else None
            sta = f"{ta:>9.3f}" if ta is not None else f"{'-':>9}"
            sto = f"{to:>9.3f}" if to is not None else f"{'-':>9}"
            print(f"{mark}{disp:<45} {sa} {so} {d:>+10.1f} {sta} {sto}")
        if spin:
            print("  * spin/collective: time is waiting on a peer, not work; "
                  "excluded from the us/step totals above")


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
        "--spin-regex",
        default=_SPIN_DEFAULT,
        help="kernels whose time is peer-wait, not work; reported separately. "
             "Pass '' to disable.",
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
        compare(args.in_csv, args.base_label, args.other_label, args.spin_regex)
        return
    if not args.db_glob:
        raise SystemExit("--db-glob is required (or use --compare --in-csv)")

    dbs = sorted(glob.glob(args.db_glob, recursive=True))
    if not dbs:
        print(f"[kern] ERROR: no db matched {args.db_glob}", file=sys.stderr)
        raise SystemExit(1)

    totals: dict[str, list[float]] = defaultdict(list)
    meds: dict[str, list[float]] = defaultdict(list)
    maxes: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, list[int]] = defaultdict(list)
    busies: list[float] = []
    n_ok = 0
    for db in dbs:
        try:
            per_kernel, busy = parse_db(db, args.raw)
        except Exception as e:  # noqa: BLE001
            print(f"[kern] WARN: {db}: {e}", file=sys.stderr)
            continue
        n_ok += 1
        busies.append(busy)
        for name, (n, total, med, mx) in per_kernel.items():
            totals[name].append(total)
            meds[name].append(med)
            maxes[name].append(mx)
            counts[name].append(n)

    if n_ok == 0:
        print(f"[kern] ERROR: nothing parsed from {len(dbs)} db(s)", file=sys.stderr)
        raise SystemExit(1)

    spin = re.compile(args.spin_regex, re.I) if args.spin_regex else None

    # MAX across ranks per kernel: the XCDs are concurrent, so the slowest one
    # sets the step time. Summing across ranks would present concurrent work as
    # serial.
    rows = []
    for name, per_rank in totals.items():
        rows.append(
            (
                name,
                len(per_rank),
                max(counts[name]),
                max(per_rank),  # total, max across ranks
                sum(per_rank) / len(per_rank),  # total, mean across ranks
                max(meds[name]),  # median per dispatch, max across ranks
                max(maxes[name]),  # slowest single dispatch anywhere
                bool(spin and spin.search(name)),
            )
        )
    rows.sort(key=lambda r: -r[3])

    sum_of_maxes = sum(r[3] for r in rows)
    work_ns = sum(r[3] for r in rows if not r[7])
    spin_ns = sum_of_maxes - work_ns
    print(
        f"[kern] label={args.label or '(none)'} S={args.seq_len} "
        f"B={args.batch} dbs={n_ok}"
    )
    print(
        f"  rank GPU-busy (all kernels, per-rank total): "
        f"max {max(busies) / 1e6:.3f} ms   mean {sum(busies) / len(busies) / 1e6:.3f} ms"
    )
    print(
        f"  sum of per-kernel maxes                    : "
        f"{sum_of_maxes / 1e6:.3f} ms   "
        f"(work {work_ns / 1e6:.3f} + spin/collective {spin_ns / 1e6:.3f})"
    )
    print(
        f"{'kernel':<46} {'ranks':>5} {'disp':>5} {'tot_max_ms':>11} "
        f"{'tot_mean_ms':>11} {'med_us':>9} {'max_us':>9}"
    )
    print("-" * 108)
    shown = rows if args.top <= 0 else rows[: args.top]
    for name, nranks, n, mx, mean, med, dmax, is_spin in shown:
        disp = name if len(name) <= 45 else name[:42] + "..."
        mark = "*" if is_spin else " "
        print(
            f"{mark}{disp:<45} {nranks:>5} {n:>5} {mx / 1e6:>11.3f} "
            f"{mean / 1e6:>11.3f} {med / 1e3:>9.2f} {dmax / 1e3:>9.2f}"
        )
    if args.top > 0 and len(rows) > args.top:
        rest = sum(r[3] for r in rows[args.top:])
        print(f" {'(' + str(len(rows) - args.top) + ' more)':<45} "
              f"{'':>5} {'':>5} {rest / 1e6:>11.3f}")
    if spin:
        print("  * spin/collective: duration is dominated by waiting on a peer "
              "(and, for an out-of-loop collective, by rank skew), not by work")

    if args.out_csv:
        write_header = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a") as f:
            if write_header:
                f.write(
                    "label,seq_len,batch,kernel,n_ranks,n_dispatch,"
                    "total_max_ns,total_mean_ns,med_ns_per_dispatch,"
                    "max_ns_per_dispatch\n"
                )
            for name, nranks, n, mx, mean, med, dmax, _ in rows:
                safe = name.replace(",", ";")
                f.write(
                    f"{args.label},{args.seq_len},{args.batch},{safe},"
                    f"{nranks},{n},{mx:.0f},{mean:.0f},{med:.0f},{dmax:.0f}\n"
                )
        print(f"[kern] appended {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
