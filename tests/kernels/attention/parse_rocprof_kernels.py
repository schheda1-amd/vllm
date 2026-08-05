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

The two barriers are separated, and so are floor and skew
---------------------------------------------------------
``fused_paged_merge`` dispatches ONE barrier kernel symbol twice per decode
step, on either side of the merge. Left folded into a single row they average
together, which is the one shape that cannot answer whether the barriers are
worth attacking: barrier 1 guards a cross-device RAW and could be deleted by a
put-based producer, while barrier 2 guards buffer reuse and stays regardless.
:func:`_split_barrier_phases` tells them apart by position relative to the merge
kernel.

Each barrier is then split again, across ranks rather than across dispatches.
The rank that arrives LAST barely waits, so its per-dispatch median is close to
the barrier's irreducible arrive/spin/release cost; the spread up to the slowest
rank is time spent waiting for peers. Hence the ``minrank_us`` column beside
``max_us``: floor and skew respond to completely different fixes. Removing a
barrier removes the floor. It does not remove the skew -- that is work
imbalance, and a put-based scheme can only let transfers hide behind it.
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

# What a decoded mangled component must look like to be believed.
_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")

# One source of truth for the CSV schema: --out-csv writes it, --compare reads
# it. SPX and CPX cannot be profiled in one boot of the compute partition, so
# this file is APPENDED to across runs that may be days and a parser revision
# apart. Both ends therefore check the header rather than trusting it.
CSV_HEADER = (
    "label,seq_len,batch,kernel,n_ranks,n_steps,n_disp_win,n_disp_all,"
    "per_step_ns,total_max_ns,total_mean_ns,med_ns_per_dispatch,"
    "max_ns_per_dispatch,min_ns_per_dispatch"
)

# The schema written before per-dispatch medians were carried. Recognised only
# so the error can name it -- its rows are deliberately NOT auto-migrated. The
# old ns_per_dispatch is a MEAN, and a mean over dispatches is exactly the
# statistic one out-of-loop barrier of ~100 ms wrecks; silently sliding it into
# the median column would produce a table that looks fine and is wrong.
_LEGACY_HEADERS = {
    "label,seq_len,batch,kernel,n_ranks,n_dispatch,max_ns,mean_ns,ns_per_dispatch",
    # Pre-window schema: its med_ns_per_dispatch is a real median, but summing
    # it across kernels assumed one dispatch per step, which charges the step
    # with one-time KV-cache fill. Not convertible without the dispatch counts.
    "label,seq_len,batch,kernel,n_ranks,n_dispatch,"
    "total_max_ns,total_mean_ns,med_ns_per_dispatch,max_ns_per_dispatch",
    # Pre-min schema. Every column it has is still correct, but its barrier rows
    # are the *unsplit* symbol -- one row covering both call sites -- and it
    # carries no min-across-ranks column, so neither barrier-1-vs-barrier-2 nor
    # overhead-vs-skew can be recovered from it. Re-parsing the dbs is cheap.
    "label,seq_len,batch,kernel,n_ranks,n_steps,n_disp_win,n_disp_all,"
    "per_step_ns,total_max_ns,total_mean_ns,med_ns_per_dispatch,"
    "max_ns_per_dispatch",
}


def _csv_header_of(path: str) -> str:
    with open(path) as f:
        return f.readline().strip()


def _schema_error(path: str, found: str) -> SystemExit:
    what = (
        "the schema of an older parser revision"
        if found in _LEGACY_HEADERS
        else "an unrecognised schema"
    )
    return SystemExit(
        f"ERROR: {path} has {what}.\n"
        f"  found:    {found}\n"
        f"  expected: {CSV_HEADER}\n"
        "\n"
        "These rows are not auto-converted: the old per-dispatch columns cannot\n"
        "be turned into per-step costs without the dispatch counts this schema\n"
        "carries, and guessing would silently change what the table means.\n"
        "Delete the CSV and re-parse the .db files\n"
        "already on disk -- this costs no re-profiling and no partition flip:\n"
        f"  rm -f {path}\n"
        "  ./run_paged_kerntime.sh reparse kern_spx_<ts> spx\n"
        "  ./run_paged_kerntime.sh reparse kern_cpx_<ts> cpx\n"
        "  ./run_paged_kerntime.sh compare"
    )


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
        if ln == 0 or len(n) - j < ln:  # length prefix overruns the string
            return name
        part = n[j : j + ln]
        if not _IDENT.fullmatch(part):
            return name
        parts.append(part)
        i = j + ln
        if not nested:  # free function: one component, the rest is arg types
            break
    if not parts:
        return name
    # A nested name must close with 'E'. Landing anywhere else means a length
    # prefix was misread and the "identifier" is a fragment of the neighbouring
    # component -- returning that fragment is worse than not decoding at all,
    # because the spin/collective regex matches on this string and a truncated
    # name silently reclassifies a peer-wait barrier as work.
    if nested and not n.startswith("E", i):
        return name
    return parts[-1]


def _short_name(name: str) -> str:
    n = _demangle(name)
    n = n.split("(")[0]
    n = _TEMPLATE_ARGS.sub("", n)
    return n.strip().split("::")[-1] or name


# Kernels whose duration is dominated by waiting on a peer rather than by work.
# Reported separately from the busy total; see the module docstring.
_SPIN_DEFAULT = r"signal_pad_barrier|ncclDevKernel|ncclKernel|rccl"

# fused_paged_merge calls the SAME barrier kernel twice per decode step, so both
# call sites land on one row and their very different costs are averaged into a
# number that answers nothing. They are told apart by position relative to the
# merge kernel; see _split_barrier_phases.
_BARRIER_DEFAULT = r"signal_pad_barrier"
_MERGE_DEFAULT = r"fused_paged_merge"

_PHASE_TAG = {1: " [1: pre-merge]", 2: " [2: post-merge]"}

# The kernel the steady-state window is anchored on: dispatched exactly once per
# decode step by both harnesses, under either partition mode, and present in
# every cell. Naming it beats inferring it -- see the comment in parse_db.
_ANCHOR_DEFAULT = r"paged_attention_ll4mi_QKV"


def _split_barrier_phases(ka, barrier_re, merge_re):
    """Split the one barrier symbol into its two per-step call sites.

    ``starscream_paged_symm_merge.fused_paged_merge`` dispatches the same
    ``_signal_pad_barrier_kernel`` before and after the merge:

        reduce -> barrier 1 -> merge -> barrier 2 -> (next step) barrier 1 -> ...

    They are not interchangeable. Barrier 1 is a cross-device RAW guard that a
    put-based producer could delete outright; barrier 2 is a WAR guard on buffer
    reuse that stays. Reported as one row they average together, which is
    precisely the number that cannot inform whether the put is worth building.

    Classification is by ORDER, not parity: for each barrier dispatch, look at
    the next barrier-or-merge event on the same agent. A merge next means this
    was barrier 1; another barrier next means the merge already happened and
    this is barrier 2. Parity would work only if the steady-state window always
    opened on the same call site, which it does not -- the window is anchored on
    the QKV kernel and clips whole steps, so which barrier lands first depends
    on the cell. The tail dispatch, which has no successor inside the trace,
    falls back to the mirrored backward rule.

    Returns ``ka`` unchanged when either kernel is absent (the SPX side has
    neither), so this is a no-op on the baseline.
    """
    import bisect

    bnames = [n for n in ka if barrier_re.search(n)]
    mnames = [n for n in ka if merge_re.search(n)]
    if not bnames or not mnames:
        return ka

    merge_starts = sorted(s for n in mnames for s, _ in ka[n])
    barrier_starts = sorted(s for n in bnames for s, _ in ka[n])

    def _after(xs, t):
        i = bisect.bisect_right(xs, t)
        return xs[i] if i < len(xs) else None

    def _before(xs, t):
        i = bisect.bisect_left(xs, t)
        return xs[i - 1] if i > 0 else None

    def _phase(t):
        nm, nb = _after(merge_starts, t), _after(barrier_starts, t)
        if nm is not None and (nb is None or nm < nb):
            return 1
        if nb is not None:
            return 2
        # No successor: this is the last barrier in the trace. Mirror the rule
        # backwards -- if the most recent event was the merge, we are behind it.
        pm, pb = _before(merge_starts, t), _before(barrier_starts, t)
        if pm is not None and (pb is None or pm > pb):
            return 2
        return 1

    out = {n: v for n, v in ka.items() if n not in bnames}
    for n in bnames:
        for s, e in ka[n]:
            out.setdefault(n + _PHASE_TAG[_phase(s)], []).append((s, e))
    return out


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


def parse_db(
    db_path: str,
    raw: bool,
    spin_re: str,
    warmup: int,
    barrier_re: str = _BARRIER_DEFAULT,
    merge_re: str = _MERGE_DEFAULT,
    anchor_re: str = _ANCHOR_DEFAULT,
):
    """Return per-kernel timings restricted to the steady-state window.

    ``({kernel: (n_win, total_win, med, max, n_all)}, busy, n_steps, anchor,
    (n_outside, ns_outside))``.

    One db can hold several agents (rocprof sees every device the process
    touched). Under torchrun each rank's process drives one XCD, so keying on
    the agent with the most kernel time picks that rank's device and drops
    idle ones.

    Durations come back per dispatch rather than pre-SUMmed: a total alone
    cannot distinguish "expensive every step" from "one 100 ms outlier at the
    end", and the per-step cost needs the dispatch COUNT, not just the sum.

    Counting dispatches is what separates a per-step kernel from setup. In a
    20-iteration run the attention kernels are dispatched once per step while
    the RNG that fills the KV cache is dispatched 2-4 times total and the
    allocator's zero-fill hundreds of times, none of them in the timed region.
    Treating every kernel as once-per-step charges the step with megabytes of
    one-time cache fill and inflates both sides of an SPX-vs-CPX comparison.
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
            f'SELECT kd.agent_id, ks."{name_col}", kd.start, kd."end" '
            f"FROM rocpd_kernel_dispatch kd "
            f'JOIN {ksym} ks ON kd.kernel_id = ks."{id_col}"' + agent_filter
        ).fetchall()
        if not rows:
            raise RuntimeError("no kernel dispatches recorded")

        per_agent: dict[int, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        for aid, name, start, end in rows:
            key = name if raw else _short_name(name)
            per_agent[int(aid)][key].append((int(start), int(end)))

        busiest = max(
            per_agent,
            key=lambda a: sum(
                sum(e - s for s, e in v) for v in per_agent[a].values()
            ),
        )
        ka = per_agent[busiest]

        # Before anything is measured, separate the two barrier call sites.
        # Both halves still match the spin regex, so their classification as
        # peer-wait rather than work is unchanged.
        if barrier_re and merge_re:
            ka = _split_barrier_phases(
                ka, re.compile(barrier_re, re.I), re.compile(merge_re, re.I)
            )

        # Anchor the steady-state window on the once-per-step QKV kernel. The
        # profile spans the WHOLE process -- KV-cache fill, allocator zeroing,
        # RNG -- while the benchmark number covers only the timed loop, so a
        # per-step total that includes setup answers a question nobody asked.
        # Anchoring on a per-step kernel rather than on wall clock keeps this
        # independent of how long setup happened to take.
        #
        # The anchor is NAMED, not inferred as "busiest non-spin kernel". That
        # heuristic holds only while attention dominates, and the cells where it
        # does not are exactly the ones being profiled: at 8192/B1 a rank owns
        # 1024 tokens and its QKV kernel is a few microseconds, which the
        # KV-cache fill RNG -- hundreds of milliseconds, dispatched 2-4 times in
        # the whole run -- outweighs by orders of magnitude. Anchoring there
        # yields a 1-3 dispatch "window" spanning setup, n_steps in the single
        # digits, and per-step costs off by the ratio of the two. The failure is
        # silent: every column still prints, plausibly.
        spin = re.compile(spin_re, re.I) if spin_re else None
        anchor = None
        if anchor_re:
            arx = re.compile(anchor_re, re.I)
            hits = [n for n in ka if arx.search(n)]
            if hits:
                # Several QKV instantiations can coexist in one trace (mfma4 for
                # gqa_ratio <= 4, mfma16 above). Take the most-dispatched, and
                # on a tie the busiest -- the one that ran every step.
                anchor = max(
                    hits, key=lambda n: (len(ka[n]), sum(e - s for s, e in ka[n]))
                )
        if anchor is None:
            cand = [n for n in ka if not (spin and spin.search(n))] or list(ka)
            anchor = max(cand, key=lambda n: sum(e - s for s, e in ka[n]))
        starts = sorted(s for s, _ in ka[anchor])
        # Drop the harness's warmup iterations: they run before the timed
        # region and are the slow, cold-cache ones.
        skip = warmup if len(starts) - warmup >= 2 else 0
        if len(starts) - skip >= 2:
            # Half-open [first anchor start, LAST anchor start): whole steps
            # only. Closing on the last anchor's *end* instead would cut the
            # final step's trailing kernels -- the reduce and the merge run
            # after the QKV they belong to -- so those kernels would show
            # n-1 dispatches over n steps and come out ~5% cheap. Dropping the
            # final partial step costs one sample and biases nothing, because
            # the divisor drops with it.
            win_start, win_end = starts[skip], starts[-1]
            n_steps = len(starts) - skip - 1
        else:  # too few dispatches to bound a step; take everything
            win_start = starts[0]
            win_end = max(e for _, e in ka[anchor]) + 1
            n_steps = len(starts)

        out = {}
        busy = 0.0
        outside_n = 0
        outside_ns = 0.0
        for name, disp in ka.items():
            durs = [float(e - s) for s, e in disp if win_start <= s < win_end]
            if not durs:  # setup/teardown only: no dispatch in the timed region
                outside_n += 1
                outside_ns += sum(float(e - s) for s, e in disp)
                continue
            out[name] = (len(durs), sum(durs), _median(durs), max(durs), len(disp))
            busy += sum(durs)
        return out, busy, n_steps, anchor, (outside_n, outside_ns)
    finally:
        con.close()


def _barrier_summary(rows, n_steps):
    """Print the barrier-1 / barrier-2 split and its overhead-vs-skew shape.

    Only fires when :func:`_split_barrier_phases` found both call sites, i.e.
    on the CPX side. The two questions it answers, in the order they matter:

      * how much of the per-step cost is barrier 1 -- the one a put-based
        producer could delete -- versus barrier 2, which is a WAR guard on
        buffer reuse and stays either way;
      * how much of each barrier is irreducible cost versus rank skew. The
        rank that arrives LAST barely waits, so its per-dispatch median is
        close to the pure arrive/spin/release cost; the spread up to the
        slowest rank is peers it had to wait for. Deleting a barrier removes
        the first part. It does not remove the second -- skew is work
        imbalance, and a put-based scheme only lets transfers hide behind it.
    """
    phases = {}
    for r in rows:
        for ph, tag in _PHASE_TAG.items():
            if r[0].endswith(tag):
                phases[ph] = r
    if len(phases) != 2:
        return
    print()
    print("  barrier call sites (same kernel symbol, split by position):")
    total = 0.0
    for ph in sorted(phases):
        name, _nranks, n, mx, _mean, med, _dmax, _sp, per_step, _na, dmin = (
            phases[ph]
        )
        total += per_step
        skew = max(0.0, med - dmin)
        role = "RAW, cross-device read-after-write" if ph == 1 else \
               "WAR, buffer reuse next step"
        print(
            f"    barrier {ph} ({role}):\n"
            f"      {per_step / 1e3:8.2f} us/step over {n / n_steps:.1f} "
            f"dispatch(es); floor {dmin / 1e3:.2f} us + skew "
            f"{skew / 1e3:.2f} us"
        )
    print(f"    both barriers: {total / 1e3:.2f} us/step")


def compare(in_csv: str, base: str, other: str, spin_re: str):
    """Print base-vs-other per-kernel time, per cell, from the appended CSV.

    SPX and CPX cannot be profiled in the same boot of the compute partition,
    so the two runs land in this CSV separately and are joined here. A kernel
    present in one mode only (the merge and the barrier exist only under CPX)
    shows a blank on the missing side rather than a fabricated zero-delta.

    Leads with us PER STEP, not per dispatch, which is what maps onto the
    harness's us/step. The distinction is not pedantic: the KV-cache RNG is
    dispatched 2-4 times in a 20-step run and the allocator's zero-fill several
    hundred, so a per-dispatch column summed down the table reports a step cost
    that includes one-time setup. Spin/collective kernels are listed but
    excluded from the totals, and marked with a ``*`` -- see the module
    docstring for why folding a peer-wait into a work comparison misleads.
    """
    import csv as _csv

    spin = re.compile(spin_re, re.I) if spin_re else None

    if not os.path.exists(in_csv) or os.path.getsize(in_csv) == 0:
        raise SystemExit(
            f"ERROR: {in_csv} is missing or empty -- run the spx and cpx passes first"
        )
    header = _csv_header_of(in_csv)
    if header != CSV_HEADER:
        raise _schema_error(in_csv, header)

    # (S, B) -> kernel -> label -> (per_step_ns, total_ns, disp_per_step)
    cells: dict[tuple[int, int], dict[str, dict[str, tuple]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with open(in_csv) as f:
        for row in _csv.DictReader(f):
            key = (int(row["seq_len"]), int(row["batch"]))
            cells[key][row["kernel"]][row["label"]] = (
                float(row["per_step_ns"]),
                float(row["total_max_ns"]),
                int(row["n_disp_win"]) / max(1, int(row["n_steps"])),
            )

    for (s, b), kernels in sorted(cells.items()):
        def _is_spin(name):
            return bool(spin and spin.search(name))

        def _ps(v, lbl):  # per-step ns for label, or None
            return v[lbl][0] if lbl in v else None

        tot = {
            lbl: sum(
                (_ps(v, lbl) or 0.0)
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
            f"{'kernel':<46} {'d/stp':>6} {base + '_us':>10} "
            f"{other + '_us':>10} {'delta_us':>10} {base + '_ms':>9} "
            f"{other + '_ms':>9}"
        )
        print("-" * 105)

        # Sort by the delta, largest regression first: the point of this view is
        # which kernel is responsible for the gap, not which is biggest.
        def _delta(item):
            v = item[1]
            return -((_ps(v, other) or 0.0) - (_ps(v, base) or 0.0))

        for name, v in sorted(kernels.items(), key=_delta):
            mark = "*" if _is_spin(name) else " "
            disp = name if len(name) <= 45 else name[:42] + "..."
            a, o = _ps(v, base), _ps(v, other)
            sa = f"{a / 1e3:>10.1f}" if a is not None else f"{'-':>10}"
            so = f"{o / 1e3:>10.1f}" if o is not None else f"{'-':>10}"
            d = ((o or 0.0) - (a or 0.0)) / 1e3
            ta = v[base][1] / 1e6 if base in v else None
            to = v[other][1] / 1e6 if other in v else None
            sta = f"{ta:>9.3f}" if ta is not None else f"{'-':>9}"
            sto = f"{to:>9.3f}" if to is not None else f"{'-':>9}"
            dps = v[other][2] if other in v else v[base][2]
            print(
                f"{mark}{disp:<45} {dps:>6.1f} {sa} {so} {d:>+10.1f} "
                f"{sta} {sto}"
            )
        if spin:
            print("  * spin/collective: time is waiting on a peer, not work; "
                  "excluded from the us/step totals above")
        print("  d/stp = dispatches per step; costs are per step, so a kernel "
              "that does not run every step contributes its true share")


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
        "--barrier-regex",
        default=_BARRIER_DEFAULT,
        help="barrier kernel to split into its pre-merge and post-merge call "
             "sites. Pass '' to leave the two folded into one row.",
    )
    p.add_argument(
        "--merge-regex",
        default=_MERGE_DEFAULT,
        help="kernel that separates the two barrier call sites; the split is "
             "by position relative to it",
    )
    p.add_argument(
        "--anchor-regex",
        default=_ANCHOR_DEFAULT,
        help="once-per-step kernel the steady-state window is anchored on. "
             "Pass '' to fall back to 'busiest non-spin kernel', which "
             "misfires at small batch where setup outweighs attention.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=15,
        help="print only the N slowest kernels (0 = all)",
    )
    p.add_argument(
        "--num-iters",
        type=int,
        default=0,
        help="the harness's --num-iters, used only to sanity-check the derived "
             "step count (0 = skip the check)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="anchor-kernel dispatches to drop from the front of the steady-"
             "state window. Default 3 matches the warmup loop in "
             "test_harness_1_paged*.py; change both together or the per-step "
             "cost includes cold-cache iterations the benchmark does not time.",
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
    counts_all: dict[str, list[int]] = defaultdict(list)
    busies: list[float] = []
    steps: list[int] = []
    anchors: set[str] = set()
    outside = [0, 0.0]
    n_ok = 0
    for db in dbs:
        try:
            per_kernel, busy, n_steps, anchor, outs = parse_db(
                db, args.raw, args.spin_regex, args.warmup,
                args.barrier_regex, args.merge_regex, args.anchor_regex,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[kern] WARN: {db}: {e}", file=sys.stderr)
            continue
        n_ok += 1
        busies.append(busy)
        steps.append(n_steps)
        anchors.add(anchor)
        outside[0] = max(outside[0], outs[0])
        outside[1] = max(outside[1], outs[1])
        for name, (n, total, med, mx, n_all) in per_kernel.items():
            totals[name].append(total)
            meds[name].append(med)
            maxes[name].append(mx)
            counts[name].append(n)
            counts_all[name].append(n_all)

    if n_ok == 0:
        print(f"[kern] ERROR: nothing parsed from {len(dbs)} db(s)", file=sys.stderr)
        raise SystemExit(1)

    # Ranks run the same loop, so a disagreement here means one rank's trace is
    # truncated -- the per-step divisor would then differ per rank. Use the max
    # and say so rather than silently averaging over a broken rank.
    n_steps = max(steps)
    if len(set(steps)) > 1:
        print(
            f"[kern] WARN: ranks disagree on step count {sorted(set(steps))}; "
            f"using {n_steps}",
            file=sys.stderr,
        )

    # The window is the divisor for every per-step number below, so check it
    # against what the harness ran instead of trusting it. The timed loop is
    # num_iters steps and the half-open window keeps num_iters - 1 of them, so a
    # mismatch means the anchor is not the once-per-step kernel -- the failure
    # mode is a plausible-looking table scaled by an arbitrary factor, which is
    # worse than an error. Only checked when --num-iters is passed, since the
    # parser cannot otherwise know it.
    if args.num_iters and n_steps != args.num_iters - 1:
        print(
            f"[kern] WARN: window has {n_steps} steps but --num-iters "
            f"{args.num_iters} implies {args.num_iters - 1}. The anchor "
            f"({', '.join(sorted(anchors))[:60]}) is probably not the "
            f"once-per-step kernel; per-step costs are scaled wrong. "
            f"Check --anchor-regex and --warmup.",
            file=sys.stderr,
        )

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
                max(counts[name]),  # dispatches inside the window
                max(per_rank),  # total, max across ranks
                sum(per_rank) / len(per_rank),  # total, mean across ranks
                max(meds[name]),  # median per dispatch, max across ranks
                max(maxes[name]),  # slowest single dispatch anywhere
                bool(spin and spin.search(name)),
                max(per_rank) / n_steps,  # per-step cost
                max(counts_all[name]),  # dispatches over the whole process
                # Median per dispatch, MIN across ranks. On a barrier this is
                # the rank that arrived last and therefore waited least, so it
                # approximates the barrier's irreducible cost; the spread up to
                # the max column is rank skew. On a work kernel the two columns
                # are close and the min means little.
                min(meds[name]),
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
        f"  steady-state window: {n_steps} steps, anchored on "
        f"{', '.join(sorted(anchors))[:52]} (warmup {args.warmup} dropped)"
    )
    if outside[0]:
        print(
            f"  excluded as setup/teardown: {outside[0]} kernel(s), "
            f"{outside[1] / 1e6:.3f} ms with no dispatch in the window"
        )
    print(
        f"  rank GPU-busy in window (per-rank total)   : "
        f"max {max(busies) / 1e6:.3f} ms   "
        f"mean {sum(busies) / len(busies) / 1e6:.3f} ms"
    )
    print(
        f"  sum of per-kernel maxes                    : "
        f"{sum_of_maxes / 1e6:.3f} ms   "
        f"(work {work_ns / 1e6:.3f} + spin/collective {spin_ns / 1e6:.3f})"
    )
    print(
        f"  work per step                              : "
        f"{work_ns / n_steps / 1e3:.1f} us"
    )
    print(
        f"{'kernel':<46} {'ranks':>5} {'d/stp':>6} {'per_step_us':>12} "
        f"{'tot_max_ms':>11} {'med_us':>9} {'max_us':>9} {'minrank_us':>10}"
    )
    print("-" * 119)
    shown = rows if args.top <= 0 else rows[: args.top]
    for (
        name, nranks, n, mx, mean, med, dmax, is_spin, per_step, n_all, dmin
    ) in shown:
        disp = name if len(name) <= 45 else name[:42] + "..."
        mark = "*" if is_spin else " "
        print(
            f"{mark}{disp:<45} {nranks:>5} {n / n_steps:>6.1f} "
            f"{per_step / 1e3:>12.1f} {mx / 1e6:>11.3f} "
            f"{med / 1e3:>9.2f} {dmax / 1e3:>9.2f} {dmin / 1e3:>10.2f}"
        )
    if args.top > 0 and len(rows) > args.top:
        rest = sum(r[3] for r in rows[args.top:])
        print(f" {'(' + str(len(rows) - args.top) + ' more)':<45} "
              f"{'':>5} {'':>6} {rest / n_steps / 1e3:>12.1f} {rest / 1e6:>11.3f}")
    if spin:
        print("  * spin/collective: duration is dominated by waiting on a peer "
              "(and, for an out-of-loop collective, by rank skew), not by work")
    print("  med_us/max_us are the per-dispatch median taken across ranks with "
          "max; minrank_us takes it with min")

    _barrier_summary(rows, n_steps)

    if args.out_csv:
        # Append, so check the existing header BEFORE writing: appending
        # 10-field rows under a 9-field header corrupts the file quietly and
        # the failure only shows up later, in --compare, as a KeyError.
        # Treat a zero-byte file as absent (an interrupted earlier run).
        have = os.path.exists(args.out_csv) and os.path.getsize(args.out_csv) > 0
        if have:
            header = _csv_header_of(args.out_csv)
            if header != CSV_HEADER:
                raise _schema_error(args.out_csv, header)
        with open(args.out_csv, "a") as f:
            if not have:
                f.write(CSV_HEADER + "\n")
            for (
                name, nranks, n, mx, mean, med, dmax, _, per_step, n_all, dmin
            ) in rows:
                safe = name.replace(",", ";")
                f.write(
                    f"{args.label},{args.seq_len},{args.batch},{safe},"
                    f"{nranks},{n_steps},{n},{n_all},{per_step:.0f},"
                    f"{mx:.0f},{mean:.0f},{med:.0f},{dmax:.0f},{dmin:.0f}\n"
                )
        print(f"[kern] appended {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
