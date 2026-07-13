# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diagnostic: per-kernel time + memory traffic breakdown in a rocprofv3 .db.

Shows which kernels dominate duration and FETCH/WRITE bytes, so we can tell
whether the CPX bandwidth number is contaminated by the spin-wait barrier or
the RCCL allgather rather than the attention kernels themselves.

Usage:
    python inspect_kernels.py <db-or-glob>
    python inspect_kernels.py '/workspace/vllm/bw_llama3-70b_cpx_*/S8192_B1/FETCH_SIZE/**/*.db'
"""

import glob
import sqlite3
import sys


def _resolve(con, *candidates):
    """Return the first existing table/view name from candidates."""
    have = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    for c in candidates:
        if c in have:
            return c
    # also try suffixed base tables (rocpd_<x>_<uuid>)
    for c in candidates:
        for name in have:
            if name.startswith(c + "_"):
                return name
    return None


def main():
    if len(sys.argv) < 2:
        print("usage: inspect_kernels.py <db-or-glob>", file=sys.stderr)
        sys.exit(1)
    arg = sys.argv[1]
    matches = sorted(glob.glob(arg, recursive=True)) if any(
        ch in arg for ch in "*?[") else [arg]
    if not matches:
        print(f"no db matched {arg}", file=sys.stderr)
        sys.exit(1)
    db = matches[-1]
    print(f"DB: {db}\n")

    con = sqlite3.connect(con_path := db)

    disp = _resolve(con, "rocpd_kernel_dispatch")
    ksym = _resolve(con, "kernel_symbols", "rocpd_info_kernel_symbol")
    agent = _resolve(con, "rocpd_info_agent")
    pmc_ev = _resolve(con, "rocpd_pmc_event", "pmc_events")
    pmc_info = _resolve(con, "rocpd_info_pmc")

    print(f"resolved tables: dispatch={disp} ksym={ksym} agent={agent} "
          f"pmc_event={pmc_ev} pmc_info={pmc_info}\n")

    # GPU agent ids
    gpu_ids = set()
    if agent:
        try:
            gpu_ids = {int(r[0]) for r in con.execute(
                f"SELECT id FROM {agent} WHERE upper(type)='GPU'")}
        except sqlite3.Error:
            pass
    print(f"GPU agent ids: {sorted(gpu_ids)}\n")

    # Figure out the kernel-symbol table's columns: the name column varies
    # across rocprofv3 versions (name / kernel_name / display_name / formatted_
    # kernel_name) and the join key (id / kernel_id).
    ksym_cols = []
    if ksym:
        ksym_cols = [r[1] for r in con.execute(f'PRAGMA table_info("{ksym}")')]
        print(f"kernel-symbol cols: {ksym_cols}")
    name_col = next((c for c in ("kernel_name", "formatted_kernel_name",
                                 "display_name", "demangled_name", "name")
                     if c in ksym_cols), None)
    id_col = next((c for c in ("id", "kernel_id") if c in ksym_cols), None)
    print(f"using name_col={name_col} id_col={id_col}\n")

    # Per-kernel duration breakdown
    if disp and ksym and name_col and id_col:
        print("=== per-kernel: total_ms | count | avg_us | name ===")
        try:
            q = (f'SELECT ks."{name_col}", COUNT(*), SUM(kd."end"-kd.start), '
                 f'AVG(kd."end"-kd.start) '
                 f'FROM {disp} kd JOIN {ksym} ks ON kd.kernel_id = ks."{id_col}" '
                 f'GROUP BY ks."{name_col}" ORDER BY 3 DESC LIMIT 25')
            for name, cnt, tot, avg in con.execute(q):
                nm = (name or "")[:75]
                print(f"  {tot/1e6:10.3f} ms  n={cnt:<6} avg={avg/1e3:8.2f}us  {nm}")
        except sqlite3.Error as e:
            print(f"  kernel-join failed: {e}")
    else:
        print("  (could not resolve kernel name/id columns; dumping by agent)")
        for aid, cnt, tot in con.execute(
            f'SELECT agent_id, COUNT(*), SUM("end"-start) FROM {disp} '
            f'GROUP BY agent_id'):
            print(f"  agent {aid}: n={cnt} total={tot/1e6:.3f} ms")

    # Total counter values (which counters are present, summed)
    if pmc_ev and pmc_info:
        print("\n=== counter totals (name | summed value) ===")
        try:
            q = (f"SELECT ip.name, SUM(pe.value) "
                 f"FROM {pmc_ev} pe JOIN {pmc_info} ip ON pe.pmc_id = ip.id "
                 f"WHERE ip.name NOT LIKE 'KFD%' GROUP BY ip.name")
            for name, val in con.execute(q):
                print(f"  {name}: {val}")
        except sqlite3.Error as e:
            print(f"  counter query failed: {e}")

    con.close()


if __name__ == "__main__":
    main()
