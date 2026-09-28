"""Phase 2 dataset: sweep workload families x sizes x backends, labelling each row with the exact
resources the transpiler produces. Labels are free (no QPU), so thousands of rows cost minutes.

Features are problem-level: known before any circuit is transpiled. Labels are circuit-level.

Each (workload, backend) transpile runs in a forked worker with a hard timeout. Some layouts make
Qiskit's VF2 layout search very slow (a 70-qubit ring has no embedding in heavy-hex: ~28 s vs ~1 s for
an 80-qubit ring that embeds). Timed-out rows are kept with empty labels and timed_out=True.
"""
import multiprocessing as mp
import os
import time
from itertools import product
from pathlib import Path

import pandas as pd
from qiskit import transpile

from hybrid_mapper.profiler import transpile_for, two_qubit_count
from hybrid_mapper.workloads import maxcut, random_workload, vqe

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "resources.csv"
ENTANGLEMENT_PAIRS = {"reverse_linear": lambda n: n - 1, "circular": lambda n: n if n > 2 else n - 1,
                      "full": lambda n: n * (n - 1) // 2}
LABELS = ["two_qubit_gates", "depth", "measurement_groups"]
TRANSPILE_TIMEOUT_S = 60


def build(family: str, size: int, reps: int, entanglement: str = "none", seed: int = 0):
    if family == "maxcut":
        return maxcut(size, reps, seed=seed)
    if family == "vqe":
        return vqe(size, reps, seed=seed, entanglement=entanglement)
    return random_workload(size, reps, seed=seed)


def logical_two_qubit_count(circuit) -> int:
    """Two-qubit gates before routing (decomposition only, no coupling map)."""
    return two_qubit_count(transpile(circuit, basis_gates=["cz", "rz", "sx", "x"], optimization_level=0))


def problem_features(family: str, size: int, reps: int, entanglement: str = "none", seed: int = 0,
                     workload=None) -> dict:
    """Problem-level features. Needs the graph / Hamiltonian, never a transpiled circuit."""
    w = workload or build(family, size, reps, entanglement, seed)
    ham = w.hamiltonian
    if family == "maxcut":
        analytic = 2 * reps * w.graph.number_of_edges()  # one ZZ = 2 two-qubit gates, per edge per layer
    elif family == "vqe":
        analytic = reps * ENTANGLEMENT_PAIRS[entanglement](size)
    else:
        analytic = logical_two_qubit_count(w.circuit)  # random circuits have no closed form
    return {
        "family": family,
        "entanglement": entanglement,
        "n_qubits": w.circuit.num_qubits,
        "reps": reps,  # QAOA p / ansatz reps / random-circuit depth
        "n_edges": w.graph.number_of_edges() if w.graph is not None else 0,
        "n_pauli_terms": len(ham) if ham is not None else 0,
        "sum_abs_coeffs": float(sum(abs(ham.coeffs))) if ham is not None else 0.0,
        "analytic_2q": analytic,
        "seed": seed,
    }


def label(workload, backend: str) -> dict:
    t0 = time.perf_counter()
    isa = transpile_for(workload.circuit, backend)
    ham = workload.hamiltonian
    return {
        "backend": backend,
        "two_qubit_gates": two_qubit_count(isa),
        "depth": isa.depth(),
        "measurement_groups": len(ham.group_commuting(qubit_wise=True)) if ham is not None else None,
        "transpile_s": time.perf_counter() - t0,
    }


def sweep_specs(quick: bool = False) -> list[tuple]:
    """(family, size, reps, entanglement, seed) tuples. Sizes run past 30 qubits so R2 does not decide everything."""
    if quick:
        sizes, seeds = [6, 12, 20, 40, 60], [0]
    else:
        sizes, seeds = [6, 10, 16, 20, 26, 30, 36, 40, 50, 60, 70, 80, 90, 100, 110, 120], [0, 1]
    specs = [("maxcut", n, p, "none", s) for n, p, s in product(sizes, [1, 2, 3], seeds)]
    # The VQE circuit does not depend on the seed (only its Hamiltonian does), so one seed avoids duplicate rows.
    specs += [("vqe", n, r, ent, 0) for n, r, ent in product(sizes, [1, 2, 4], ["reverse_linear", "circular"])]
    specs += [("vqe", n, r, "full", 0) for n, r in product([n for n in sizes if n <= 40], [1, 2])]
    specs += [("random", n, d, "none", s) for n, d, s in product(sizes, [5, 10, 20], seeds)]
    return specs


def _worker(spec, backend, conn) -> None:
    w = build(*spec)
    conn.send(problem_features(*spec, workload=w) | {"backend": backend})
    conn.send(label(w, backend))
    conn.close()


def _run(tasks: list, workers: int, timeout: float) -> list[dict]:
    """Run (spec, backend) tasks in forked processes; kill any that exceed timeout."""
    ctx = mp.get_context("fork")  # fork: children inherit imports, so no per-task import cost
    queue, running, rows, done = list(tasks), {}, [], 0
    while queue or running:
        while queue and len(running) < workers:
            recv, send = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_worker, args=(*queue.pop(0), send))
            proc.start()
            running[proc] = {"recv": recv, "t0": time.monotonic(), "row": None}
        for proc, st in list(running.items()):
            try:
                while st["recv"].poll():
                    msg = st["recv"].recv()
                    st["row"] = msg if st["row"] is None else st["row"] | msg | {"timed_out": False}
            except EOFError:  # child closed its end (finished or crashed)
                pass
            finished = st["row"] is not None and "timed_out" in st["row"]
            expired = time.monotonic() - st["t0"] > timeout
            if finished or expired or not proc.is_alive():
                if not finished:
                    proc.kill()
                    if st["row"] is not None:  # features arrived, transpile did not finish
                        rows.append(st["row"] | {"transpile_s": timeout, "timed_out": True})
                else:
                    rows.append(st["row"])
                proc.join()
                del running[proc]
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"{done}/{len(tasks)} transpiles done", flush=True)
        time.sleep(0.02)
    return rows


def generate(quick: bool = False, backends=("heron_r3", "nighthawk_r2"), path: Path = DATA_PATH,
             workers: int | None = None, timeout: float = TRANSPILE_TIMEOUT_S) -> pd.DataFrame:
    tasks = [(spec, b) for spec in sweep_specs(quick) for b in backends]
    rows = _run(tasks, workers or max(1, (os.cpu_count() or 2) // 2), timeout)
    df = pd.DataFrame(rows).sort_values(["family", "entanglement", "n_qubits", "reps", "seed", "backend"])
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} rows ({int(df.timed_out.sum())} transpile timeouts) to {path.parent.name}/{path.name}")
    return df
