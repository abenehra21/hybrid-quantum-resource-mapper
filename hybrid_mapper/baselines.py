"""Run small workloads under each strategy and append outcomes to logs/decisions.csv (the Phase 2 dataset).

classical: exact diagonalisation (VQE Hamiltonians) / brute force (Max-Cut).
hybrid:    SPSA-optimised VQE/QAOA on qiskit-aer with the FakeFez noise model.
"""
import csv
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez

from hybrid_mapper.profile import FIELD_NAMES
from hybrid_mapper.profiler import profile, transpile_for
from hybrid_mapper.rules import decide, load_rules
from hybrid_mapper.workloads import Workload, maxcut, vqe

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "decisions.csv"
COLUMNS = ["timestamp", "workload", "strategy", *FIELD_NAMES, "decision", "fired_rule",
           "runtime_s", "result", "exact", "error", "shots"]
MAX_QUBITS, MAX_NODES = 12, 20


def brute_force_maxcut(graph: nx.Graph) -> int:
    n = graph.number_of_nodes()
    assert n <= MAX_NODES, f"brute force capped at {MAX_NODES} nodes"
    edges = np.array(graph.edges)
    bits = (np.arange(2 ** (n - 1))[:, None] >> np.arange(n)) & 1  # node n-1 fixed to 0 (cut symmetry)
    return int((bits[:, edges[:, 0]] != bits[:, edges[:, 1]]).sum(1).max())


def exact_ground_energy(w: Workload) -> float:
    assert w.circuit.num_qubits <= MAX_QUBITS
    return float(np.linalg.eigvalsh(w.hamiltonian.to_matrix())[0])


def spsa(f_pair, x0: np.ndarray, iters: int, rng, a: float = 0.1, c: float = 0.1) -> np.ndarray:
    """Minimise f with SPSA; f_pair evaluates [x+, x-] in one batch. Standard gain exponents."""
    x = x0.copy()
    for k in range(iters):
        ak, ck = a / (k + 1) ** 0.602, c / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], size=x.size)
        fp, fm = f_pair(np.array([x + ck * delta, x - ck * delta]))
        x -= ak * (fp - fm) / (2 * ck) * delta
    return x


def hybrid_vqe(w: Workload, sim, iters: int, cfg: dict, rng) -> tuple[float, int]:
    isa = transpile_for(w.circuit, "heron_r3")
    obs = w.hamiltonian.apply_layout(isa.layout)
    prec = cfg["estimator_precision"]
    est = BackendEstimatorV2(backend=sim, options={"default_precision": prec, "seed_simulator": cfg["seed"]})
    energies = lambda xs: est.run([(isa, obs, xs)]).result()[0].data.evs
    x = spsa(energies, rng.uniform(-0.1, 0.1, isa.num_parameters), iters, rng)
    energy = float(energies(x[None, :])[0])
    # ponytail: shot count assumes qubit-wise-commuting grouping, which is what BackendEstimatorV2 does.
    shots_per_eval = len(w.hamiltonian.group_commuting(qubit_wise=True)) * math.ceil(1 / prec**2)
    return energy, shots_per_eval * (2 * iters + 1)


def hybrid_qaoa(w: Workload, sim, iters: int, cfg: dict, rng) -> tuple[float, int]:
    isa = transpile_for(w.circuit, "heron_r3")
    shots = cfg["sampler_shots"]
    sampler = BackendSamplerV2(backend=sim, options={"default_shots": shots, "seed_simulator": cfg["seed"]})
    edges = np.array(w.graph.edges)

    def cut_samples(xs):
        meas = sampler.run([(isa, xs)]).result()[0].data.meas
        bits = [meas[i].to_bool_array(order="little") for i in range(len(xs))]
        return [(b[:, edges[:, 0]] != b[:, edges[:, 1]]).sum(1) for b in bits]

    neg_mean_cut = lambda xs: np.array([-c.mean() for c in cut_samples(xs)])
    x = spsa(neg_mean_cut, rng.uniform(0, 0.5, isa.num_parameters), iters, rng)
    best = int(cut_samples(x[None, :])[0].max())  # QAOA returns the best sampled bitstring
    return best, shots * (2 * iters + 1)


def run(quick: bool = False, log_path: Path = LOG_PATH) -> list[dict]:
    cfg = load_rules()["baselines"]
    iters = cfg["optimizer_max_iter_quick" if quick else "optimizer_max_iter"]
    workloads = [vqe(4), maxcut(6)] if quick else [vqe(4), vqe(8), vqe(12), maxcut(6), maxcut(10), maxcut(12)]
    sim = AerSimulator.from_backend(FakeFez())
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for w in workloads:
        prof = profile(w, "heron_r3")
        dec = decide(prof)
        is_vqe = w.algorithm == "vqe"
        t0 = time.perf_counter()
        exact = exact_ground_energy(w) if is_vqe else brute_force_maxcut(w.graph)
        outcomes = {"classical": (exact, 0, time.perf_counter() - t0)}
        t0 = time.perf_counter()
        result, shots = (hybrid_vqe if is_vqe else hybrid_qaoa)(w, sim, iters, cfg, rng)
        outcomes["hybrid"] = (result, shots, time.perf_counter() - t0)
        for strategy, (res, shots, runtime) in outcomes.items():
            row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "workload": w.name,
                   "strategy": strategy, **prof.to_dict(), "decision": dec.decision, "fired_rule": dec.fired_rule,
                   "runtime_s": round(runtime, 3), "result": res, "exact": exact, "error": abs(res - exact),
                   "shots": shots}
            rows.append(row)
            print(f"{w.name:22s} {strategy:9s} decision={dec.decision}/{dec.fired_rule}  result={res:.4g}  "
                  f"exact={exact:.4g}  error={abs(res - exact):.3g}  shots={shots}  {runtime:.1f}s")
    new = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
    print(f"appended {len(rows)} rows to {log_path.parent.name}/{log_path.name}")
    return rows
