"""Run small workloads under each strategy and append outcomes to logs/decisions.csv (the Phase 2 dataset).

classical: exact diagonalisation (VQE Hamiltonians) / brute force (Max-Cut).
hybrid:    SPSA-optimised VQE/QAOA on qiskit-aer with the FakeFez noise model.

Hybrid runs are simulated, so their wall clock includes simulating the QPU. The projected runtime
replaces that simulated part with the estimated QPU time (shots / backend throughput).
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

from hybrid_mapper.backends import get_backend
from hybrid_mapper.profile import FIELD_NAMES
from hybrid_mapper.profiler import profile, transpile_for
from hybrid_mapper.rules import decide, load_rules
from hybrid_mapper.workloads import Workload, maxcut, vqe

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "decisions.csv"
COLUMNS = ["timestamp", "workload", "strategy", *FIELD_NAMES, "decision", "fired_rule",
           "runtime_s", "projected_runtime_s", "qpu_seconds_est", "qpu_calls", "energy_j_est", "shots",
           "result", "exact", "error", "success"]
MAX_QUBITS, MAX_NODES = 12, 20
BACKEND = "heron_r3"  # the noise model below is FakeFez, i.e. heron_r3


def brute_force_maxcut(graph: nx.Graph) -> int:
    n = graph.number_of_nodes()
    assert n <= MAX_NODES, f"brute force capped at {MAX_NODES} nodes"
    edges = np.array(graph.edges)
    bits = (np.arange(2 ** (n - 1))[:, None] >> np.arange(n)) & 1  # node n-1 fixed to 0 (cut symmetry)
    return int((bits[:, edges[:, 0]] != bits[:, edges[:, 1]]).sum(1).max())


def exact_ground_energy(w: Workload) -> float:
    assert w.circuit.num_qubits <= MAX_QUBITS
    return float(np.linalg.eigvalsh(w.hamiltonian.to_matrix())[0])


def spsa(f_pair, x0: np.ndarray, iters: int, rng, c: float = 0.2, target_step: float = 0.6,
         calibration_steps: int = 5) -> np.ndarray:
    """Minimise f with SPSA; f_pair evaluates [x+, x-] in one batch.

    The learning rate is calibrated (as in Qiskit's SPSA) so the first update moves each parameter by
    about target_step, whatever the scale of f. An uncalibrated rate barely moved VQE parameters.
    """
    x = x0.copy()
    grads = []
    for _ in range(calibration_steps):
        delta = rng.choice([-1.0, 1.0], size=x.size)
        fp, fm = f_pair(np.array([x + c * delta, x - c * delta]))
        grads.append(abs(fp - fm) / (2 * c))
    a = target_step / max(float(np.mean(grads)), 1e-12)
    for k in range(iters):
        ak, ck = a / (k + 1) ** 0.602, c / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], size=x.size)
        fp, fm = f_pair(np.array([x + ck * delta, x - ck * delta]))
        x -= ak * (fp - fm) / (2 * ck) * delta
    return x


class QPUMeter:
    """Wraps the simulated-QPU call: counts round trips and parameter sets, times the simulation."""

    def __init__(self, fn):
        self.fn, self.calls, self.evals, self.seconds = fn, 0, 0, 0.0

    def __call__(self, xs):
        t0 = time.perf_counter()
        out = self.fn(xs)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.evals += len(xs)
        return out


def spsa_kwargs(cfg: dict) -> dict:
    return dict(c=cfg["spsa_perturbation"], target_step=cfg["spsa_target_step"],
                calibration_steps=cfg["spsa_calibration_steps"])


def hybrid_vqe(w: Workload, sim, iters: int, cfg: dict, rng) -> tuple[float, int, QPUMeter]:
    isa = transpile_for(w.circuit, BACKEND)
    obs = w.hamiltonian.apply_layout(isa.layout)
    prec = cfg["estimator_precision"]
    est = BackendEstimatorV2(backend=sim, options={"default_precision": prec, "seed_simulator": cfg["seed"]})
    energies = QPUMeter(lambda xs: est.run([(isa, obs, xs)]).result()[0].data.evs)
    x = spsa(energies, rng.uniform(-np.pi, np.pi, isa.num_parameters), iters, rng, **spsa_kwargs(cfg))
    energy = float(energies(x[None, :])[0])
    # ponytail: shot count assumes qubit-wise-commuting grouping, which is what BackendEstimatorV2 does.
    shots_per_eval = len(w.hamiltonian.group_commuting(qubit_wise=True)) * math.ceil(1 / prec**2)
    return energy, shots_per_eval * energies.evals, energies


def hybrid_qaoa(w: Workload, sim, iters: int, cfg: dict, rng) -> tuple[float, int, QPUMeter]:
    isa = transpile_for(w.circuit, BACKEND)
    shots = cfg["sampler_shots"]
    sampler = BackendSamplerV2(backend=sim, options={"default_shots": shots, "seed_simulator": cfg["seed"]})
    edges = np.array(w.graph.edges)

    def cut_samples(xs):
        meas = sampler.run([(isa, xs)]).result()[0].data.meas
        bits = [meas[i].to_bool_array(order="little") for i in range(len(xs))]
        return [(b[:, edges[:, 0]] != b[:, edges[:, 1]]).sum(1) for b in bits]

    sample = QPUMeter(cut_samples)
    neg_mean_cut = lambda xs: np.array([-c.mean() for c in sample(xs)])
    x = spsa(neg_mean_cut, rng.uniform(0, 0.5, isa.num_parameters), iters, rng, **spsa_kwargs(cfg))
    best = int(sample(x[None, :])[0].max())  # QAOA returns the best sampled bitstring
    return best, shots * sample.evals, sample


def workloads(quick: bool, reps: int) -> list[Workload]:
    if quick:
        return [vqe(4, reps), maxcut(6)]
    return [vqe(4, reps), vqe(6, reps), vqe(8, reps), maxcut(6), maxcut(8), maxcut(10), maxcut(12)]


def open_log(log_path: Path):
    """Open the CSV for appending. A log written with different columns is moved aside, never overwritten."""
    if log_path.exists():
        with log_path.open() as f:
            header = next(csv.reader(f), None)
        if header != COLUMNS:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            moved = log_path.with_name(f"{log_path.stem}.{stamp}.old.csv")
            log_path.rename(moved)
            print(f"column layout changed: previous log moved to {moved.parent.name}/{moved.name}")
    new = not log_path.exists()
    f = log_path.open("a", newline="")
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    if new:
        writer.writeheader()
    return f, writer


def run(quick: bool = False, log_path: Path = LOG_PATH) -> list[dict]:
    rules = load_rules()
    cfg, watts = rules["baselines"], rules["energy"]
    iters = cfg["optimizer_max_iter_quick" if quick else "optimizer_max_iter"]
    throughput = get_backend(BACKEND)["throughput_per_s"]
    sim = AerSimulator.from_backend(FakeFez())
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for w in workloads(quick, cfg["vqe_reps"]):
        prof = profile(w, BACKEND)
        dec = decide(prof)
        is_vqe = w.algorithm == "vqe"

        t0 = time.perf_counter()
        exact = exact_ground_energy(w) if is_vqe else brute_force_maxcut(w.graph)
        wall = time.perf_counter() - t0
        outcomes = {"classical": dict(result=exact, runtime_s=wall, projected_runtime_s=wall, qpu_seconds_est=0.0,
                                      qpu_calls=0, shots=0, energy_j_est=wall * watts["classical_node_watts"])}

        t0 = time.perf_counter()
        result, shots, meter = (hybrid_vqe if is_vqe else hybrid_qaoa)(w, sim, iters, cfg, rng)
        wall = time.perf_counter() - t0
        classical_s, qpu_s = wall - meter.seconds, shots / throughput
        outcomes["hybrid"] = dict(result=result, runtime_s=wall, projected_runtime_s=classical_s + qpu_s,
                                  qpu_seconds_est=qpu_s, qpu_calls=meter.calls, shots=shots,
                                  energy_j_est=classical_s * watts["classical_node_watts"]
                                  + qpu_s * watts["qpu_system_watts"])

        for strategy, o in outcomes.items():
            error = abs(o["result"] - exact)
            row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "workload": w.name,
                   "strategy": strategy, **prof.to_dict(), "decision": dec.decision, "fired_rule": dec.fired_rule,
                   **{k: round(v, 4) if isinstance(v, float) else v for k, v in o.items()},
                   "exact": exact, "error": error, "success": error <= cfg["success_rel_error"] * abs(exact)}
            rows.append(row)
            print(f"{w.name:22s} {strategy:9s} rule={dec.decision}/{dec.fired_rule}  result={o['result']:.4g}  "
                  f"exact={exact:.4g}  error={error:.3g}  shots={o['shots']}  wall={o['runtime_s']:.1f}s  "
                  f"projected={o['projected_runtime_s']:.2f}s")
    f, writer = open_log(log_path)
    with f:
        writer.writerows(rows)
    print(f"appended {len(rows)} rows to {log_path.parent.name}/{log_path.name}")
    return rows
