"""F2 + F4: derived quantities and the ordered rule list. First rule that fires decides."""
import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

from hybrid_mapper.backends import CONFIG_DIR, get_backend
from hybrid_mapper.profile import TaskProfile
from hybrid_mapper.templates import SPLITS

KIND = {"vqe": "expect", "dyn": "expect", "qaoa": "sample", "samp": "sample", "sqd": "recover"}


@dataclass
class Decision:
    decision: str
    fired_rule: str
    trace: list[dict]
    derived: dict
    split: dict | None = field(default=None)


@cache
def load_rules(path: Path = CONFIG_DIR / "rules.yaml") -> dict:
    """Cached per path: treat the result as read-only (build a new dict to try other thresholds)."""
    return yaml.safe_load(Path(path).read_text())


def derive(p: TaskProfile) -> dict:
    kind = KIND.get(p.algorithm)
    fidelity = math.exp(p.two_qubit_gates * math.log1p(-p.eplg)) if p.eplg < 1 else 0.0
    if kind == "expect":
        shots = (p.sum_abs_coeffs / p.precision_target) ** 2 / fidelity**2 if fidelity > 0 else math.inf
    elif kind in ("sample", "recover"):
        shots = float(p.samples_needed)
    else:
        shots = 0.0
    qpu_seconds = shots / p.throughput_per_s if p.throughput_per_s > 0 else math.inf
    return {
        "kind": kind,
        "fidelity": fidelity,
        "shots": shots,
        "qpu_seconds": qpu_seconds,
        "statevector_bytes": 16 * 2**p.n_qubits,
    }


def decide(p: TaskProfile, rules_path: Path = CONFIG_DIR / "rules.yaml", cfg: dict | None = None) -> Decision:
    """cfg overrides rules_path with an already-loaded config (Phase 3 threshold tuning)."""
    cfg = cfg or load_rules(rules_path)
    d = derive(p)
    mem_bytes = cfg["statevector_memory_gib"] * 2**30
    gib = d["statevector_bytes"] / 2**30

    def backend_qubits():
        return get_backend(p.target_qpu)["qubits"]

    def f_min():
        return float(cfg["fidelity_min"][d["kind"]])

    def runtime_min():
        return p.queue_wait_min + d["qpu_seconds"] / 60

    # (id, outcome, predicate, reason-if-fired, reason-if-passed). Lambdas keep evaluation lazy and in order.
    rules = [
        ("R1", "CLASSICAL", lambda: p.problem_class == "classical" or p.algorithm == "none",
         lambda: f"problem_class={p.problem_class}, algorithm={p.algorithm}: no quantum algorithm applies",
         lambda: "task has a quantum algorithm candidate"),
        ("R2", "CLASSICAL", lambda: d["statevector_bytes"] <= mem_bytes,
         lambda: f"{p.n_qubits} qubits: statevector {gib:.3g} GiB fits in {cfg['statevector_memory_gib']} GiB, simulate exactly",
         lambda: f"{p.n_qubits} qubits: statevector {gib:.3g} GiB exceeds {cfg['statevector_memory_gib']} GiB"),
        ("R3", "CLASSICAL", lambda: p.low_entanglement,
         lambda: "low entanglement: tensor-network/MPS methods apply",
         lambda: "entanglement not flagged as low"),
        ("R4", "CLASSICAL", lambda: p.classical_meets_target,
         lambda: "a classical method already meets the precision target",
         lambda: "no classical method known to meet the target"),
        ("R5", "REDESIGN", lambda: p.n_qubits > backend_qubits(),
         lambda: f"{p.n_qubits} qubits > {backend_qubits()} on {p.target_qpu}",
         lambda: f"{p.n_qubits} qubits fit on {p.target_qpu} ({backend_qubits()})"),
        ("R6", "REDESIGN", lambda: d["fidelity"] < f_min(),
         lambda: f"F={d['fidelity']:.3g} < F_min[{d['kind']}]={f_min():.3g}",
         lambda: f"F={d['fidelity']:.3g} >= F_min[{d['kind']}]={f_min():.3g}"),
        ("R7", "REDESIGN", lambda: d["shots"] > p.shot_budget,
         lambda: f"shots {d['shots']:.3g} > budget {p.shot_budget:.3g}",
         lambda: f"shots {d['shots']:.3g} <= budget {p.shot_budget:.3g}"),
        ("R8", "CLASSICAL", lambda: runtime_min() > p.deadline_min,
         lambda: f"queue+QPU {runtime_min():.1f} min > deadline {p.deadline_min:g} min (classical fallback)",
         lambda: f"queue+QPU {runtime_min():.1f} min <= deadline {p.deadline_min:g} min"),
        ("R9", "HYBRID", lambda: True,
         lambda: f"all checks passed: run {p.algorithm} as hybrid workflow on {p.target_qpu}",
         lambda: ""),
    ]

    trace = []
    for rule_id, outcome, pred, why_fired, why_passed in rules:
        fired = bool(pred())
        trace.append({"rule": rule_id, "fired": fired, "reason": why_fired() if fired else why_passed()})
        if fired:
            split = SPLITS[p.algorithm] if outcome == "HYBRID" else None
            return Decision(outcome, rule_id, trace, d, split)
    raise AssertionError("R9 always fires")
