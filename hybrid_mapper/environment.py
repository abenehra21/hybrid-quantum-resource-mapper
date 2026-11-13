"""Phase 3 ground truth: what would actually happen if a task ran classically or as a hybrid workflow.

This is a simulation study's "world model", standing in for real execution feedback until Phase 5.
It deliberately models effects the Phase 1 rules do not:

  * tensor-network (MPS) simulation: weakly entangled circuits are classically easy at any width
    (Vidal 2003), so "more than 30 qubits" does not imply "needs a QPU";
  * strong classical heuristics: Goemans-Williamson's 0.878 Max-Cut guarantee beats low-depth QAOA
    worst cases (Farhi et al. 2014; Wurtz & Love 2021), and small graphs are solved exactly by enumeration;
  * an HPC node, not a 16 GiB workstation, doing the statevector simulation;
  * readout error on sampling tasks, and the shots-for-fidelity trade-off: a sampling task with low
    fidelity can still succeed by taking more shots, if the budget allows;
  * a random queue wait around the nominal value the allocator is told.

Nothing here is read by rules.py. The allocator only sees its outcomes as labels.
"""
import math
from functools import cache
from pathlib import Path

import yaml

from hybrid_mapper.backends import CONFIG_DIR, get_backend
from hybrid_mapper.rules import KIND, load_rules

FAMILY_ALGORITHM = {"maxcut": "qaoa", "vqe": "vqe", "random": "samp"}
ACTIONS = ["CLASSICAL", "HYBRID", "REDESIGN"]


@cache
def load_environment(path: Path = CONFIG_DIR / "environment.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def classical_option(t: dict, env: dict) -> dict:
    """Cheapest classical method that meets the accuracy target before the deadline."""
    c = env["classical"]
    n, deadline_s = int(t["n_qubits"]), t["deadline_min"] * 60
    gates = 3 * t["two_qubit_gates"]  # ponytail: ~2 single-qubit gates per two-qubit gate
    mem = c["node_memory_gib"] * 2**30
    methods = []  # (name, seconds, meets accuracy target)
    if t["family"] == "maxcut":
        methods.append(("heuristic", 1e-4 * t["n_edges"], c["maxcut_heuristic_ratio"] >= t["maxcut_ratio_target"]))
        methods.append(("exact", 2.0 ** (n - 1) * t["n_edges"] / c["enumeration_rate"], True))
    else:
        if 16 * 2.0**n <= mem:
            methods.append(("statevector", gates * 2.0**n * 32 / (c["memory_bandwidth_gbs"] * 1e9), True))
        chi = 2.0 ** min(int(t["entanglement_bound"]), n // 2)
        if chi <= c["mps_max_bond"] and n * chi**2 * 16 <= mem:
            methods.append(("mps", gates * 8 * chi**3 / c["flops"], True))
    variants = t.get("variants", 1)  # Phase 4: a cut circuit's fragment is simulated once per variant
    methods = [(name, s * variants + c["job_overhead_s"], meets) for name, s, meets in methods]
    ok = [(name, s) for name, s, meets in methods if meets and s <= deadline_s]
    if ok:
        name, s = min(ok, key=lambda m: m[1])
        return {"feasible": True, "method": name, "time_s": s, "reason": f"{name} in {s:.3g} s"}
    if methods:
        name, s, meets = min(methods, key=lambda m: m[1])
        why = "misses accuracy target" if not meets else "misses deadline"
        return {"feasible": False, "method": name, "time_s": min(s, deadline_s), "reason": f"{name} {why}"}
    return {"feasible": False, "method": "none", "time_s": 0.0,
            "reason": "too large and too entangled for statevector or MPS"}


def hybrid_option(t: dict, env: dict) -> dict:
    """Run the quantum part on the target QPU, with effects the rules ignore."""
    q = env["quantum"]
    spec = get_backend(t["backend"])
    n, deadline_s = int(t["n_qubits"]), t["deadline_min"] * 60
    queue_s = t["queue_real_min"] * 60
    fidelity = 0.0

    def fail(reason: str, shots_attempted: float = 0.0) -> dict:  # cost of an attempt that cannot succeed
        qpu_s = shots_attempted / spec["throughput_per_s"]
        return {"feasible": False, "fidelity": fidelity, "shots": shots_attempted, "qpu_s": qpu_s,
                "time_s": queue_s + qpu_s if shots_attempted else 0.0, "reason": reason}

    if n > spec["qubits"]:
        return fail(f"{n} qubits > {spec['qubits']}")
    kind = KIND[FAMILY_ALGORITHM[t["family"]]]
    fidelity = math.exp(t["two_qubit_gates"] * math.log1p(-spec["eplg"]))
    if kind != "expect":
        fidelity *= (1 - q["readout_error"][t["backend"]]) ** n
    if kind == "expect":
        if fidelity < q["mitigation_floor_fidelity"]:
            return fail(f"F={fidelity:.2g} below mitigation floor")
        shots = (t["sum_abs_coeffs"] / t["precision_target"]) ** 2 / fidelity**2
    elif t["family"] == "maxcut":
        ratio = fidelity * q["qaoa_ratio"][int(t["reps"])] + (1 - fidelity) * 0.5
        if ratio < t["maxcut_ratio_target"]:
            return fail(f"QAOA ratio {ratio:.3f} < target {t['maxcut_ratio_target']}", t["samples_needed"])
        shots = t["samples_needed"]
    else:
        shots = max(t["samples_needed"], q["good_samples_required"] / fidelity) if fidelity > 0 else math.inf
    if shots > t["shot_budget"]:
        return fail(f"shots {shots:.3g} > budget {t['shot_budget']:.3g}", t["shot_budget"])
    qpu_s = shots / spec["throughput_per_s"]
    if queue_s + qpu_s > deadline_s:
        return fail(f"queue+QPU {(queue_s + qpu_s) / 60:.0f} min > deadline", min(shots, t["shot_budget"]))
    return {"feasible": True, "fidelity": fidelity, "shots": shots, "qpu_s": qpu_s, "time_s": queue_s + qpu_s,
            "reason": f"F={fidelity:.2g}, {shots:.3g} shots, {(queue_s + qpu_s) / 60:.0f} min"}


def cost(time_s: float, energy_j: float, qpu_s: float) -> float:
    w = load_rules()["oracle_weights"]
    return w["w_time"] * time_s + w["w_energy"] * energy_j + w["w_qpu"] * qpu_s


def oracle(t: dict, env: dict | None = None) -> dict:
    """Both options, their costs, and the label: the cheapest feasible option, else REDESIGN."""
    env = env or load_environment()
    cl, hy = classical_option(t, env), hybrid_option(t, env)
    cl["energy_j"] = cl["time_s"] * env["classical"]["node_watts"]
    cl["qpu_s"], cl["shots"] = 0.0, 0.0
    hy["energy_j"] = hy["qpu_s"] * env["quantum"]["qpu_system_watts"]
    for o in (cl, hy):
        o["J"] = cost(o["time_s"], o["energy_j"], o["qpu_s"])
    feasible = [(a, o) for a, o in (("CLASSICAL", cl), ("HYBRID", hy)) if o["feasible"]]
    label = min(feasible, key=lambda x: x[1]["J"])[0] if feasible else "REDESIGN"
    return {"label": label, "classical": cl, "hybrid": hy}


def flatten(result: dict) -> dict:
    """Oracle result as flat CSV columns."""
    row = {"oracle_label": result["label"]}
    for side in ("classical", "hybrid"):
        for k, v in result[side].items():
            row[f"{side}_{k}"] = v
    return row


def evaluate_action(action: str, t: dict, penalty_s: float) -> dict:
    """What choosing `action` costs on task t (a row with flattened oracle columns).

    Regret is the extra cost over the oracle's choice. Any decision that yields no accepted result
    (a failed run, or declining a feasible task) is charged penalty_s, plus whatever it wasted.
    """
    label = t["oracle_label"]
    j_oracle = t[f"{label.lower()}_J"] if label != "REDESIGN" else 0.0
    out = {"success": False, "executed": action != "REDESIGN", "time_s": 0.0, "energy_j": 0.0, "qpu_s": 0.0,
           "shots": 0.0, "unnecessary_qpu": action == "HYBRID" and bool(t["classical_feasible"])}
    if action == "REDESIGN":
        out["success"] = label == "REDESIGN"
        out["regret"] = 0.0 if out["success"] else penalty_s
        return out
    side = action.lower()
    for k in ("time_s", "energy_j", "qpu_s", "shots"):
        out[k] = t[f"{side}_{k}"]
    out["success"] = bool(t[f"{side}_feasible"])
    # success: extra cost over the oracle; failure: the wasted attempt plus the missed-result penalty
    out["regret"] = (t[f"{side}_J"] - j_oracle) if out["success"] else t[f"{side}_J"] + penalty_s
    return out
