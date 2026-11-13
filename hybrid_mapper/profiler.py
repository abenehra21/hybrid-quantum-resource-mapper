"""F1: transpile a workload for a target QPU and fill the circuit-level TaskProfile fields.

Also bounds the circuit's entanglement (no transpile needed), which sets low_entanglement for rule R3."""
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile

from hybrid_mapper.backends import get_backend, transpile_kwargs
from hybrid_mapper.profile import TaskProfile
from hybrid_mapper.rules import KIND, load_rules
from hybrid_mapper.workloads import Workload


def transpile_for(circuit: QuantumCircuit, backend: str, seed: int = 0) -> QuantumCircuit:
    return transpile(circuit, optimization_level=3, seed_transpiler=seed, **transpile_kwargs(backend))


def two_qubit_count(circuit: QuantumCircuit) -> int:
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)


def logical(circuit: QuantumCircuit) -> QuantumCircuit:
    """The circuit in CZ + single-qubit gates, before any routing (no coupling map, no measurements)."""
    return transpile(circuit.remove_final_measurements(inplace=False), basis_gates=["cz", "rz", "sx", "x"],
                     optimization_level=0)


def entanglement_bound(circuit, order: list[int] | None = None) -> int:
    """Upper bound on the entanglement (in ebits) across any cut of the qubit line.

    Each CZ gate that crosses a cut can at most double the Schmidt rank there, and a cut with k qubits
    on its smaller side holds at most k ebits. An exact matrix-product-state simulation therefore needs
    bond dimension at most 2**bound. order[q] is qubit q's position on the MPS chain.
    """
    n = circuit.num_qubits
    if n < 2:
        return 0
    dec = logical(circuit)
    pos = order or list(range(n))
    diff = np.zeros(n + 1, dtype=int)
    for inst in dec.data:
        if inst.operation.num_qubits == 2:
            a, b = sorted(pos[dec.find_bit(q).index] for q in inst.qubits)
            diff[a + 1] += 1  # the gate crosses every cut k with a < k <= b
            diff[b + 1] -= 1
    crossings = np.cumsum(diff)[1:n]
    k = np.arange(1, n)
    return int(np.max(np.minimum(crossings, np.minimum(k, n - k))))


def workload_entanglement_bound(workload) -> int:
    if workload.graph is not None:  # an MPS simulator would reorder graph nodes to shrink cuts
        ordering = list(nx.utils.reverse_cuthill_mckee_ordering(workload.graph))
        position = {node: i for i, node in enumerate(ordering)}
        return entanglement_bound(workload.circuit, [position[q] for q in range(workload.circuit.num_qubits)])
    return entanglement_bound(workload.circuit)


def profile(workload: Workload, backend: str = "heron_r3", **overrides) -> TaskProfile:
    isa = transpile_for(workload.circuit, backend)
    spec = get_backend(backend)
    rules = load_rules()
    defaults = rules["workload_defaults"]
    ham = workload.hamiltonian
    fields = dict(
        task_id=workload.name,
        problem_class=workload.problem_class,
        algorithm=workload.algorithm,
        precision_target=float(defaults["precision_target"]),
        deadline_min=float(defaults["deadline_min"]),
        classical_meets_target=False,  # needs domain knowledge of classical methods; caller can override
        low_entanglement=workload_entanglement_bound(workload) <= rules["low_entanglement_max_ebits"],
        n_qubits=workload.circuit.num_qubits,
        depth=isa.depth(),
        two_qubit_gates=two_qubit_count(isa),
        sum_abs_coeffs=float(sum(abs(ham.coeffs))) if ham is not None else 0.0,
        measurement_groups=len(ham.group_commuting(qubit_wise=True)) if ham is not None else 0,
        samples_needed=int(defaults["samples_needed"]) if KIND[workload.algorithm] == "sample" else 0,
        target_qpu=backend,
        eplg=float(spec["eplg"]),
        throughput_per_s=float(spec["throughput_per_s"]),
        queue_wait_min=float(defaults["queue_wait_min"]),
        shot_budget=float(defaults["shot_budget"]),
    )
    return TaskProfile(**(fields | overrides))
