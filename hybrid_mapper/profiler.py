"""F1: transpile a workload for a target QPU and fill the circuit-level TaskProfile fields."""
from qiskit import QuantumCircuit, transpile

from hybrid_mapper.backends import get_backend, transpile_kwargs
from hybrid_mapper.profile import TaskProfile
from hybrid_mapper.rules import KIND, load_rules
from hybrid_mapper.workloads import Workload


def transpile_for(circuit: QuantumCircuit, backend: str, seed: int = 0) -> QuantumCircuit:
    return transpile(circuit, optimization_level=3, seed_transpiler=seed, **transpile_kwargs(backend))


def two_qubit_count(circuit: QuantumCircuit) -> int:
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)


def profile(workload: Workload, backend: str = "heron_r3", **overrides) -> TaskProfile:
    isa = transpile_for(workload.circuit, backend)
    spec = get_backend(backend)
    defaults = load_rules()["workload_defaults"]
    ham = workload.hamiltonian
    fields = dict(
        task_id=workload.name,
        problem_class=workload.problem_class,
        algorithm=workload.algorithm,
        precision_target=float(defaults["precision_target"]),
        deadline_min=float(defaults["deadline_min"]),
        # Not inferable from the circuit; left False unless the caller knows better.
        classical_meets_target=False,
        low_entanglement=False,
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
