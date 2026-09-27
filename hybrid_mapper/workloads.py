"""Benchmark generators. Hamiltonians here are SYNTHETIC (seeded random), never real molecular data."""
from dataclasses import dataclass

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2, qaoa_ansatz
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp


@dataclass
class Workload:
    name: str
    problem_class: str
    algorithm: str
    circuit: QuantumCircuit
    hamiltonian: SparsePauliOp | None = None
    graph: nx.Graph | None = None


def synthetic_hamiltonian(n_qubits: int, n_terms: int | None = None, seed: int = 0) -> SparsePauliOp:
    """Seeded random Pauli sum standing in for a molecular Hamiltonian. Not physical."""
    rng = np.random.default_rng(seed)
    n_terms = n_terms or 4 * n_qubits
    paulis = ["".join(rng.choice(list("IXYZ"), n_qubits)) for _ in range(n_terms)]
    coeffs = rng.normal(0, 1, n_terms)
    return SparsePauliOp(paulis, coeffs).simplify()


def maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """Sum of Z_i Z_j over edges; cut = (|E| - <H>) / 2, so minimising H maximises the cut."""
    return SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0) for i, j in graph.edges], graph.number_of_nodes())


def maxcut(n_nodes: int, p: int = 1, seed: int = 0) -> Workload:
    graph = nx.random_regular_graph(3, n_nodes, seed=seed)
    ham = maxcut_hamiltonian(graph)
    circ = qaoa_ansatz(ham, reps=p)
    circ.measure_all()
    return Workload(f"maxcut{n_nodes}_p{p}", "opt", "qaoa", circ, hamiltonian=ham, graph=graph)


def vqe(n_qubits: int, reps: int = 1, seed: int = 0) -> Workload:
    ham = synthetic_hamiltonian(n_qubits, seed=seed)
    return Workload(f"synthetic_vqe{n_qubits}_r{reps}", "chem", "vqe", efficient_su2(n_qubits, reps=reps), hamiltonian=ham)


def random_workload(n_qubits: int, depth: int = 10, seed: int = 0) -> Workload:
    circ = random_circuit(n_qubits, depth, max_operands=2, measure=True, seed=seed)
    return Workload(f"random{n_qubits}_d{depth}", "samp", "samp", circ)


GENERATORS = {"maxcut": maxcut, "vqe": vqe, "random": random_workload}
