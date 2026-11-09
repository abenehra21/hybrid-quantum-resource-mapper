import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2

from hybrid_mapper.environment import entanglement_bound, evaluate_action, flatten, oracle


@pytest.mark.parametrize("entanglement,reps,expected", [("reverse_linear", 2, 2), ("circular", 2, 4)])
def test_entanglement_bound_of_ansatz(entanglement, reps, expected):
    # one CX per layer crosses the middle cut of a line, two for a ring
    assert entanglement_bound(efficient_su2(10, reps=reps, entanglement=entanglement)) == expected


def test_entanglement_bound_of_ghz_chain_is_one_ebit():
    ghz = QuantumCircuit(8)
    ghz.h(0)
    for q in range(7):
        ghz.cx(q, q + 1)
    assert entanglement_bound(ghz) == 1


def _task(**kw):
    base = dict(family="random", backend="heron_r3", n_qubits=50, reps=20, n_edges=0, sum_abs_coeffs=0.0,
                two_qubit_gates=3747, entanglement_bound=25, precision_target=1e-2, maxcut_ratio_target=0.85,
                deadline_min=480, queue_wait_min=30, queue_real_min=30, shot_budget=1e8, samples_needed=1e4)
    return base | kw


def test_low_fidelity_sampling_is_rescued_by_shots_but_not_without_budget():
    assert oracle(_task())["label"] == "HYBRID"            # F ~ 2e-4, 1e8 shots buy ~2e4 clean samples
    assert oracle(_task(shot_budget=1e6))["label"] == "REDESIGN"


def test_weakly_entangled_circuit_is_classical_at_any_width():
    t = _task(family="vqe", n_qubits=100, reps=2, two_qubit_gates=198, entanglement_bound=2, sum_abs_coeffs=290.0)
    o = oracle(t)
    assert o["label"] == "CLASSICAL" and o["classical"]["method"] == "mps"


def test_regret_is_zero_for_the_oracle_choice_and_penalises_failures():
    t = _task()
    row = t | flatten(oracle(t))
    assert evaluate_action("HYBRID", row, 3600)["regret"] == 0
    assert evaluate_action("REDESIGN", row, 3600)["regret"] == 3600
