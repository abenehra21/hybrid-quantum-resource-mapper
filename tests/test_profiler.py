from hybrid_mapper.profiler import profile
from hybrid_mapper.workloads import maxcut


def test_profile_small_maxcut():
    p = profile(maxcut(6, p=1), "nighthawk_r2")
    assert (p.n_qubits, p.algorithm, p.target_qpu) == (6, "qaoa", "nighthawk_r2")
    assert p.two_qubit_gates >= 9  # >= one ZZ interaction per edge of a 3-regular 6-node graph
    assert p.sum_abs_coeffs == 9.0


def test_r3_fires_for_weakly_entangled_wide_circuit():
    from hybrid_mapper.rules import decide
    from hybrid_mapper.workloads import random_workload, vqe

    linear = profile(vqe(40, reps=2), "heron_r3")  # nearest-neighbour ansatz: 2 ebits across any cut
    assert linear.low_entanglement and decide(linear).fired_rule == "R3"
    assert not profile(random_workload(40, 10), "heron_r3").low_entanglement
