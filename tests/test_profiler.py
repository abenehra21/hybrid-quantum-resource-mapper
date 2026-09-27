from hybrid_mapper.profiler import profile
from hybrid_mapper.workloads import maxcut


def test_profile_small_maxcut():
    p = profile(maxcut(6, p=1), "nighthawk_r2")
    assert (p.n_qubits, p.algorithm, p.target_qpu) == (6, "qaoa", "nighthawk_r2")
    assert p.two_qubit_gates >= 9  # >= one ZZ interaction per edge of a 3-regular 6-node graph
    assert p.sum_abs_coeffs == 9.0
