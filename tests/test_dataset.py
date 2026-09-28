import pytest

from hybrid_mapper.dataset import build, logical_two_qubit_count, problem_features


@pytest.mark.parametrize("spec", [("maxcut", 10, 2, "none"), ("vqe", 6, 2, "reverse_linear"),
                                  ("vqe", 6, 2, "circular"), ("vqe", 6, 2, "full")])
def test_analytic_two_qubit_count_matches_unrouted_circuit(spec):
    assert problem_features(*spec)["analytic_2q"] == logical_two_qubit_count(build(*spec).circuit)
