"""Named reference profiles. fes_* are spec-only (from the published IBM-RIKEN SQD experiment):
no circuit is built or transpiled for them."""
from hybrid_mapper.backends import get_backend
from hybrid_mapper.profile import TaskProfile


def _preset(**kw) -> TaskProfile:
    spec = get_backend(kw.get("target_qpu", "heron_r3"))
    kw.setdefault("eplg", spec["eplg"])
    kw.setdefault("throughput_per_s", spec["throughput_per_s"])
    return TaskProfile(**kw)


_maxcut120 = dict(problem_class="opt", algorithm="qaoa", target_qpu="nighthawk_r2", n_qubits=120,
                  two_qubit_gates=600, samples_needed=100_000, shot_budget=1e7, queue_wait_min=60, deadline_min=480)
_fes = dict(problem_class="chem", target_qpu="heron_r3", n_qubits=77, two_qubit_gates=3500,
            samples_needed=3_000_000, shot_budget=1e7, queue_wait_min=60, deadline_min=480)

PRESETS = {
    # sum_abs_coeffs=1.0 is a placeholder, NOT derived from the H2 Hamiltonian (R2 fires first anyway).
    "h2_vqe": _preset(task_id="h2_vqe", problem_class="chem", algorithm="vqe", n_qubits=4, two_qubit_gates=3,
                      sum_abs_coeffs=1.0, shot_budget=1e7),
    "maxcut20": _preset(task_id="maxcut20", problem_class="opt", algorithm="qaoa", n_qubits=20,
                        two_qubit_gates=60, samples_needed=100_000, shot_budget=1e7),
    "maxcut120": _preset(task_id="maxcut120", classical_meets_target=True, **_maxcut120),
    "maxcut120_open": _preset(task_id="maxcut120_open", classical_meets_target=False, **_maxcut120),
    "fes_sqd": _preset(task_id="fes_sqd", algorithm="sqd", **_fes),
    "fes_vqe": _preset(task_id="fes_vqe", algorithm="vqe", sum_abs_coeffs=50.0, precision_target=1e-3, **_fes),
    "lin_solve": _preset(task_id="lin_solve", problem_class="classical", algorithm="none"),
    "deep_random": _preset(task_id="deep_random", problem_class="samp", algorithm="samp", target_qpu="heron_r3",
                           n_qubits=100, two_qubit_gates=20_000, samples_needed=100_000, shot_budget=1e7,
                           queue_wait_min=60),
}
