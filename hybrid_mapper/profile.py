"""TaskProfile: the single input record for the decision engine (and a Phase 2 dataset row)."""
import json
from dataclasses import asdict, dataclass, fields

PROBLEM_CLASSES = {"classical", "chem", "opt", "dyn", "samp"}
ALGORITHMS = {"none", "vqe", "dyn", "qaoa", "sqd", "samp"}


@dataclass
class TaskProfile:
    # Problem-level
    task_id: str
    problem_class: str
    algorithm: str
    precision_target: float = 1e-3
    deadline_min: float = 480.0
    classical_meets_target: bool = False
    low_entanglement: bool = False
    # Circuit-level
    n_qubits: int = 0
    depth: int = 0
    two_qubit_gates: int = 0
    sum_abs_coeffs: float = 0.0
    measurement_groups: int = 0  # qubit-wise-commuting Pauli groups; 0 = unknown / not applicable
    samples_needed: int = 0
    # Hardware-level
    target_qpu: str = "heron_r3"
    eplg: float = 0.0
    throughput_per_s: float = 0.0
    queue_wait_min: float = 0.0
    shot_budget: float = 0.0

    def __post_init__(self):
        if self.problem_class not in PROBLEM_CLASSES:
            raise ValueError(f"problem_class must be one of {sorted(PROBLEM_CLASSES)}, got {self.problem_class!r}")
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {sorted(ALGORITHMS)}, got {self.algorithm!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "TaskProfile":
        return cls(**json.loads(text))


FIELD_NAMES = [f.name for f in fields(TaskProfile)]
