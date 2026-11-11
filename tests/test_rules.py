import math
from pathlib import Path

import pytest
import yaml

from hybrid_mapper.presets import PRESETS
from hybrid_mapper.profile import TaskProfile
from hybrid_mapper.rules import decide

CASES = [
    ("h2_vqe", "CLASSICAL", "R2"),
    ("maxcut20", "CLASSICAL", "R2"),
    ("maxcut120", "CLASSICAL", "R4"),
    ("maxcut120_open", "HYBRID", "R9"),
    ("fes_sqd", "HYBRID", "R9"),
    ("fes_vqe", "REDESIGN", "R6"),
    ("lin_solve", "CLASSICAL", "R1"),
    ("deep_random", "REDESIGN", "R6"),
]


@pytest.mark.parametrize("name,decision,rule", CASES)
def test_preset_decisions(name, decision, rule):
    d = decide(PRESETS[name])
    assert (d.decision, d.fired_rule) == (decision, rule)
    assert d.trace[-1]["rule"] == rule and d.trace[-1]["fired"]
    assert all(not t["fired"] for t in d.trace[:-1])


def test_fes_sqd_numbers():
    d = decide(PRESETS["fes_sqd"])
    assert d.derived["fidelity"] == pytest.approx(5.35e-4, rel=0.02)
    assert d.derived["qpu_seconds"] == pytest.approx(750, rel=1e-6)
    assert d.split is not None


def test_threshold_change_needs_no_code_edit(tmp_path):
    cfg = yaml.safe_load(Path("config/rules.yaml").read_text())
    cfg["fidelity_min"]["recover"] = 1.0e-3  # stricter than fes_sqd's F ~ 5.35e-4
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(cfg))
    d = decide(PRESETS["fes_sqd"], rules_path=path)
    assert (d.decision, d.fired_rule) == ("REDESIGN", "R6")


def test_profile_json_roundtrip():
    p = PRESETS["fes_sqd"]
    assert TaskProfile.from_json(p.to_json()) == p
    with pytest.raises(ValueError):
        TaskProfile(task_id="x", problem_class="bogus", algorithm="none")


@pytest.mark.parametrize("bad", [{"n_qubits": -1}, {"eplg": 1.5}, {"precision_target": 0}])
def test_profile_rejects_impossible_values(bad):
    with pytest.raises(ValueError):
        TaskProfile(task_id="x", problem_class="chem", algorithm="vqe", **bad)
