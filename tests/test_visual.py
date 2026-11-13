import json
import re

from hybrid_mapper import visual
from hybrid_mapper.partition import PLAN_LABELS


def _page_data() -> dict:
    block = re.search(r'<script type="application/json" id="data">(.*?)</script>', visual.PAGE.read_text(), re.S)
    return json.loads(block.group(1))


def test_demo_page_carries_pipeline_output_for_every_showcase():
    data = _page_data()
    assert [c["title"] for c in data["cases"]] == [c[0] for c in visual.CASES]
    for c in data["cases"]:
        p4 = c["phase4"]
        assert p4["action"] in PLAN_LABELS and p4["oracle"] in PLAN_LABELS
        assert [t["rule"] for t in c["phase1"]["trace"]] == visual.RULES
        assert (c["run"]["dag"] is not None) == (p4["action"] != "REDESIGN")
        if p4["action"] == "PARTITION":
            assert p4["candidates"][p4["choice"] - 1]["accepted"] and c["run"]["works"]
    assert {w["name"] for w in data["workflows"]} == {"screening_pipeline", "tight_deadline"}
    assert set(data["scores"]) == {"phase1", "phase2", "phase3", "phase4"}


def test_demo_page_matches_a_fresh_run_of_the_pipeline():
    stored = next(c for c in _page_data()["cases"] if c["title"] == "Fully entangled, 40 qubits")
    fresh = visual.case(*next(c for c in visual.CASES if c[0] == stored["title"]))
    for phase, key in (("phase1", "decision"), ("phase3", "ai"), ("phase3", "truth"), ("phase4", "action")):
        assert fresh[phase][key] == stored[phase][key]
    json.dumps(fresh, default=lambda o: o.item())  # everything the page needs is serialisable
