"""python -m hybrid_mapper demo --html: refresh the data inside docs/demo.html, the visual demo.

The page is a static viewer that works offline. Everything it shows is produced here by running the real
pipeline on each showcase application: the Phase 2 prediction, the Phase 1 rule trace, the Phase 3
allocator, the Phase 4 planner, and the ground-truth execution and schedule of every candidate plan.
"""
import json
import math
import re
import time
from pathlib import Path

import networkx as nx
import pandas as pd

from hybrid_mapper import allocate as al
from hybrid_mapper import partition as pt
from hybrid_mapper.backends import get_backend
from hybrid_mapper.predict import evaluate, load_dataset
from hybrid_mapper.presets import PRESETS
from hybrid_mapper.profiler import transpile_for
from hybrid_mapper.rules import decide

PAGE = Path(__file__).resolve().parent.parent / "docs" / "demo.html"
BASE = {"precision_target": 5e-2, "maxcut_ratio_target": 0.9, "deadline_min": 480, "queue_wait_min": 30,
        "shot_budget": 1e7, "samples_needed": 1e5}
# (title, what to watch for, plan_one arguments, clustered options)
CASES = [
    ("Max-Cut, 120 nodes", "The accuracy target (0.9) is above what the classical heuristic guarantees (0.878) and far "
     "above noisy QAOA. As one task it is REDESIGN. Split into small graphs, each part is solved exactly.",
     ("maxcut", 120, 1, "none", 0, "nighthawk_r2", BASE), {}),
    ("Modular circuit, 3 x 26 qubits", "Three fully entangled blocks joined by one CZ gate each: too wide to simulate, "
     "too noisy to run. Cutting two gates leaves blocks a simulator can handle.",
     ("clustered", 26, 1, "full", 0, "heron_r3", BASE), {"blocks": 3, "links": 1}),
    ("Deep sampling circuit, 50 qubits", "The rules reject it (fidelity 1e-4). The learned allocator knows a large shot "
     "budget still buys enough clean samples. This is the QPU's job, and sampling cannot be partitioned.",
     ("random", 50, 20, "none", 0, "heron_r3", BASE | {"shot_budget": 1e8, "samples_needed": 1e4}), {}),
    ("Chain ansatz, 100 qubits", "Wide but weakly entangled, so a tensor-network simulator finishes in a second. "
     "Partitions exist, and the planner correctly turns them down: cutting would only add work.",
     ("vqe", 100, 2, "reverse_linear", 0, "heron_r3", BASE), {}),
    ("Fully entangled, 40 qubits", "Every qubit pair is entangled, so there is nothing to cut. Partitioning is not "
     "magic: this task stays REDESIGN.",
     ("vqe", 40, 2, "full", 0, "heron_r3", BASE), {}),
    ("Modular circuit, 4 x 30 qubits, 30 min", "Twelve gates to cut means over a million circuit variants for the "
     "middle blocks. The planner rejects the plan, and the ground truth agrees: it cannot finish in 30 minutes.",
     ("clustered", 30, 2, "full", 0, "heron_r3", BASE | {"deadline_min": 30}), {"blocks": 4, "links": 2}),
    ("Max-Cut, 80 nodes, easier target", "With a 0.85 target the classical heuristic is enough. The rules look only at "
     "QAOA fidelity and say REDESIGN; the allocator says CLASSICAL and no partition is needed.",
     ("maxcut", 80, 2, "none", 1, "heron_r3", BASE | {"maxcut_ratio_target": 0.85}), {}),
]
RULES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]


def dag_json(dag: nx.DiGraph) -> list[dict]:
    """Nodes with their schedule and a drawing layer (longest path from a source)."""
    layer = {}
    for n in nx.topological_sort(dag):
        layer[n] = max((layer[p] + 1 for p in dag.predecessors(n)), default=0)
    return [{"id": n, "layer": layer[n], "after": list(dag.predecessors(n)),
             **{k: d[k] for k in ("kind", "label", "duration", "start", "finish", "lane")}} for n, d in dag.nodes(data=True)]


def maxcut_picture(graph: nx.Graph, parts: list, side: dict | None) -> dict:
    """Node positions with every part laid out on its own patch of a circle; edges flagged as crossing parts."""
    k = len(parts)
    radius = min(0.6, 0.85 * math.sin(math.pi / k)) if k > 1 else 1.0
    pos, part = {}, {}
    for i, p in enumerate(parts):
        cx, cy = (math.cos(2 * math.pi * i / k), math.sin(2 * math.pi * i / k)) if k > 1 else (0.0, 0.0)
        for v, (x, y) in nx.spring_layout(graph.subgraph(p), seed=0).items():
            pos[v], part[v] = (cx + radius * x, cy + radius * y), i
    return {"kind": "graph",
            "nodes": [[round(pos[v][0], 3), round(pos[v][1], 3), part[v], side[v] if side else 0] for v in sorted(pos)],
            "edges": [[u, v, int(part[u] != part[v])] for u, v in graph.edges]}


def circuit_picture(circuit, parts: list, pieces: list) -> dict:
    """Fragments as blocks (qubits, gates inside) and the cut gates between them."""
    graph = pt.interaction_graph(circuit)
    part = {q: i for i, p in enumerate(parts) for q in p}
    links = {}
    for u, v, w in graph.edges(data="weight"):
        if part[u] != part[v]:
            key = tuple(sorted((part[u], part[v])))
            links[key] = links.get(key, 0) + w
    return {"kind": "circuit", "blocks": [[p[0], p[1], p[2]] for p in pieces],
            "links": [[a, b, w] for (a, b), w in sorted(links.items())]}


def case(title: str, story: str, args: tuple, extra: dict) -> dict:
    r = pt.plan_one(*args, **extra)
    t, w, said, rules = r["task"], r["workload"], r["ai"], r["rules"]
    allocator = pt.default_planner()[0]
    row = pd.DataFrame([t])
    t0 = time.perf_counter()
    allocator.predictor.predict(row)
    predict_ms = (time.perf_counter() - t0) * 1e3
    transpile_ms = None
    if t["n_qubits"] <= get_backend(t["backend"])["qubits"]:
        t0 = time.perf_counter()
        transpile_for(w.circuit, t["backend"])
        transpile_ms = (time.perf_counter() - t0) * 1e3
    fired = {s["rule"]: s for s in rules.trace}
    candidates = []
    for c in r["candidates"]:
        truth = c["truth"]
        picture = (maxcut_picture(w.graph, c["node_sets"], pt.solve_maxcut(w.graph, c["node_sets"])["side"])
                   if w.graph is not None else
                   circuit_picture(w.circuit, c["node_sets"], pt.partitions(r["key"])[c["candidate"] - 1]["pieces"]))
        candidates.append({k: c[k] for k in ("candidate", "parts", "sizes", "cuts", "max_cuts_per_part", "accepted")} | {
            "part_decisions": c["part_decisions"], "feasible": truth["feasible"], "why": truth["why"],
            "certified_ratio": truth.get("certified_ratio"), "cut": truth.get("cut"), "upper_bound": truth.get("upper_bound"),
            "picture": picture})
    final = r["final"]
    whole_picture = (maxcut_picture(w.graph, [sorted(w.graph)], None) if w.graph is not None else
                     {"kind": "circuit", "blocks": [[t["n_qubits"], t["analytic_2q"], 0]], "links": []})
    return {
        "title": title, "story": story, "name": t["task_id"], "family": r["key"][0], "backend": t["backend"],
        "qubits": t["n_qubits"], "picture": whole_picture,
        "conditions": {k: t[k] for k in al.CONDITIONS},
        "phase2": {"predicted": r["predicted"]["two_qubit_gates"], "exact": t["two_qubit_gates"],
                   "analytic": t["analytic_2q"], "predict_ms": predict_ms, "transpile_ms": transpile_ms},
        "phase1": {"decision": rules.decision, "rule": rules.fired_rule, "fidelity": rules.derived["fidelity"],
                   "shots": rules.derived["shots"],
                   "trace": [{"rule": name, "state": ("fire" if fired[name]["fired"] else "pass") if name in fired else "skip",
                              "reason": fired[name]["reason"] if name in fired else ""} for name in RULES]},
        "phase3": {"ai": said.ai, "guarded": bool(said.guarded),
                   "proba": {a: float(said[f"p_{a}"]) for a in ("CLASSICAL", "HYBRID", "REDESIGN")},
                   "truth": t["oracle_label"], "classical": t["classical_reason"], "hybrid": t["hybrid_reason"]},
        "phase4": {"candidates": candidates, "choice": r["choice"], "action": r["action"], "oracle": r["oracle"]["label"],
                   "why_none": None if candidates else ("Sampling tasks are not partitioned: cutting rebuilds expectation "
                                                        "values, not samples." if t["family"] == "random" else
                                                        "The splitter finds no modules: every qubit pair is entangled.")},
        "run": {"works": bool(final.get("feasible")), "why": final.get("why", ""),
                "dag": dag_json(final["dag"]) if final.get("dag") is not None else None,
                **{k: final.get(k) for k in ("time_s", "qpu_s", "energy_j", "J", "shots", "certified_ratio")}},
    }


def workflow(name: str) -> dict:
    r = pt.plan_workflow(name)
    kernels = [{"kernel": k, "task": v["task"]["task_id"], "qubits": v["task"]["n_qubits"], "phase3": v["ai"].ai,
                "truth": v["task"]["oracle_label"], "action": v["action"], "blocked": k in r["blocked"],
                "parts": v["candidates"][v["choice"] - 1]["parts"] if v["choice"] > 0 else 0, "after": v["after"],
                "why": "" if k not in r["blocked"] else v["final"].get("why", "") if v["action"] != "REDESIGN"
                else "no plan works: " + (v["candidates"][0]["truth"]["why"] if v["candidates"] else v["whole"]["why"])}
               for k, v in r["kernels"].items()]
    return {"name": name, "description": r["description"], "backend": r["backend"], "kernels": kernels,
            "blocked": r["blocked"], "dag": dag_json(r["dag"]), "deadline_s": r["deadline_s"],
            **{k: r[k] for k in ("time_s", "qpu_s", "energy_j", "J")}}


def scoreboard() -> dict:
    """The headline number of every phase, recomputed from the committed data."""
    acc, agree = evaluate(load_dataset())
    extra = acc[acc.split.str.startswith("extrapolation") & (acc.target == "two_qubit_gates")].set_index("model")
    agreement = agree[agree.split.str.startswith("extrapolation")].set_index("model")
    tasks = al.load_tasks()
    p3 = al.evaluate_split(*next(iter(al.splits(tasks).values())))["metrics"]
    apps, plans = pt.load_apps(), pt.load_plans()
    p4 = pt.evaluate_split(*next(iter(pt.splits(apps).values())), plans)["metrics"]
    best = pt.plan_oracle(apps, plans)
    stuck = (apps.oracle_label == "REDESIGN").to_numpy()
    return {
        "phase1": [{"name": n, "decision": d.decision, "rule": d.fired_rule} for n, d in
                   ((n, decide(p)) for n, p in PRESETS.items())],
        "phase2": {"mape": {m: float(extra.loc[m, "MAPE"]) for m in ("analytic", "gbt", "power-law")},
                   "agreement": float(agreement.loc["power-law", "decision agreement"]),
                   "predict_ms": float(agreement.loc["power-law", "predict+decide ms/task"]),
                   "transpile_ms": float(agreement.loc["power-law", "exact transpile ms/task"]), "rows": len(load_dataset())},
        "phase3": {"tasks": len(tasks), "rows": [
            {"name": label, "accuracy": float(p3[k]["Decision accuracy (vs oracle)"]), "regret": float(p3[k]["Mean regret (cost units)"])}
            for label, k in (("Rules as specified", "Rules (exact transpile)"), ("Rules, tuned thresholds", "Rules, tuned thresholds"),
                             ("Rules + entanglement check", "Rules + R3 entanglement check"),
                             ("Learned: decision tree", "AI: decision tree"), ("Learned: gradient boosting", "AI: gradient boosting"))]},
        "phase4": {"apps": len(apps), "plans": len(plans), "stuck": int(stuck.sum()),
                   "rescuable": int((stuck & (best.label == "PARTITION").to_numpy()).sum()), "rows": [
            {"name": label, "valid": float(p4[k]["Tasks that get a valid result"]), "regret": float(p4[k]["Mean regret (cost units)"]),
             "rescued": float(p4[k]["REDESIGN tasks rescued"])}
            for label, k in (("Rules, no partitioning", "Rules + R3, no partitioning"), ("Phase 3 alone", "Phase 3 AI, no partitioning"),
                             ("Partition whenever stuck", "Phase 4: partition whenever stuck"), ("Rule planner", "Phase 4: rule planner"),
                             ("Learned planner", "Phase 4: learned planner"), ("Plan oracle (ceiling)", "Plan oracle"))]},
    }


def build() -> dict:
    return {"cases": [case(*c) for c in CASES], "workflows": [workflow(n) for n in ("screening_pipeline", "tight_deadline")],
            "scores": scoreboard()}


def write(page: Path = PAGE) -> None:
    data = json.dumps(build(), default=lambda o: o.item() if hasattr(o, "item") else str(o), separators=(",", ":"))
    data = data.replace("</", "<\\/").replace("NaN", "null")
    html, n = re.subn(r'(<script type="application/json" id="data">).*?(</script>)', lambda m: m.group(1) + data + m.group(2),
                      page.read_text(), count=1, flags=re.S)
    if n != 1:
        raise RuntimeError(f"{page} has no data block to refresh")
    page.write_text(html)
    print(f"refreshed {page.parent.name}/{page.name} ({len(data) / 1e3:.0f} kB of pipeline output)")
