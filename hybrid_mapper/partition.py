"""Phase 4: automatic partitioning. An application that is REDESIGN as one task is split into subtasks,
every subtask goes back through Phase 2 (predicted resources) and Phase 3 (allocation), and the accepted
split becomes a task DAG scheduled on classical nodes and the QPU.

One splitter (modules of a weighted graph), two ways to use it:
  Max-Cut       split the problem graph, solve every part exactly, then choose which parts to flip.
                Really executed here, so each result carries a certificate:
                cut / (sum of part optima + crossing edges) <= cut / optimum.
  gate cutting  split the circuit's qubit-interaction graph and cut the two-qubit gates between modules
                (Mitarai & Fujii 2021). Each cut CZ becomes 6 local circuits and multiplies the shots by 9.
                cut_expectation() rebuilds the uncut expectation value exactly at small sizes.
Sampling tasks are not partitioned: gate cutting reconstructs expectation values, not samples.

The planner takes the candidate with the fewest cuts that it expects to work (cutting only adds overhead).
Planners are compared against the plan oracle, which executes every candidate in the Phase 3 ground-truth
model: no partitioning (Phase 3 alone), partition whenever stuck, a rule (every part allocatable, few cuts
per fragment) and a classifier trained on executed plans (which hands over to the rule for applications
larger than any it was trained on).
"""
import json
import math
from dataclasses import replace
from functools import cache
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

from hybrid_mapper.allocate import (CONDITIONS, TASKS_PATH, Allocator, default_allocator, load_tasks,
                                    trusted_qubits)
from hybrid_mapper.backends import CONFIG_DIR, get_backend
from hybrid_mapper.dataset import build, problem_features
from hybrid_mapper.environment import cost, flatten, load_environment, oracle
from hybrid_mapper.profiler import (entanglement_bound, logical, transpile_for, two_qubit_count,
                                    workload_entanglement_bound)
from hybrid_mapper.rules import KIND, decide, load_rules
from hybrid_mapper.templates import SPLITS
from hybrid_mapper.workloads import clustered

CLUSTERED_PATH = TASKS_PATH.parent / "clustered.csv"
PLANS_PATH = TASKS_PATH.parent / "plans.csv"
SPEC = ["app_family", "family", "n_qubits", "reps", "entanglement", "seed", "blocks", "links"]


# ---------------------------------------------------------------- splitting a graph into modules

def interaction_graph(circuit) -> nx.Graph:
    """Qubits as nodes; an edge's weight is the number of two-qubit gates on that pair (what cutting it costs)."""
    dec = logical(circuit)
    g = nx.Graph()
    g.add_nodes_from(range(dec.num_qubits))
    for inst in dec.data:
        if inst.operation.num_qubits == 2:
            a, b = (dec.find_bit(q).index for q in inst.qubits)
            g.add_edge(a, b, weight=g.get_edge_data(a, b, {"weight": 0})["weight"] + 1)
    return g


def split(graph: nx.Graph, cap: int | None = None, resolution: float = 1.0, seed: int = 0) -> list[list]:
    """Modules of a weighted graph: Louvain communities, bisected (Kernighan-Lin) until none exceeds cap."""
    parts = [set(c) for c in nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)]
    while cap and (big := next((p for p in parts if len(p) > cap), None)):
        parts.remove(big)
        parts += nx.community.kernighan_lin_bisection(graph.subgraph(big), weight="weight", seed=seed)
    return sorted((sorted(p) for p in parts), key=lambda p: p[0])


def crossing(graph: nx.Graph, parts: list) -> int:
    """Total weight of the edges between different parts."""
    part = {v: i for i, p in enumerate(parts) for v in p}
    return sum(w for u, v, w in graph.edges(data="weight", default=1) if part[u] != part[v])


# ---------------------------------------------------------------- Max-Cut: divide, solve exactly, merge

def exact_maxcut(graph: nx.Graph) -> tuple[int, dict]:
    """Maximum (weighted) cut by enumerating all 2^(n-1) cuts. Returns (value, side of every node)."""
    nodes = list(graph)
    index = {v: i for i, v in enumerate(nodes)}
    cuts = np.arange(2 ** (len(nodes) - 1), dtype=np.uint32)  # the last node stays on side 0
    value = np.zeros(len(cuts), dtype=np.int32)
    for u, v, w in graph.edges(data="weight", default=1):
        value += w * (((cuts >> index[u]) ^ (cuts >> index[v])) & 1).astype(np.int32)
    best = int(value.argmax())
    return int(value[best]), {v: (best >> index[v]) & 1 for v in nodes}


def solve_maxcut(graph: nx.Graph, parts: list) -> dict:
    """Run a Max-Cut partition plan: each part exactly, then the best choice of parts to flip, then
    single-node improvements. The optimum is at most (sum of part optima + crossing edges), so
    cut / upper_bound is a certified approximation ratio for this instance."""
    side, optimum = {}, 0
    for p in parts:
        value, sides = exact_maxcut(graph.subgraph(p))
        optimum += value
        side |= sides
    part = {v: i for i, p in enumerate(parts) for v in p}
    coarse = nx.Graph()  # one node per part; flipping one end of an edge gains its weight
    coarse.add_nodes_from(range(len(parts)))
    for u, v in graph.edges:
        if part[u] != part[v]:
            gain = 1 if side[u] == side[v] else -1
            a, b = part[u], part[v]
            coarse.add_edge(a, b, weight=coarse.get_edge_data(a, b, {"weight": 0})["weight"] + gain)
    if len(parts) <= 22:  # ponytail: exhaustive merge up to 2^21 flip patterns; beyond that only the local search
        flip = exact_maxcut(coarse)[1]
        side = {v: s ^ flip[part[v]] for v, s in side.items()}
    improved = True
    while improved:  # move any node that has more neighbours on its own side than across
        improved = False
        for v in graph:
            if sum(1 if side[v] == side[u] else -1 for u in graph[v]) > 0:
                side[v] ^= 1
                improved = True
    cut = sum(side[u] != side[v] for u, v in graph.edges)
    bound = optimum + crossing(graph, parts)
    return {"cut": cut, "upper_bound": bound, "certified_ratio": cut / bound if bound else 1.0, "side": side,
            "coarse_edges": coarse.number_of_edges()}


# ---------------------------------------------------------------- gate cutting

# One CZ = (phase) Rz(pi/2) x Rz(pi/2) exp(i pi/4 Z x Z). As a channel the last factor is
#   1/2 [I x I] + 1/2 [Z x Z] + 1/2 sum over a, b = +-1 of a b ([P_a x R_b] + [R_a x P_b]),
# with P_a the projector on Z = a and R_b = exp(i b pi/4 Z). "M" below is "measure Z and multiply the
# result by the outcome" (sum_a a P_a rho P_a), so six local circuits rebuild the gate. Their weights
# sum to gamma = 3 in magnitude; a quasi-probability estimate therefore needs gamma^2 = 9 times the shots.
QPD = [(0.5, "I", "I"), (0.5, "Z", "Z"), (0.5, "M", "R+"), (-0.5, "M", "R-"), (0.5, "R+", "M"), (-0.5, "R-", "M")]
KET_BRA = {"M": [(0.5, "Z", "I"), (0.5, "I", "Z")]}  # M(rho) = (Z rho + rho Z) / 2 as (weight, ket op, bra op)


def _programs(circuit, parts: list) -> tuple[list, int]:
    """Per part, its gates in order as (operation, local qubits). A gate between two parts is cut: it
    appears in both as ((cut index, side), [local qubit])."""
    dec = logical(circuit)
    where = {q: (f, i) for f, p in enumerate(parts) for i, q in enumerate(p)}
    programs, n_cuts = [[] for _ in parts], 0
    for inst in dec.data:
        if inst.operation.name == "barrier":
            continue
        qubits = [where[dec.find_bit(q).index] for q in inst.qubits]
        if len({f for f, _ in qubits}) == 1:
            programs[qubits[0][0]].append((inst.operation, [i for _, i in qubits]))
        else:
            for side, (f, i) in enumerate(qubits):
                programs[f].append(((n_cuts, side), [i]))
            n_cuts += 1
    return programs, n_cuts


def fragments(circuit, parts: list) -> tuple[list, int]:
    """[(fragment circuit, cut gates touching it)] per part, and the total number of cut gates."""
    programs, n_cuts = _programs(circuit, parts)
    out = []
    for part, program in zip(parts, programs):
        frag = QuantumCircuit(len(part))
        for op, qubits in program:
            if not isinstance(op, tuple):
                frag.append(op, qubits)
        out.append((frag, sum(isinstance(op, tuple) for op, _ in program)))
    return out, n_cuts


def cut_expectation(circuit, hamiltonian: SparsePauliOp, parts: list) -> float:
    """<H> of a (parameter-free) circuit computed only from its fragments: every gate between parts is
    replaced by the six-term decomposition above. Exact, and exponential in the number of cuts: this is
    the small-scale check that a gate-cut plan gives the right answer, not a simulator."""
    programs, n_cuts = _programs(circuit, parts)
    sides = {(op[0], f): op[1] for f, program in enumerate(programs) for op, _ in program if isinstance(op, tuple)}
    where = {q: (f, i) for f, p in enumerate(parts) for i, q in enumerate(p)}

    @cache
    def state(f: int, ops: tuple) -> np.ndarray:  # fragment f with local operation ops[k] at cut k
        qc = QuantumCircuit(len(parts[f]))
        for op, qubits in programs[f]:
            if not isinstance(op, tuple):
                qc.append(op, qubits)
                continue
            qc.rz(math.pi / 2, qubits)  # this qubit's local phase of the CZ
            name = ops[op[0]]
            if name == "Z":
                qc.z(qubits)
            elif name in ("R+", "R-"):
                qc.rz(-math.pi / 2 if name == "R+" else math.pi / 2, qubits)
        return Statevector(qc).data

    @cache
    def pauli(f: int, letters: str, qubits: tuple) -> np.ndarray:
        return SparsePauliOp.from_sparse_list([(letters, list(qubits), 1.0)], len(parts[f])).to_matrix()

    @cache
    def fragment_value(f: int, ket: tuple, bra: tuple, letters: str, qubits: tuple) -> complex:
        k, b = state(f, ket), state(f, bra)
        return np.vdot(b, pauli(f, letters, qubits) @ k if letters else k)

    terms = []  # (coefficient, per fragment: (letters, local qubits))
    for letters, qubits, coeff in hamiltonian.to_sparse_list():
        local = [("", ())] * len(parts)
        for letter, q in zip(letters, qubits):
            f, i = where[q]
            local[f] = (local[f][0] + letter, local[f][1] + (i,))
        terms.append((coeff, local))
    total = 0.0
    for choice in product(range(len(QPD)), repeat=n_cuts):
        weight = math.prod(QPD[c][0] for c in choice)
        pairs = []  # per fragment: [(weight, ket ops, bra ops)]
        for f in range(len(parts)):
            ops = [KET_BRA.get(op, [(1.0, op, op)]) for op in
                   (QPD[c][1 + sides[k, f]] if (k, f) in sides else None for k, c in enumerate(choice))]
            pairs.append([(math.prod(o[0] for o in combo), tuple(o[1] for o in combo), tuple(o[2] for o in combo))
                          for combo in product(*ops)])
        for coeff, local in terms:
            value = weight * coeff
            for f, (letters, qubits) in enumerate(local):
                value *= sum(w * fragment_value(f, ket, bra, letters, qubits) for w, ket, bra in pairs[f])
            total += value
    return float(np.real(total))


# ---------------------------------------------------------------- applications and their candidate partitions

@cache
def workload(app_family: str, family: str, n_qubits: int, reps: int, entanglement: str, seed: int, blocks: int = 1,
             links: int = 1):
    if app_family == "clustered":
        return clustered(blocks, n_qubits // blocks, reps, links, seed)
    return build(family, n_qubits, reps, entanglement, seed)


def spec(t: dict) -> tuple:
    return (t["app_family"], t["family"], int(t["n_qubits"]), int(t["reps"]), t["entanglement"], int(t["seed"]),
            int(t["blocks"]), int(t["links"]))


@cache
def partitions(key: tuple) -> tuple:
    """Every way the splitter finds to partition one workload, fewest cuts first. Each has its parts, the
    number of cuts, and per part [size, two-qubit gates (or edges) inside, cuts touching it, entanglement bound]."""
    w = workload(*key)
    cfg = load_environment()["partition"]
    if w.graph is not None:
        graph, cap = w.graph, cfg["maxcut_part_nodes"]
    elif KIND[w.algorithm] == "expect":
        graph, cap = interaction_graph(w.circuit), None
    else:
        return ()
    found = {}
    for resolution in cfg["resolutions"]:
        parts = split(graph, cap, resolution)
        if len(parts) > 1:
            found[tuple(map(tuple, parts))] = parts
    out = []
    for parts in found.values():
        if w.graph is not None:
            pieces = [[len(p), graph.subgraph(p).number_of_edges(), 0, len(p)] for p in parts]
            out.append({"parts": parts, "cuts": crossing(graph, parts), "pieces": pieces})
        else:
            frags, cuts = fragments(w.circuit, parts)
            pieces = [[f.num_qubits, two_qubit_count(f), k, entanglement_bound(f)] for f, k in frags]
            out.append({"parts": parts, "cuts": cuts, "pieces": pieces, "fragments": [f for f, _ in frags]})
    return tuple(sorted(out, key=lambda c: c["cuts"]))


def subtasks(t: dict, pieces: list, cuts: int) -> pd.DataFrame:
    """The parts of one partition as Phase 3 task rows, so the same predictor, rules and allocator apply.

    A Max-Cut part has to be solved exactly (ratio target 1). A circuit fragment estimates the whole
    observable to precision eps / gamma^cuts, which is the gamma^(2 cuts) shot overhead in the units the
    rules already use, and it is run once per variant (6^cuts touching it).
    """
    cfg = load_environment()["partition"]
    base = {c: t[c] for c in [*CONDITIONS, "backend", "reps", "seed", "family", "entanglement", "queue_real_min"]}
    rows = []
    for size, inside, touching, ebits in pieces:
        if t["family"] == "maxcut":
            row = dict(n_qubits=size, n_edges=inside, n_pauli_terms=inside, sum_abs_coeffs=float(inside),
                       analytic_2q=2 * int(t["reps"]) * inside, maxcut_ratio_target=1.0, variants=1)
        else:
            row = dict(n_qubits=size, n_edges=0, n_pauli_terms=round(t["n_pauli_terms"] * size / t["n_qubits"]),
                       sum_abs_coeffs=t["sum_abs_coeffs"], analytic_2q=inside,
                       precision_target=t["precision_target"] / cfg["gate_cut_gamma"] ** cuts,
                       variants=float(cfg["gate_cut_circuits"]) ** touching)
        rows.append(base | row | {"entanglement_bound": ebits, "cuts_touching": touching, "workload_id": "subtask"})
    return pd.DataFrame(rows)


_labels: dict = {}


def fragment_labels(frag: QuantumCircuit, backend: str) -> dict:
    """Exact transpiled size of a fragment (cached by gate structure: identical blocks transpile once)."""
    key = (backend, frag.num_qubits, tuple((i.operation.name, tuple(frag.find_bit(q).index for q in i.qubits))
                                           for i in frag.data))
    if key not in _labels:
        if frag.num_qubits > get_backend(backend)["qubits"]:
            _labels[key] = {"two_qubit_gates": two_qubit_count(frag), "depth": 0}  # cannot run there anyway
        else:
            isa = transpile_for(frag, backend)
            _labels[key] = {"two_qubit_gates": two_qubit_count(isa), "depth": isa.depth()}
    return _labels[key]


# ---------------------------------------------------------------- task DAG and schedule

def task_dag(name: str, steps: list, algorithm: str | None = None, merge_s: float | None = None) -> nx.DiGraph:
    """Nodes carry kind (classical/quantum), duration (s) and a label; edges are "must finish before".

    One step and no merge: the task as it is (a hybrid task gets classical prepare/post-process steps
    around its QPU job, from the Phase 1 work split). Otherwise: split -> steps in parallel -> merge.
    """
    overhead = load_environment()["classical"]["job_overhead_s"]
    dag = nx.DiGraph()

    def add(node, kind, duration, label, after=()):
        dag.add_node(f"{name}/{node}", kind=kind, duration=float(duration), label=label)
        dag.add_edges_from((f"{name}/{a}", f"{name}/{node}") for a in after)

    if merge_s is None:
        kind, duration, label = steps[0]
        if kind == "classical":
            add("run", kind, duration, label)
        else:
            add("prepare", "classical", overhead, "build and transpile the circuit")
            add("qpu", "quantum", duration, SPLITS[algorithm]["quantum"], ["prepare"])
            add("post", "classical", overhead, SPLITS[algorithm]["classical"], ["qpu"])
        return dag
    add("split", "classical", overhead, "partition the problem")
    for i, (kind, duration, label) in enumerate(steps, 1):
        add(f"part{i}", kind, duration, label, ["split"])
    add("merge", "classical", merge_s, "put the parts together", [f"part{i}" for i in range(1, len(steps) + 1)])
    return dag


def schedule(dag: nx.DiGraph, cpu_nodes: int, queue_s: float = 0.0) -> float:
    """Classical steps share cpu_nodes; quantum steps take turns on the one QPU, which is free after
    queue_s. Sets start, finish and lane on every node and returns the makespan."""
    free = {"classical": [0.0] * cpu_nodes, "quantum": [queue_s]}
    for node in nx.topological_sort(dag):  # ponytail: list scheduling in topological order, not an optimal schedule
        d = dag.nodes[node]
        ready = max((dag.nodes[p]["finish"] for p in dag.predecessors(node)), default=0.0)
        pool = free[d["kind"]]
        i = min(range(len(pool)), key=pool.__getitem__)
        d["start"] = max(ready, pool[i])
        d["finish"] = pool[i] = d["start"] + d["duration"]
        d["lane"] = f"{'CPU' if d['kind'] == 'classical' else 'QPU'} {i + 1}"
    return max(d["finish"] for _, d in dag.nodes(data=True))


def totals(dag: nx.DiGraph, makespan: float, env: dict) -> dict:
    """Time, energy, QPU time and cost J of a scheduled DAG, in the same units as Phase 3."""
    busy = {k: sum(d["duration"] for _, d in dag.nodes(data=True) if d["kind"] == k) for k in ("classical", "quantum")}
    energy = busy["classical"] * env["classical"]["node_watts"] + busy["quantum"] * env["quantum"]["qpu_system_watts"]
    return {"time_s": makespan, "energy_j": energy, "qpu_s": busy["quantum"], "J": cost(makespan, energy, busy["quantum"])}


# ---------------------------------------------------------------- ground truth for one plan

def execute(t: dict, candidate: dict | None, env: dict | None = None) -> dict:
    """What a plan really does. None is the task as it is (its Phase 3 oracle result). A partition runs
    every part through the Phase 3 ground-truth model on exactly transpiled fragments, schedules the DAG,
    and for Max-Cut really solves the instance to get its certified ratio."""
    env = env or load_environment()
    c, part_cfg = env["classical"], env["partition"]
    deadline_s, name = t["deadline_min"] * 60, t.get("task_id", "task")
    if candidate is None:
        label = t["oracle_label"]
        out = {"feasible": label != "REDESIGN", "labels": [label], "why": "the task as it is"}
        if label == "REDESIGN":
            return out | {"J": 0.0, "time_s": 0.0, "energy_j": 0.0, "qpu_s": 0.0, "shots": 0.0,
                          "why": f"classical: {t['classical_reason']}; quantum: {t['hybrid_reason']}"}
        side = label.lower()
        step = ("classical", t["classical_time_s"], t["classical_method"]) if label == "CLASSICAL" else \
               ("quantum", t["hybrid_qpu_s"], "")
        dag = task_dag(name, [step], algorithm={"maxcut": "qaoa", "vqe": "vqe", "random": "samp"}[t["family"]])
        schedule(dag, c["cpu_nodes"], t["queue_real_min"] * 60)
        return out | {"J": t[f"{side}_J"], "time_s": t[f"{side}_time_s"], "energy_j": t[f"{side}_energy_j"],
                      "qpu_s": t[f"{side}_qpu_s"], "shots": t[f"{side}_shots"], "dag": dag}

    rows = subtasks(t, candidate["pieces"], candidate["cuts"])
    is_maxcut = t["family"] == "maxcut"
    if is_maxcut:  # QAOA can never reach ratio 1, so its transpiled size cannot change the outcome
        rows["two_qubit_gates"] = rows.analytic_2q
    else:
        rows["two_qubit_gates"] = [fragment_labels(f, t["backend"])["two_qubit_gates"] for f in candidate["fragments"]]
    steps, labels, shots, problems = [], [], 0.0, []
    for i, r in enumerate(rows.to_dict("records"), 1):
        o = oracle(r, env)
        labels.append(o["label"])
        if o["label"] == "HYBRID":
            steps.append(("quantum", o["hybrid"]["qpu_s"], f"{r['n_qubits']} qubits on the QPU"))
            shots += o["hybrid"]["shots"]
            continue
        if o["label"] == "REDESIGN":
            problems.append(f"part {i} ({r['n_qubits']} qubits): {o['classical']['reason']}")
        variants = f" x {r['variants']:.0f} variants" if r["variants"] > 1 else ""
        steps.append(("classical", o["classical"]["time_s"], f"{r['n_qubits']} qubits, {o['classical']['method']}{variants}"))
    merge_s = c["job_overhead_s"]
    out = {"labels": labels}
    if is_maxcut:
        solution = solve_maxcut(workload(*spec(t)).graph, candidate["parts"])
        merge_s += 2.0 ** (len(labels) - 1) * solution["coarse_edges"] / c["enumeration_rate"]
        out |= {k: solution[k] for k in ("cut", "upper_bound", "certified_ratio")}
        if solution["certified_ratio"] < t["maxcut_ratio_target"]:
            problems.append(f"certified ratio {solution['certified_ratio']:.3f} < target {t['maxcut_ratio_target']}")
    dag = task_dag(name, steps, merge_s=merge_s)
    makespan = schedule(dag, c["cpu_nodes"], t["queue_real_min"] * 60)
    if makespan > deadline_s:
        problems.append(f"takes {makespan / 60:.0f} min > deadline {t['deadline_min']:g} min")
    if shots > t["shot_budget"]:
        problems.append(f"shots {shots:.3g} > budget {t['shot_budget']:.3g}")
    out |= totals(dag, min(makespan, deadline_s), env)  # a plan that overruns is stopped at the deadline
    return out | {"feasible": not problems, "shots": shots, "dag": dag,
                  "why": "; ".join(problems) or f"all {len(labels)} parts run, merged in time"}


# ---------------------------------------------------------------- the Phase 4 dataset

def app_features(key: tuple) -> dict:
    """Problem-level features of one application, as Phases 2 and 3 expect them. A clustered circuit is
    shown to them as family vqe / entanglement full: its blocks are fully entangled, but the circuit as a
    whole is outside every family they were trained on."""
    w = workload(*key)
    if key[0] != "clustered":
        return dict(zip(SPEC, key)) | problem_features(key[1], key[2], key[3], key[4], key[5], workload=w)
    return dict(zip(SPEC, key)) | {
        "n_edges": 0, "n_pauli_terms": len(w.hamiltonian), "sum_abs_coeffs": float(sum(abs(w.hamiltonian.coeffs))),
        "analytic_2q": two_qubit_count(logical(w.circuit))}


def clustered_apps() -> pd.DataFrame:
    """The clustered sweep as Phase 3 task rows: exact transpile, entanglement bound, sampled operating
    conditions and the oracle label for running each one as a single task."""
    env = load_environment()
    sc = env["scenarios"]
    rng = np.random.default_rng(sc["seed"] + 4)
    rows = []
    for blocks, size, reps, links in product([2, 3, 4], [16, 20, 26, 30], [1, 2], [1, 2]):
        key = ("clustered", "vqe", blocks * size, reps, "full", 0, blocks, links)
        w = workload(*key)
        feats = app_features(key) | {"entanglement_bound": entanglement_bound(w.circuit)}
        for backend in ("heron_r3", "nighthawk_r2"):
            wid = f"clustered-{blocks}x{size}-l{links}-r{reps}-{backend}"
            exact = fragment_labels(w.circuit, backend)
            for _ in range(sc["per_workload"]):
                cond = {k: rng.choice(sc[k]).item() for k in CONDITIONS}
                cond["queue_real_min"] = cond["queue_wait_min"] * rng.lognormal(0, env["quantum"]["queue_lognormal_sigma"])
                t = feats | exact | cond | {"backend": backend, "workload_id": wid}
                rows.append(t | flatten(oracle(t, env)))
    return pd.DataFrame(rows)


def load_apps() -> pd.DataFrame:
    """Every application Phase 4 is evaluated on: the Phase 3 tasks plus the clustered sweep."""
    if not CLUSTERED_PATH.exists():
        raise FileNotFoundError(f"{CLUSTERED_PATH} missing: run `python -m hybrid_mapper plans` first")
    tasks = load_tasks()
    apps = pd.concat([tasks.assign(app_family=tasks.family, blocks=1, links=1), pd.read_csv(CLUSTERED_PATH)],
                     ignore_index=True)
    apps["app_id"] = apps.workload_id + "#" + apps.groupby("workload_id").cumcount().astype(str)
    return apps


def build_plans() -> pd.DataFrame:
    """Execute every candidate partition of every application in the ground-truth model
    (data/clustered.csv, data/plans.csv). One row per (application, candidate)."""
    clustered_apps().to_csv(CLUSTERED_PATH, index=False)
    apps, env, rows = load_apps(), load_environment(), []
    for n, t in enumerate(apps.to_dict("records"), 1):
        for i, cand in enumerate(partitions(spec(t)), 1):
            r = execute(t, cand, env)
            rows.append({"app_id": t["app_id"], "candidate": i, "parts": len(cand["pieces"]), "cuts": cand["cuts"],
                         "pieces": json.dumps(cand["pieces"]), "feasible": r["feasible"], "J": r["J"],
                         "time_s": r["time_s"], "energy_j": r["energy_j"], "qpu_s": r["qpu_s"], "shots": r["shots"],
                         "classical_parts": r["labels"].count("CLASSICAL"), "quantum_parts": r["labels"].count("HYBRID"),
                         "certified_ratio": r.get("certified_ratio"), "why": r["why"]})
        if n % 500 == 0 or n == len(apps):
            print(f"{n}/{len(apps)} applications planned", flush=True)
    plans = pd.DataFrame(rows)
    plans.to_csv(PLANS_PATH, index=False)
    print(f"wrote {len(plans)} candidate plans for {plans.app_id.nunique()} of {len(apps)} applications "
          f"to data/{PLANS_PATH.name} ({int(plans.feasible.sum())} feasible)")
    return plans


def load_plans() -> pd.DataFrame:
    if not PLANS_PATH.exists():
        raise FileNotFoundError(f"{PLANS_PATH} missing: run `python -m hybrid_mapper plans` first")
    return pd.read_csv(PLANS_PATH)


# ---------------------------------------------------------------- planners

PLAN_LABELS = ["CLASSICAL", "HYBRID", "PARTITION", "REDESIGN"]
EXTRAPOLATION_CUTOFF = 40


def plan_oracle(apps: pd.DataFrame, plans: pd.DataFrame) -> pd.DataFrame:
    """Per application: the cheapest plan that works, among the task as it is and every partition.
    Columns: best (0 = as it is, i = candidate i, -1 = nothing works), J and the plan label."""
    mono = np.select([apps.oracle_label == "CLASSICAL", apps.oracle_label == "HYBRID"],
                     [apps.classical_J, apps.hybrid_J], np.inf)
    part = plans[plans.feasible.astype(bool)].sort_values("J").groupby("app_id").first().reindex(apps.app_id)
    wins = (part.J.to_numpy() < mono)  # no feasible partition: NaN, never wins
    return pd.DataFrame({"best": np.where(wins, part.candidate.fillna(0), np.where(np.isfinite(mono), 0, -1)).astype(int),
                         "J": np.where(wins, part.J, np.where(np.isfinite(mono), mono, 0.0)),
                         "label": np.where(wins, "PARTITION", apps.oracle_label)}, index=apps.app_id)


def candidate_table(apps: pd.DataFrame, plans: pd.DataFrame, allocator: Allocator) -> pd.DataFrame:
    """What a planner may know about each candidate before anything runs: its structure, the application's
    conditions, and what Phases 2 and 3 say about the task as it is and about every part."""
    mono = allocator.decide(apps).set_index(apps.app_id.to_numpy())
    by_id = apps.set_index("app_id").to_dict("index")
    parts = pd.concat([subtasks(by_id[p.app_id], json.loads(p.pieces), p.cuts).assign(row=i)
                       for i, p in enumerate(plans.itertuples())], ignore_index=True)
    said = allocator.decide(parts)
    per = pd.DataFrame({"row": parts.row, "size": parts.n_qubits, "touching": parts.cuts_touching,
                        "inside": parts.analytic_2q, "gates": said.two_qubit_gates,
                        "ai_no": said.ai == "REDESIGN", "rules_no": said.rules == "REDESIGN"}).groupby("row")
    app = apps.set_index("app_id").loc[plans.app_id].reset_index()
    m = mono.loc[plans.app_id].reset_index(drop=True)
    return pd.DataFrame({
        "cuts": plans.cuts.to_numpy(), "parts": plans.parts.to_numpy(),
        "largest_part": per["size"].max().to_numpy(), "max_cuts_per_part": per.touching.max().to_numpy(),
        "cut_share": plans.cuts.to_numpy() / (plans.cuts.to_numpy() + per.inside.sum().to_numpy()),
        "log10_largest_part_gates": np.log10(per.gates.max().to_numpy() + 1),
        "parts_ai_rejects": per.ai_no.mean().to_numpy(), "parts_rules_reject": per.rules_no.mean().to_numpy(),
        "is_maxcut": (app.family == "maxcut").astype(float), "n_qubits": app.n_qubits, "reps": app.reps,
        "on_heron": (app.backend == "heron_r3").astype(float), "deadline_min": app.deadline_min,
        "maxcut_ratio_target": app.maxcut_ratio_target, "log10_precision_target": np.log10(app.precision_target),
        "log10_shot_budget": np.log10(app.shot_budget),
        "whole_p_classical": m.p_CLASSICAL, "whole_p_hybrid": m.p_HYBRID, "whole_p_redesign": m.p_REDESIGN,
    })


def choose(apps: pd.DataFrame, plans: pd.DataFrame, whole: pd.Series, accept: np.ndarray, always: bool) -> pd.DataFrame:
    """Turn per-candidate accept flags into one choice per application: the accepted candidate with the
    fewest cuts. `whole` is the Phase 3 decision for the task as it is; unless `always`, partitions are
    only considered when that decision is REDESIGN. choice: 0 = as it is, i = candidate i, -1 = decline."""
    first = (plans[accept].sort_values("cuts").groupby("app_id").candidate.first().reindex(apps.app_id)
             .to_numpy(dtype=float))
    use = ~np.isnan(first) & (always | (whole.to_numpy() == "REDESIGN"))
    choice = np.where(use, np.nan_to_num(first, nan=0), np.where(whole.to_numpy() == "REDESIGN", -1, 0)).astype(int)
    return pd.DataFrame({"choice": choice, "action": np.where(choice > 0, "PARTITION", whole.to_numpy())},
                        index=apps.app_id)


def score(chosen: pd.DataFrame, apps: pd.DataFrame, plans: pd.DataFrame, best: pd.DataFrame, penalty: float) -> dict:
    """Same accounting as Phase 3: regret is the extra cost over the plan oracle, and a plan that yields
    no accepted result pays the failure penalty on top of what it wasted."""
    truth = plans.set_index(["app_id", "candidate"])
    rows = []
    for t, (choice, action), (_, j_best, label) in zip(apps.to_dict("records"), chosen.to_numpy(), best.to_numpy()):
        if action == "REDESIGN":  # declining: right only if nothing works, otherwise the missed-result penalty
            rows.append({"correct": label == "REDESIGN", "ok": True, "delivered": False, "qpu_s": 0.0,
                         "regret": 0.0 if label == "REDESIGN" else penalty, "rescued": False})
            continue
        if choice > 0:
            p = truth.loc[(t["app_id"], choice)]
            ok, j, qpu = bool(p.feasible), p.J, p.qpu_s
        else:
            side = action.lower()
            ok, j, qpu = bool(t[f"{side}_feasible"]), t[f"{side}_J"], t[f"{side}_qpu_s"]
        rows.append({"correct": action == label, "ok": ok, "delivered": ok, "qpu_s": qpu,
                     "regret": j - j_best if ok else j + penalty,
                     "rescued": ok and t["oracle_label"] == "REDESIGN"})
    r = pd.DataFrame(rows)
    stuck = (apps.oracle_label == "REDESIGN").to_numpy()
    return {"Decision accuracy (vs plan oracle)": r.correct.mean(), "Tasks that get a valid result": r.delivered.mean(),
            "Failed attempts": (~r.ok).mean(),
            "REDESIGN tasks rescued": r.rescued.sum() / max(stuck.sum(), 1),
            "Mean regret (cost units)": r.regret.mean(), "QPU time (s, mean)": r.qpu_s.mean()}


def worth_it(plans: pd.DataFrame, apps: pd.DataFrame) -> np.ndarray:
    """Training label for the learned planner: the candidate works and costs less than the task as it is."""
    mono = pd.Series(np.select([apps.oracle_label == "CLASSICAL", apps.oracle_label == "HYBRID"],
                               [apps.classical_J, apps.hybrid_J], np.inf), index=apps.app_id)
    return (plans.feasible & (plans.J < mono.loc[plans.app_id].to_numpy())).to_numpy()


def fit_planner(X: pd.DataFrame, y: np.ndarray) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0).fit(X, y)


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame, plans: pd.DataFrame) -> dict:
    """Fit the Phase 3 allocator (on the training applications that are Phase 3 tasks) and the learned
    planner (on the training candidates), then score every planner on the test applications."""
    env = load_environment()
    penalty, cap = env["scenarios"]["failure_penalty_s"], env["partition"]["max_cuts_per_fragment"]
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    allocator = Allocator(train[train.app_family != "clustered"])
    p_tr = plans[plans.app_id.isin(train.app_id)].reset_index(drop=True)
    p_te = plans[plans.app_id.isin(test.app_id)].reset_index(drop=True)
    X_tr, X_te = candidate_table(train, p_tr, allocator), candidate_table(test, p_te, allocator)
    model = fit_planner(X_tr, worth_it(p_tr, train))
    said = allocator.decide(test)
    whole, whole_rules = said.ai.set_axis(test.app_id), said.rules.set_axis(test.app_id)
    allocatable = (X_te.parts_ai_rejects == 0).to_numpy()
    few_cuts = allocatable & (X_te.max_cuts_per_part <= cap).to_numpy()
    never = np.zeros(len(p_te), dtype=bool)
    rule = choose(test, p_te, whole, few_cuts, False)
    learned = choose(test, p_te, whole, model.predict(X_te).astype(bool), True)
    apps_in_range = (test.n_qubits <= trusted_qubits(train)).to_numpy()  # range guard, as in the Phase 3 allocator
    best = plan_oracle(test, p_te)
    chosen = {
        "Rules + R3, no partitioning": choose(test, p_te, whole_rules, never, False),
        "Phase 3 AI, no partitioning": choose(test, p_te, whole, never, False),
        "Phase 4: partition whenever stuck": choose(test, p_te, whole, ~never, False),
        "Phase 4: rule planner": rule,
        "Phase 4: learned planner": learned.where(pd.Series(apps_in_range, index=test.app_id), rule, axis=0),
        "Plan oracle": best.rename(columns={"best": "choice", "label": "action"})[["choice", "action"]],
    }
    return {"metrics": {k: score(c, test, p_te, best, penalty) for k, c in chosen.items()}, "chosen": chosen,
            "best": best, "test": test, "plans": p_te, "X_test": X_te, "model": model, "allocator": allocator}


def splits(apps: pd.DataFrame, seed: int = 0) -> dict:
    tr, te = next(GroupShuffleSplit(test_size=0.3, random_state=seed).split(apps, groups=apps.workload_id))
    small = apps.n_qubits <= EXTRAPOLATION_CUTOFF
    return {"random 30% of workloads held out": (apps.iloc[tr], apps.iloc[te]),
            f"extrapolation (train n<={EXTRAPOLATION_CUTOFF}, test n>{EXTRAPOLATION_CUTOFF})": (apps[small], apps[~small])}


def stability(apps: pd.DataFrame, plans: pd.DataFrame, seeds=range(5)) -> pd.DataFrame:
    """The headline metrics over several random workload splits (mean and standard deviation)."""
    cols = {"Tasks that get a valid result": "{:.3f}", "REDESIGN tasks rescued": "{:.3f}", "Mean regret (cost units)": "{:,.0f}"}
    rows = [{"planner": name} | {c: m[c] for c in cols}
            for seed in seeds for name, m in evaluate_split(*next(iter(splits(apps, seed).values())), plans)["metrics"].items()]
    agg = pd.DataFrame(rows).groupby("planner", sort=False).agg(["mean", "std"])
    return pd.DataFrame({"Planner": agg.index} | {c: [f"{fmt.format(m)} ± {fmt.format(sd)}" for m, sd in agg[c].to_numpy()]
                                                  for c, fmt in cols.items()})


# ---------------------------------------------------------------- one application, end to end

@cache
def default_planner() -> tuple:
    """(allocator, learned planner) trained on every application: for the CLI and the demos only."""
    apps, plans = load_apps(), load_plans()
    allocator = default_allocator()
    return allocator, fit_planner(candidate_table(apps, plans, allocator), worth_it(plans, apps))


def dag_rows(dag: nx.DiGraph) -> list[dict]:
    return [{"id": n, "after": list(dag.predecessors(n))} | d for n, d in dag.nodes(data=True)]


def plan_one(app_family: str, size: int, reps: int, entanglement: str, seed: int, backend: str, conditions: dict,
             blocks: int = 1, links: int = 1) -> dict:
    """All four phases for one application. The planner sees only pre-execution information; the exact
    transpile and the ground-truth execution of every candidate are attached for comparison."""
    allocator, model = default_planner()
    env = load_environment()
    family = "vqe" if app_family == "clustered" else app_family
    key = (app_family, family, size * blocks, reps, "full" if app_family == "clustered" else entanglement, seed, blocks, links)
    w = workload(*key)
    t = app_features(key) | conditions | {"backend": backend, "workload_id": w.name, "task_id": w.name, "app_id": "app",
                                          "queue_real_min": conditions["queue_wait_min"],  # nominal queue for one-off runs
                                          "entanglement_bound": workload_entanglement_bound(w)}
    t |= fragment_labels(w.circuit, backend)
    t |= flatten(oracle(t, env))
    apps = pd.DataFrame([t])
    pred, profs, _ = allocator.inputs(apps)
    said = allocator.decide(apps).iloc[0]
    low = bool(t["entanglement_bound"] <= load_rules()["low_entanglement_max_ebits"])
    rules = decide(replace(profs[0], low_entanglement=low))
    cands = partitions(key)
    plans = pd.DataFrame([{"app_id": "app", "candidate": i, "parts": len(c["pieces"]), "cuts": c["cuts"],
                           "pieces": json.dumps(c["pieces"])} for i, c in enumerate(cands, 1)],
                         columns=["app_id", "candidate", "parts", "cuts", "pieces"])
    accept = model.predict(candidate_table(apps, plans, allocator)).astype(bool) if cands else np.zeros(0, bool)
    choice = int(choose(apps, plans, pd.Series([said.ai]), accept, True).choice.iloc[0])
    whole = execute(t, None, env)
    candidates = []
    for i, (c, ok) in enumerate(zip(cands, accept), 1):
        parts = allocator.decide(subtasks(t, c["pieces"], c["cuts"]))
        truth = execute(t, c, env)
        candidates.append({"candidate": i, "parts": len(c["pieces"]), "sizes": [p[0] for p in c["pieces"]],
                           "cuts": c["cuts"], "max_cuts_per_part": max(p[2] for p in c["pieces"]),
                           "part_decisions": parts.ai.value_counts().to_dict(), "accepted": bool(ok), "truth": truth,
                           "node_sets": c["parts"]})
    if choice > 0:
        final, action = candidates[choice - 1]["truth"], "PARTITION"
    else:
        action = "REDESIGN" if choice < 0 else said.ai
        ok = action != "REDESIGN" and bool(t[f"{action.lower()}_feasible"])
        final = whole if ok and action == t["oracle_label"] else {"feasible": ok, "why": whole["why"]}
    best = plan_oracle(apps, plans.assign(feasible=[c["truth"]["feasible"] for c in candidates],
                                          J=[c["truth"]["J"] for c in candidates])).iloc[0]
    return {"task": t, "workload": w, "key": key, "predicted": pred.iloc[0].to_dict(), "rules": rules, "ai": said,
            "whole": whole, "candidates": candidates, "choice": choice, "action": action, "final": final,
            "oracle": {"best": int(best.best), "label": best.label}}


def describe(r: dict) -> str:
    """plan_one's result as text."""
    t, said, rules, whole = r["task"], r["ai"], r["rules"], r["whole"]
    lines = [f"APPLICATION  {t['task_id']} on {t['backend']}  ({t['n_qubits']} qubits)",
             f"Phase 2 prediction: {r['predicted']['two_qubit_gates']:.0f} two-qubit gates "
             f"(exact transpile: {t['two_qubit_gates']})",
             f"Phase 1 rules:      {rules.decision} via {rules.fired_rule}: {rules.trace[-1]['reason']}",
             f"Phase 3 AI:         {said.ai}  " + "  ".join(f"P({a})={said[f'p_{a}']:.2f}" for a in ("CLASSICAL", "HYBRID", "REDESIGN")),
             f"  ground truth as one task: {t['oracle_label']}"
             + (f"  ({whole['why']})" if t["oracle_label"] == "REDESIGN" else ""), "",
             "Phase 4 candidates (fewest cuts first):"]
    if not r["candidates"]:
        lines.append("  none: " + ("a sampling task cannot be rebuilt from parts" if t["family"] == "random" else
                                   "the splitter finds no modules (every qubit pair is entangled)"))
    for c in r["candidates"]:
        sizes = ", ".join(map(str, sorted(c["sizes"], reverse=True)[:6])) + (", ..." if c["parts"] > 6 else "")
        overhead = "" if t["family"] == "maxcut" else f", shots x {9.0 ** c['cuts']:.3g}"
        truth = c["truth"]
        lines += [f"  {c['candidate']}. {c['parts']} parts ({sizes} qubits), {c['cuts']} cuts{overhead}",
                  f"       parts through Phases 2+3: {c['part_decisions']}   planner: {'ACCEPT' if c['accepted'] else 'reject'}",
                  f"       ground truth: {'works' if truth['feasible'] else 'fails'} ({truth['why']})"
                  + (f", certified ratio {truth['certified_ratio']:.3f}" if "certified_ratio" in truth else "")]
    lines += ["", f"DECISION: {r['action']}" + (f" (candidate {r['choice']})" if r["choice"] > 0 else "")
              + f"     plan oracle: {r['oracle']['label']}"]
    final = r["final"]
    if final.get("dag") is not None:
        lines.append(f"  schedule: {final['time_s']:.3g} s, {final['qpu_s']:.3g} QPU-s, {final['energy_j'] / 1e3:.3g} kJ, cost J {final['J']:.4g}")
        for n in dag_rows(final["dag"]):
            lines.append(f"    {n['start']:>10.1f} s -> {n['finish']:>10.1f} s  {n['lane']:<6} {n['id'].split('/')[-1]:<8} {n['label']}")
    elif r["action"] != "REDESIGN":
        lines.append(f"  the chosen plan fails in the ground truth: {final['why']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- an application of several kernels

def plan_workflow(name: str) -> dict:
    """Plan every kernel of an application from config/applications.yaml, join the kernel DAGs along the
    `after` dependencies and schedule them together. A kernel with no working plan is reported as blocked,
    and so is everything that waits for it."""
    app = yaml.safe_load((CONFIG_DIR / "applications.yaml").read_text())[name]
    env = load_environment()
    dag, kernels, blocked = nx.DiGraph(), {}, set()
    for kernel, k in app["kernels"].items():
        k = dict(k)
        after, family = k.pop("after", []), k.pop("workload")
        args = (family, k.pop("size"), k.pop("reps", 1), k.pop("entanglement", "none" if family != "vqe" else "reverse_linear"),
                k.pop("seed", 0), app["backend"], app["conditions"] | {c: k.pop(c) for c in CONDITIONS if c in k})
        r = kernels[kernel] = plan_one(*args, **k) | {"after": after}
        if r["final"].get("dag") is None or not r["final"]["feasible"] or blocked & set(after):
            blocked.add(kernel)
            continue
        sub = nx.relabel_nodes(r["final"]["dag"], lambda n: f"{kernel}/{n.split('/')[-1]}")
        dag.update(sub)
        for dep in after:  # everything `dep` ends with must finish before anything this kernel starts with
            dag.add_edges_from((a, b) for a in dag if a.startswith(f"{dep}/") and dag.out_degree(a) == 0
                               for b in sub if sub.in_degree(b) == 0)
    queue_s = app["conditions"]["queue_wait_min"] * 60
    makespan = schedule(dag, env["classical"]["cpu_nodes"], queue_s) if len(dag) else 0.0
    return {"name": name, "description": app["description"], "backend": app["backend"], "kernels": kernels,
            "blocked": sorted(blocked), "dag": dag, "deadline_s": app["conditions"]["deadline_min"] * 60,
            **totals(dag, makespan, env)}


def describe_workflow(r: dict) -> str:
    lines = [f"APPLICATION  {r['name']} on {r['backend']}: {r['description']}", ""]
    for kernel, k in r["kernels"].items():
        how = f"split into {k['candidates'][k['choice'] - 1]['parts']} parts" if k["choice"] > 0 else "as one task"
        state = "BLOCKED" if kernel in r["blocked"] else k["action"]
        lines.append(f"  {kernel:<10} {k['task']['task_id']:<32} Phase 3: {k['ai'].ai:<9} Phase 4: {state:<9} "
                     + (how if state not in ("BLOCKED", "REDESIGN") else "no plan works" if k["action"] == "REDESIGN"
                        else "waits for a blocked kernel" if k["final"].get("feasible") else k["final"]["why"]))
    lines += ["", f"  task DAG: {r['dag'].number_of_nodes()} steps, {r['dag'].number_of_edges()} dependencies"]
    for n in sorted(dag_rows(r["dag"]), key=lambda n: (n["start"], n["id"])):
        lines.append(f"    {n['start']:>9.1f} s -> {n['finish']:>9.1f} s  {n['lane']:<6} {n['id']:<20} {n['label']}")
    lines.append(f"\n  finishes in {r['time_s'] / 60:.1f} min (deadline {r['deadline_s'] / 60:g} min), {r['qpu_s']:.0f} QPU-s, "
                 f"{r['energy_j'] / 1e3:,.0f} kJ" + (f"   blocked: {', '.join(r['blocked'])}" if r["blocked"] else ""))
    return "\n".join(lines)
