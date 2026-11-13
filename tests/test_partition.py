import json

import networkx as nx
import numpy as np
import pytest
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector, random_density_matrix

from hybrid_mapper import partition as pt
from hybrid_mapper.environment import load_environment
from hybrid_mapper.workloads import clustered, synthetic_hamiltonian, vqe

CFG = load_environment()["partition"]
BASE = {"precision_target": 5e-2, "maxcut_ratio_target": 0.9, "deadline_min": 480, "queue_wait_min": 30,
        "shot_budget": 1e7, "samples_needed": 1e5}


# ---- gate cutting: the decomposition the cost model rests on

def test_six_local_circuits_rebuild_a_cz_gate_with_gamma_3():
    assert len(pt.QPD) == CFG["gate_cut_circuits"]
    assert sum(abs(w) for w, _, _ in pt.QPD) == CFG["gate_cut_gamma"]
    z = np.diag([1, -1]).astype(complex)
    local = {"I": np.eye(2), "Z": z, "R+": np.diag(np.exp(1j * np.pi / 4 * np.array([1, -1]))),
             "R-": np.diag(np.exp(-1j * np.pi / 4 * np.array([1, -1])))}
    phase = np.diag(np.exp(-1j * np.pi / 4 * np.array([1, -1])))  # Rz(pi/2): each qubit's local part of the CZ
    rho = random_density_matrix(4, seed=5).data
    rebuilt = np.zeros((4, 4), dtype=complex)
    for weight, a, b in pt.QPD:
        for wa, ket_a, bra_a in pt.KET_BRA.get(a, [(1.0, a, a)]):
            for wb, ket_b, bra_b in pt.KET_BRA.get(b, [(1.0, b, b)]):
                ket = np.kron(local[ket_a] @ phase, local[ket_b] @ phase)
                bra = np.kron(local[bra_a] @ phase, local[bra_b] @ phase)
                rebuilt += weight * wa * wb * ket @ rho @ bra.conj().T
    cz = np.diag([1, 1, 1, -1]).astype(complex)
    assert np.allclose(rebuilt, cz @ rho @ cz)


@pytest.mark.parametrize("blocks,size,reps,links", [(2, 3, 1, 1), (2, 3, 2, 1), (3, 2, 1, 1), (2, 4, 1, 2)])
def test_cut_circuit_gives_the_uncut_expectation_value(blocks, size, reps, links):
    w = clustered(blocks, size, reps, links, seed=1)
    circuit = w.circuit.assign_parameters(np.random.default_rng(0).uniform(-np.pi, np.pi, w.circuit.num_parameters))
    parts = pt.split(pt.interaction_graph(circuit))
    assert [len(p) for p in parts] == [size] * blocks  # the splitter finds the blocks
    assert pt.fragments(circuit, parts)[1] == (blocks - 1) * links * reps
    exact = Statevector(circuit).expectation_value(w.hamiltonian).real
    assert pt.cut_expectation(circuit, w.hamiltonian, parts) == pytest.approx(exact, abs=1e-10)


def test_cutting_is_exact_for_any_circuit_and_any_split():
    circuit = random_circuit(5, 2, max_operands=2, seed=11)
    ham = synthetic_hamiltonian(5, seed=2)
    parts = [[0, 3], [1, 2, 4]]  # deliberately not what the splitter would choose
    assert 1 <= pt.fragments(circuit, parts)[1] <= 4
    exact = Statevector(circuit).expectation_value(ham).real
    assert pt.cut_expectation(circuit, ham, parts) == pytest.approx(exact, abs=1e-10)


def test_fully_entangled_circuit_has_no_partition_and_a_chain_has_one():
    assert pt.partitions(("vqe", "vqe", 20, 1, "full", 0, 1, 1)) == ()
    assert pt.partitions(("random", "random", 20, 5, "none", 0, 1, 1)) == ()  # sampling: not partitioned
    chain = pt.partitions(("vqe", "vqe", 40, 2, "reverse_linear", 0, 1, 1))
    assert chain and chain[0]["cuts"] == 2 * (len(chain[0]["pieces"]) - 1)  # two layers cross every boundary


# ---- Max-Cut: divide, solve exactly, merge

def test_exact_maxcut():
    assert pt.exact_maxcut(nx.petersen_graph())[0] == 12
    signed = nx.Graph()
    signed.add_weighted_edges_from([(0, 1, 2), (1, 2, -3), (0, 2, 1)])
    assert pt.exact_maxcut(signed)[0] == 3  # cut {0,1} and {0,2}, keep the negative edge uncut


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_maxcut_certificate_brackets_the_true_optimum(seed):
    graph = nx.random_regular_graph(3, 22, seed=seed)
    solution = pt.solve_maxcut(graph, pt.split(graph, cap=8))
    optimum = pt.exact_maxcut(graph)[0]
    assert solution["cut"] == sum(solution["side"][u] != solution["side"][v] for u, v in graph.edges)
    assert solution["cut"] <= optimum <= solution["upper_bound"]
    assert solution["certified_ratio"] <= solution["cut"] / optimum


def test_split_respects_the_part_size_cap():
    graph = nx.random_regular_graph(3, 120, seed=0)
    parts = pt.split(graph, cap=CFG["maxcut_part_nodes"])
    assert max(map(len, parts)) <= CFG["maxcut_part_nodes"]
    assert sorted(v for p in parts for v in p) == list(range(120))


# ---- task DAG and schedule

def test_schedule_respects_dependencies_and_resource_capacity():
    steps = [("classical", 5.0, "a"), ("quantum", 7.0, "b"), ("classical", 3.0, "c"), ("quantum", 2.0, "d"),
             ("classical", 4.0, "e")]
    dag = pt.task_dag("t", steps, merge_s=1.0)
    makespan = pt.schedule(dag, cpu_nodes=2, queue_s=10.0)
    nodes = dict(dag.nodes(data=True))
    assert all(nodes[b]["start"] >= nodes[a]["finish"] for a, b in dag.edges)
    assert all(d["start"] >= 10.0 for d in nodes.values() if d["kind"] == "quantum")  # the QPU queue
    for lane in {d["lane"] for d in nodes.values()}:
        spans = sorted((d["start"], d["finish"]) for d in nodes.values() if d["lane"] == lane)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))  # one job at a time per resource
    assert makespan == nodes["t/merge"]["finish"] == 10.0 + 7.0 + 2.0 + 1.0
    assert sum(d["lane"].startswith("CPU") for d in nodes.values()) == 5 and len({d["lane"] for d in nodes.values()}) == 3


# ---- plans against the ground truth

@pytest.fixture(scope="module")
def data():
    return pt.load_apps(), pt.load_plans()


def test_stored_plans_match_a_fresh_execution(data):
    apps, plans = data
    by_id = apps.set_index("app_id").to_dict("index")
    for p in plans.groupby("cuts").head(1).sample(8, random_state=0).itertuples():  # a spread of plan sizes
        t = by_id[p.app_id] | {"app_id": p.app_id}
        candidate = pt.partitions(pt.spec(t))[p.candidate - 1]
        assert candidate["pieces"] == json.loads(p.pieces)
        truth = pt.execute(t, candidate)
        assert truth["feasible"] == p.feasible and truth["J"] == pytest.approx(p.J)


def test_partitioning_never_hurts_a_task_that_already_works(data):
    apps, plans = data
    best = pt.plan_oracle(apps, plans)
    whole = np.select([apps.oracle_label == "CLASSICAL", apps.oracle_label == "HYBRID"], [apps.classical_J, apps.hybrid_J], np.inf)
    assert (best.J.to_numpy() <= np.where(np.isfinite(whole), whole, np.inf)).all()
    assert not ((best.label == "REDESIGN").to_numpy() & (apps.oracle_label != "REDESIGN").to_numpy()).any()
    chains = apps.entanglement.isin(["reverse_linear", "circular"]).to_numpy()
    assert (best.label[chains] == "CLASSICAL").all()  # already easy (MPS): cutting only adds overhead


def test_deadline_decides_whether_cutting_rescues_a_modular_circuit():
    slow = pt.plan_one("clustered", 30, 1, "full", 0, "heron_r3", BASE | {"deadline_min": 30}, blocks=3, links=1)
    fast = pt.plan_one("clustered", 30, 1, "full", 0, "heron_r3", BASE | {"deadline_min": 1440}, blocks=3, links=1)
    assert slow["task"]["oracle_label"] == fast["task"]["oracle_label"] == "REDESIGN"
    assert not slow["candidates"][0]["truth"]["feasible"] and fast["candidates"][0]["truth"]["feasible"]
    assert (slow["action"], fast["action"]) == ("REDESIGN", "PARTITION")
    assert fast["final"]["qpu_s"] == 0 and nx.is_directed_acyclic_graph(fast["final"]["dag"])


def test_learned_planner_rescues_redesign_tasks_on_unseen_workloads(data):
    apps, plans = data
    train, test = next(iter(pt.splits(apps).values()))
    assert not set(train.workload_id) & set(test.workload_id)
    m = pt.evaluate_split(train, test, plans)["metrics"]
    phase3, learned, oracle = m["Phase 3 AI, no partitioning"], m["Phase 4: learned planner"], m["Plan oracle"]
    assert phase3["REDESIGN tasks rescued"] == 0 < learned["REDESIGN tasks rescued"] <= oracle["REDESIGN tasks rescued"]
    assert learned["Tasks that get a valid result"] > phase3["Tasks that get a valid result"]
    assert learned["Mean regret (cost units)"] < phase3["Mean regret (cost units)"]
    assert learned["Mean regret (cost units)"] < m["Phase 4: partition whenever stuck"]["Mean regret (cost units)"]
    assert oracle["Mean regret (cost units)"] == 0


def test_workflow_joins_kernels_into_one_dag_and_blocks_what_cannot_run():
    ok = pt.plan_workflow("screening_pipeline")
    assert not ok["blocked"] and nx.is_directed_acyclic_graph(ok["dag"])
    kinds = {d["kind"] for _, d in ok["dag"].nodes(data=True)}
    assert kinds == {"classical", "quantum"}
    nodes = dict(ok["dag"].nodes(data=True))
    assert nodes["energy/split"]["start"] >= nodes["placement/merge"]["finish"]  # `after: [placement]`
    assert nodes["refine/run"]["start"] >= max(nodes["energy/merge"]["finish"], nodes["sampling/post"]["finish"])
    assert pt.plan_workflow("tight_deadline")["blocked"] == ["energy"]
