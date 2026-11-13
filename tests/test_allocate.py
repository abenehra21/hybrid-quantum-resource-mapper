from hybrid_mapper.allocate import evaluate_split, load_tasks, splits


def test_learned_allocator_beats_fixed_rules_on_unseen_workloads():
    train, test = next(iter(splits(load_tasks()).values()))
    m = evaluate_split(train, test)["metrics"]
    ai, rules = m["AI: gradient boosting"], m["Rules (exact transpile)"]
    assert ai["Decision accuracy (vs oracle)"] > rules["Decision accuracy (vs oracle)"]
    assert ai["Mean regret (cost units)"] < rules["Mean regret (cost units)"]
    r3 = m["Rules + R3 entanglement check"]  # the entanglement feature helps the rules, but learning still wins
    assert rules["Mean regret (cost units)"] > r3["Mean regret (cost units)"] > ai["Mean regret (cost units)"]


def test_range_guard_hands_over_to_the_rules_beyond_the_trained_sizes():
    from hybrid_mapper.allocate import Allocator, trusted_qubits

    tasks = load_tasks()
    small, large = tasks[tasks.n_qubits <= 40], tasks[tasks.n_qubits > 40]
    allocator = Allocator(small)
    assert allocator.max_qubits == trusted_qubits(small) == 40
    beyond = allocator.decide(large.sample(40, random_state=0))
    assert beyond.guarded.all() and (beyond.ai == beyond.rules).all()
    assert not allocator.decide(small.sample(40, random_state=0)).guarded.any()
    assert trusted_qubits(tasks) == 120  # trained on every size: the guard is off
