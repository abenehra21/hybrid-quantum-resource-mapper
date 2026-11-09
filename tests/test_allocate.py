from hybrid_mapper.allocate import evaluate_split, load_tasks, splits


def test_learned_allocator_beats_fixed_rules_on_unseen_workloads():
    train, test = next(iter(splits(load_tasks()).values()))
    m = evaluate_split(train, test)["metrics"]
    ai, rules = m["AI: gradient boosting"], m["Rules (exact transpile)"]
    assert ai["Decision accuracy (vs oracle)"] > rules["Decision accuracy (vs oracle)"]
    assert ai["Mean regret (cost units)"] < rules["Mean regret (cost units)"]
