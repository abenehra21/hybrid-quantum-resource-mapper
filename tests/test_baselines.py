import csv

import networkx as nx
import numpy as np

from hybrid_mapper.baselines import COLUMNS, brute_force_maxcut, open_log, spsa


def test_brute_force_maxcut():
    assert brute_force_maxcut(nx.complete_graph(3)) == 2
    assert brute_force_maxcut(nx.cycle_graph(6)) == 6
    assert brute_force_maxcut(nx.petersen_graph()) == 12


def test_spsa_minimises_quadratic_at_any_scale():
    for scale in (1e-2, 1.0, 1e3):  # calibration makes the step size independent of f's scale
        f = lambda xs: scale * np.sum((xs - 1.0) ** 2, axis=1)
        x = spsa(f, np.zeros(4), 200, np.random.default_rng(0))
        assert np.allclose(x, 1.0, atol=0.1)


def test_log_with_old_columns_is_moved_aside_not_overwritten(tmp_path):
    log = tmp_path / "decisions.csv"
    log.write_text("old,header\n1,2\n")
    f, writer = open_log(log)
    f.close()
    moved = list(tmp_path.glob("decisions.*.old.csv"))
    assert len(moved) == 1 and moved[0].read_text() == "old,header\n1,2\n"
    assert next(csv.reader(log.open())) == COLUMNS
