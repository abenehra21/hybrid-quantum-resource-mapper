import networkx as nx

from hybrid_mapper.baselines import brute_force_maxcut


def test_brute_force_maxcut():
    assert brute_force_maxcut(nx.complete_graph(3)) == 2
    assert brute_force_maxcut(nx.cycle_graph(6)) == 6
    assert brute_force_maxcut(nx.petersen_graph()) == 12
