import numpy as np
import networkx as nx
from src.graph_utils import generate_erdos_renyi, greedy_maxcut, gw_style_maxcut, evaluate_cut


def test_greedy_and_gw_return_valid_cuts():
    G = generate_erdos_renyi(8, 0.5, seed=123)
    bs_g, val_g = greedy_maxcut(G, seed=123, restarts=10)
    assert isinstance(bs_g, list) and len(bs_g) == G.number_of_nodes()
    assert val_g >= 0

    bs_gw, val_gw = gw_style_maxcut(G, repeats=50, seed=123)
    assert isinstance(bs_gw, list) and len(bs_gw) == G.number_of_nodes()
    assert val_gw >= 0

    # both should be <= number of edges
    assert val_g <= G.number_of_edges()
    assert val_gw <= G.number_of_edges()


def test_gw_often_better_than_random():
    G = generate_erdos_renyi(8, 0.5, seed=10)
    bs_gw, val_gw = gw_style_maxcut(G, repeats=200, seed=10)
    # sample random cuts and check gw is not worse than avg random cut
    rng = np.random.RandomState(10)
    random_vals = []
    for _ in range(200):
        bs = rng.randint(0, 2, size=G.number_of_nodes()).tolist()
        random_vals.append(evaluate_cut(G, bs))
    assert val_gw >= np.mean(random_vals) - 1e-6
