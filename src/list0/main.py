from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import tsplib95

from list0.algorithms.random_tours import random_tour, weighted_random_tour
from list0.eval import gen_random
from list0.utils import euc_2d, path_edges

DATASETS = Path(__file__).resolve().parents[2] / "datasets"


def read_graph(dataset):
    tsp_path = DATASETS / f"{dataset}.tsp"
    opt_path = DATASETS / f"{dataset}.opt.tour"

    problem = tsplib95.load(str(tsp_path))
    problem_opt = tsplib95.load(str(opt_path))

    assert problem.type == "TSP"
    assert problem.edge_weight_type == "EUC_2D"

    assert problem_opt.type == "TOUR"
    assert len(problem_opt.tours) == 1

    assert problem.get_weight(1, 2) == euc_2d(
        problem.node_coords[1], problem.node_coords[2]
    )

    g = problem.get_graph()
    opt_path = problem_opt.tours[0]
    assert problem.get_weight(2, 1) == g[1][2]["weight"]

    return g, opt_path


def draw(g, paths):
    plt.figure(figsize=(15, 15))
    pos = {v: g.nodes[v]["coord"] for v in g.nodes}

    nx.draw_networkx_nodes(g, pos, node_size=60)

    for path in paths:
        nx.draw_networkx_edges(g, pos, edgelist=path_edges(path, cyclic=True), width=3)

    plt.show()


if __name__ == "__main__":
    DATASET = "berlin52"
    OPT_PATH_LENGTH = 7542

    print(f"TSP for {DATASET}\nLength of the optimal cycle: {OPT_PATH_LENGTH}\n")
    g, opt_path = read_graph(dataset=DATASET)

    # EXERCISE 1
    # tour1 = random_tour(g)
    # draw(g, [opt_path])

    # EXERCISE 2

    res0 = gen_random(g=g, generate_tour=random_tour, opt_cost=OPT_PATH_LENGTH, n=1000)

    res1 = gen_random(
        g=g, generate_tour=weighted_random_tour, opt_cost=OPT_PATH_LENGTH, n=1000
    )

    print(res0)
    print(res1)
