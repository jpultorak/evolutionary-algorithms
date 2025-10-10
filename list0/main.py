import math

import networkx as nx
import tsplib95


def euc_2d(x, y):
    (x1, y1), (x2, y2) = x, y
    return int(math.hypot(x1 - x2, y1 - y2) + 0.5)


def get_tour_cost(g, path):
    return nx.path_weight(g, path, weight="weight") + g[path[-1]][path[0]]["weight"]


def read_graph(dataset):
    problem = tsplib95.load(f"../datasets/{dataset}.tsp")
    problem_opt = tsplib95.load(f"../datasets/{dataset}.opt.tour")

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


if __name__ == "__main__":
    DATASET = "berlin52"
    OPT_PATH_LENGTH = 7542

    print(f"TSP for {DATASET}\nLength of the optimal cycle: {OPT_PATH_LENGTH}\n")

    g, opt_path = read_graph(dataset=DATASET)

    tsp = nx.approximation.traveling_salesman_problem(g)
    print(get_tour_cost(g, tsp))
