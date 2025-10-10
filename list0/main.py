import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import tsplib95


def euc_2d(x, y):
    (x1, y1), (x2, y2) = x, y
    return int(math.hypot(x1 - x2, y1 - y2) + 0.5)


def path_cost(g, path, cyclic=True):
    res = nx.path_weight(g, path, weight="weight")
    if cyclic:
        res += g[path[-1]][path[0]]["weight"]
    return res


def path_edges(path, cyclic=True):
    return list(nx.utils.pairwise(path, cyclic=cyclic))


def random_tour(g, start=1):
    nodes = list(g.nodes)
    nodes.remove(start)
    random.shuffle(nodes)

    return [start] + nodes


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


def draw(g, paths):
    plt.figure(figsize=(15, 15))
    pos = {v: g.nodes[v]["coord"] for v in g.nodes}

    nx.draw_networkx_nodes(g, pos, node_size=60)

    for path in paths:
        nx.draw_networkx_edges(g, pos, edgelist=path_edges(path, cyclic=True), width=3)

    plt.show()


def gen_random(g, opt_cost, n=100):
    res = dict()

    for _ in range(n):
        tour = random_tour(g)
        cost = path_cost(g, path=tour, cyclic=True)
        ratio = cost / opt_cost
        # print(f"Random tour {i}: {cost},  ratio: {ratio}%")
        over = 100 * (ratio - 1)

        for x in range(100, 1000, 100):
            if x - 100 <= over < x:
                res[x] = res.get(x, 0) + 1

    print(res)


if __name__ == "__main__":
    DATASET = "berlin52"
    OPT_PATH_LENGTH = 7542

    print(f"TSP for {DATASET}\nLength of the optimal cycle: {OPT_PATH_LENGTH}\n")

    g, opt_path = read_graph(dataset=DATASET)

    tsp = nx.approximation.traveling_salesman_problem(g)
    print(path_cost(g, tsp, cyclic=True))

    tour1 = random_tour(g)
    # draw(g, [opt_path])

    gen_random(g=g, opt_cost=OPT_PATH_LENGTH, n=1000_00)
