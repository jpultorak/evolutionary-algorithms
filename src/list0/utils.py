import math

import networkx as nx


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
