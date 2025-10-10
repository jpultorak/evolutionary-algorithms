import random


def random_tour(g, start=1):
    nodes = list(g.nodes)
    nodes.remove(start)
    random.shuffle(nodes)

    return [start] + nodes


def weighted_random_tour(g, start=1):
    nodes = list(g.nodes)
    tour = [start]

    visited = {start}

    v = start
    while len(visited) < len(nodes):
        aval_neighbours = [u for u in g.neighbors(v) if u not in visited]

        if not aval_neighbours:
            raise RuntimeError("Incomplete graph")

        weights = [1 / g[v][u]["weight"] for u in aval_neighbours]
        v = random.choices(aval_neighbours, weights=weights)[0]

        tour.append(v)
        visited.add(v)

    return tour
