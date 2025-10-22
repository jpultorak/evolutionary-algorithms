import numpy as np


def random_path(dist: np.ndarray):
    n = len(dist)
    rng = np.random.default_rng()
    return rng.permutation(n)


def weighted_random_path(dist: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng()

    n = len(dist)
    nodes = np.arange(n)
    tour = np.empty(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool)

    v = rng.integers(low=0, high=n)
    visited[v] = True
    tour[0] = v
    total = 1

    while total < n:
        aval = nodes[~visited]
        p = 1.0 / dist[v, aval]
        p = p / np.sum(p, dtype=float)
        v = rng.choice(nodes[aval], p=p)

        tour[total] = v
        visited[v] = True
        total += 1

    return tour
