import numpy as np

from list0.algorithms.random_tours import random_path
from list0.utils import tour_cost


def _prev(i: int, n: int) -> int:
    if i == 0:
        return n - 1
    return i - 1


def _next(i: int, n: int) -> int:
    if i == n - 1:
        return 0
    return i + 1


def _swap(dist: np.ndarray, path: np.ndarray, i: int, j: int):
    # b -> a -> c .... e -> d -> f

    n = len(path)
    if i == j:
        return 0

    a = path[i]
    b = path[_prev(i, n)]
    c = path[_next(i, n)]

    d = path[j]
    e = path[_prev(j, n)]
    f = path[_next(j, n)]

    if _next(i, n) == j:
        remove = dist[b, a] + dist[d, f]
        add = dist[b, d] + dist[a, f]
        return int(add) - int(remove)

    if _next(j, n) == i:
        remove = dist[e, d] + dist[a, c]
        add = dist[e, a] + dist[d, c]
        return int(add) - int(remove)

    remove = dist[b, a] + dist[a, c] + dist[e, d] + dist[d, f]
    add = dist[b, d] + dist[d, c] + dist[e, a] + dist[a, f]
    return int(add) - int(remove)


def hamming_2(dist: np.ndarray, path: np.ndarray, first_choice=False):
    n = len(path)
    best_delta, best_swap = 0, (-1, -1)

    for i in range(n - 1):
        for j in range(i + 1, n):
            delta = _swap(dist, path, i, j)
            if delta < best_delta:
                best_swap = (i, j)
                best_delta = delta
                if first_choice:
                    return best_delta, best_swap

    delta = _swap(dist, path, n - 1, 0)
    if delta < best_delta:
        best_delta, best_swap = delta, (n - 1, 0)

    return best_delta, best_swap


def hill_climb2(dist: np.ndarray, first_choice=False):
    costs = []
    path = random_path(dist)
    costs.append(tour_cost(dist, path))

    while True:
        best_delta, (i, j) = hamming_2(dist, path, first_choice=first_choice)
        if best_delta == 0:
            return path, costs
        path[[i, j]] = path[[j, i]]
        costs.append(costs[-1] + best_delta)


if __name__ == "__main__":
    pass
