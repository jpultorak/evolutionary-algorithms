import itertools
from functools import cache

import numpy as np

from list0.algorithms.random_tours import random_path
from list0.utils import tour_cost


@cache
def derangements(k: int):
    """
    Returns and caches all derangements of size k
    """
    return tuple(
        perm
        for perm in itertools.permutations(range(k))
        if all(perm[i] != i for i in range(k))
    )


def best_neighbour_hamming(path: np.ndarray, cost_fn, m, first_choice=False):
    """
    Returns best permutation q with hamming distance to p <= m
    """
    n = len(path)

    # For the current best permutation, generate all permutations with hamming distance
    # less than m
    best_q = path.copy()
    best_cost = cost_fn(best_q)

    found_better = False
    for k in range(2, m + 1):
        for subset in itertools.combinations(range(n), k):
            for subset_perm in derangements(k):
                q = path.copy()
                for i in range(k):
                    q[subset[subset_perm[i]]] = path[subset[i]]

                cost_q = cost_fn(q)
                if cost_q < best_cost:
                    best_cost, best_q = cost_q, q
                    found_better = True
                    if first_choice:
                        return found_better, best_q, best_cost

    return found_better, best_q, best_cost


def hill_climb(dist: np.ndarray, m, first_choice=False):
    n = len(dist)
    if m < 2 or m >= n:
        raise ValueError("Must be 2 <= m <n")

    def cost_fn(path: np.ndarray):
        return tour_cost(dist, path)

    costs = []

    path = random_path(dist)

    while True:
        found_better, best_q, best_cost = best_neighbour_hamming(
            path, cost_fn=cost_fn, m=m, first_choice=first_choice
        )
        if not found_better:
            return path, costs
        costs.append(best_cost)
        path, _ = best_q, best_cost


if __name__ == "__main__":
    pass
