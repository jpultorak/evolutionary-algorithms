import itertools
import random
from functools import cache

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


def best_neighbour_hamming(p, cost_fn, m, first_choice=False):
    """
    Returns best permutation q with hamming distance to p <= m
    """
    n = len(p)

    # For the current best permutation, generate all permutations with hamming distance
    # less than m
    best_q = p.copy()
    best_cost = cost_fn(best_q)

    found_better = False
    for k in range(2, m + 1):
        for subset in itertools.combinations(range(n), k):
            for subset_perm in derangements(k):
                q = p.copy()
                for i in range(k):
                    q[subset[subset_perm[i]]] = p[subset[i]]

                cost_q = cost_fn(q)
                if cost_q < best_cost:
                    best_cost, best_q = cost_q, q
                    found_better = True
                    if first_choice:
                        return found_better, best_q, best_cost

    return found_better, best_q, best_cost


def hill_climb(g, m, first_choice=False):
    if m < 2 or m >= g.number_of_nodes():
        raise ValueError("Must be 2 <= m <n")

    def cost_fn(p):
        return tour_cost(g, p, cyclic=True)

    costs = []

    p = list(g.nodes)
    random.shuffle(p)

    cost = cost_fn(p)
    costs.append(cost)

    while True:
        found_better, best_q, best_cost = best_neighbour_hamming(
            p, cost_fn=cost_fn, m=m, first_choice=first_choice
        )
        if not found_better:
            return p, costs
        costs.append(best_cost)
        p, cost = best_q, best_cost


if __name__ == "__main__":
    pass
