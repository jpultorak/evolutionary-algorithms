import math
from collections import Counter

import numpy as np

from list0.algorithms.hill_climb import hill_climb
from list0.algorithms.hill_climb2 import hill_climb2
from list0.utils import tour_cost


def generate(dist, generate_path, opt_cost, n=100):
    bins = Counter()

    worst, best = -math.inf, math.inf
    for _ in range(n):
        path = generate_path(dist)
        cost = tour_cost(dist=dist, path=path)

        best = min(best, cost)
        worst = max(worst, cost)
        over = 100.0 * (cost / opt_cost - 1.0)

        bucket = 10 * math.floor(over / 10.0)
        bins[bucket] += 1

    return sorted(bins.items()), best, worst


def generate_hc(dist: np.ndarray, opt_cost, m=2, first_choice=False, n=100):
    """
    Do n runs of hill climb algorithm and return results sorted in bins, best cost,
     worst cost, and total iteration
    """
    bins = Counter()

    worst, best = -math.inf, math.inf
    total_iterations = []
    final_costs = []
    for _ in range(n):
        if m == 2:
            _, costs = hill_climb2(dist=dist, first_choice=first_choice)
        else:
            _, costs = hill_climb(dist, m, first_choice=first_choice)
        cost = costs[-1]
        total_iterations.append(len(costs))
        final_costs.append(cost)

        best = min(best, cost)
        worst = max(worst, cost)
        over = 100.0 * (cost / opt_cost - 1.0)

        bucket = 10 * math.floor(over / 10.0)
        bins[bucket] += 1

    return sorted(bins.items()), best, worst, total_iterations, final_costs
