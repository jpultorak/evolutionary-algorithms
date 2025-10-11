import math
from collections import Counter

from list0.algorithms.hill_climb import hill_climb
from list0.utils import path_cost


def generate(g, generate_tour, opt_cost, n=100):
    bins = Counter()

    worst, best = -math.inf, math.inf
    for _ in range(n):
        tour = generate_tour(g)
        cost = path_cost(g, path=tour, cyclic=True)

        best = min(best, cost)
        worst = max(worst, cost)
        over = 100.0 * (cost / opt_cost - 1.0)

        bucket = 10 * math.floor(over / 10.0)
        bins[bucket] += 1

    return sorted(bins.items()), best, worst


def generate_hc(g, opt_cost, m=2, first_choice=False, n=100):
    """
    Do n runs of hill climb algorithm and return results sorted in bins, best cost,
     worst cost, and total iteration
    """
    bins = Counter()

    worst, best = -math.inf, math.inf
    total_iterations = []
    for _ in range(n):
        _, costs = hill_climb(g, m=m, first_choice=first_choice)
        cost = costs[-1]
        total_iterations.append(len(costs))

        best = min(best, cost)
        worst = max(worst, cost)
        over = 100.0 * (cost / opt_cost - 1.0)

        bucket = 10 * math.floor(over / 10.0)
        bins[bucket] += 1

    return sorted(bins.items()), best, worst, total_iterations
