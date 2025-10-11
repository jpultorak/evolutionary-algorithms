import math
from collections import Counter

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
