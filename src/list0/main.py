from timeit import default_timer as timer

import matplotlib.pyplot as plt

from list0.algorithms.random_tours import random_path, weighted_random_path
from list0.eval import generate, generate_hc
from list0.read_datasets import read_dataset
from list0.visualization import draw_coords_tour

DATASET = "berlin52"
OPT_PATH_LENGTH = 7542
HC_N, HC_M, HC_FIRST_CHOICE = 100, 2, False
RANDOM_N = 100


def random():
    start = timer()
    bins, best, worst = generate(
        dist=dist, generate_path=random_path, opt_cost=OPT_PATH_LENGTH, n=RANDOM_N
    )
    end = timer()

    print(f"Uniform random tours, N={RANDOM_N}")
    print(f"Execution time: {end - start}")
    print(bins)
    print(f"Best: {best}\nWorst:{worst}")
    print()

    plot_bins(bins, f"Uniform random tours (N={RANDOM_N})")


def weighted_random():
    start = timer()
    bins, best, worst = generate(
        dist=dist,
        generate_path=weighted_random_path,
        opt_cost=OPT_PATH_LENGTH,
        n=RANDOM_N,
    )
    end = timer()

    print(f"Weighted random tours, N={RANDOM_N}")
    print(f"Execution time: {end - start}")
    print(bins)
    print(f"Best: {best}\nWorst:{worst}")
    print()

    plot_bins(bins, f"Weighted random tours (N={RANDOM_N})")


def hc():
    hc_start = timer()
    bins, best, worst, iterations = generate_hc(
        dist=dist,
        m=HC_M,
        first_choice=HC_FIRST_CHOICE,
        opt_cost=OPT_PATH_LENGTH,
        n=HC_N,
    )
    hc_end = timer()

    print(f"Hill climb N={HC_N}, M={HC_M}, first_choice={HC_FIRST_CHOICE})")
    print(f"Execution time: {hc_end - hc_start}")
    print(f"Iterations {iterations}")
    print(bins)
    print(f"Best: {best}\nWorst:{worst}")
    print()

    plot_bins(bins, f"Hill climb N={HC_N}, M={HC_M}, first_choice={HC_FIRST_CHOICE})")


# def hc_iterations():
#     """
#     How does the cost change in consecutive iterations?
#     """
#     _, costs = hill_climb(g, m=HC_M, first_choice=False)
#     print(f"COST | DIFF\n0: {costs[0]}, 0")
#     for i in range(1, len(costs)):
#         print(f"{i}: {costs[i]}, {costs[i - 1] - costs[i]}")


def draw_tours():
    draw_coords_tour(coords, [opt_path])

    random_tour = weighted_random_path(
        dist=dist,
    )
    draw_coords_tour(coords, [random_tour])


def plot_bins(bins, title):
    xs = [x for x, _ in bins]
    ys = [y for _, y in bins]

    plt.figure(figsize=(10, 5))
    plt.bar(xs, ys, width=2)
    plt.title(title)
    plt.xlabel("% over optimal (rounded to 10%)")
    plt.ylabel("count")
    plt.xticks(xs)
    plt.show()


if __name__ == "__main__":
    print(f"TSP for {DATASET}\nLength of the optimal cycle: {OPT_PATH_LENGTH}\n")
    nodes, coords, dist, opt_path = read_dataset(DATASET)

    # draw_tours()

    random()
    weighted_random()

    hc()

    # hc_iterations()
