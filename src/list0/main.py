from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

from list0.algorithms.hill_climb2 import hill_climb2
from list0.algorithms.random_tours import random_path, weighted_random_path
from list0.eval import generate, generate_hc
from list0.read_datasets import read_dataset
from list0.visualization import draw_coords_tour

DATASET = "berlin52"
OPT_PATH_LENGTH = 7542
HC_N, HC_M, HC_FIRST_CHOICE = 100, 2, False
RANDOM_N = 1000


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
    bins, best, worst, iterations, costs = generate_hc(
        dist=dist,
        m=HC_M,
        first_choice=HC_FIRST_CHOICE,
        opt_cost=OPT_PATH_LENGTH,
        n=HC_N,
    )
    hc_end = timer()

    print(f"Hill climb N={HC_N}, M={HC_M}, first_choice={HC_FIRST_CHOICE}")
    print(f"Execution time: {hc_end - hc_start}")
    print(bins)
    print(f"Best: {best}\nWorst:{worst}")
    print()

    plot_bins(bins, f"Hill climb N={HC_N}, M={HC_M}, first_choice={HC_FIRST_CHOICE}")
    plot_iters_vs_cost(
        iterations=iterations,
        costs=costs,
        title="Cost vs iterations; "
        + f"Hill climb N={HC_N}, M={HC_M}, first_choice={HC_FIRST_CHOICE}",
    )
    plot_iterations(
        iterations,
        title=f"Iterations for hill climb N={HC_N},"
        f" M={HC_M}, first_choice={HC_FIRST_CHOICE}",
    )


def hc_cost_for_each_iteration():
    _, costs = hill_climb2(dist=dist, first_choice=HC_FIRST_CHOICE)
    plot_costs(
        costs,
        title=f"Hill climb cost N={HC_N}, M={HC_M}, first_choice={HC_FIRST_CHOICE}",
    )


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


def plot_iterations(iterations, title):
    x = np.arange(1, len(iterations) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, iterations, marker=".")

    plt.title(title)
    plt.xlabel("run #")
    plt.ylabel("iterations")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()


def plot_costs(costs, title):
    x = np.arange(len(costs))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, costs, marker=".")

    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.show()


def plot_iters_vs_cost(iterations, costs, title):
    iters = np.asarray(iterations, dtype=float)
    costs = np.asarray(costs, dtype=float)

    plt.figure(figsize=(10, 5))
    plt.scatter(iters, costs, s=5)
    plt.title(title)
    plt.xlabel("iterations to stop")
    plt.ylabel("final tour cost")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()


if __name__ == "__main__":
    print(f"TSP for {DATASET}\nLength of the optimal cycle: {OPT_PATH_LENGTH}\n")
    nodes, coords, dist, opt_path = read_dataset(DATASET)

    # draw_tours()

    # random()
    # weighted_random()

    hc()

    hc_cost_for_each_iteration()
