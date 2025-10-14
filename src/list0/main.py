from timeit import default_timer as timer

from list0.algorithms.random_tours import random_path, weighted_random_path
from list0.eval import generate, generate_hc
from list0.read_datasets import read_dataset
from list0.visualization import draw_coords_tour

DATASET = "berlin52"
OPT_PATH_LENGTH = 7542
HC_N, HC_M = 1, 2


def random():
    bins, best, worst = generate(
        dist=dist, generate_path=random_path, opt_cost=OPT_PATH_LENGTH, n=100
    )
    print(bins)
    print(f"Best: {best}\nWorst:{worst}")
    print()


def weighted_random():
    bins, best, worst = generate(
        dist=dist, generate_path=weighted_random_path, opt_cost=OPT_PATH_LENGTH, n=100
    )
    print(bins)
    print(f"Best: {best}\nWorst:{worst}")
    print()


def hc():
    hc_start = timer()
    bins, best, worst, iterations = generate_hc(
        dist=dist, m=HC_M, first_choice=False, opt_cost=OPT_PATH_LENGTH, n=HC_N
    )
    hc_end = timer()

    print(f"Hill climb N={HC_N}, M={HC_M} execution time {hc_end - hc_start}")
    print(f"Iterations {iterations}")
    print(bins)
    print(f"Best: {best}\nWorst:{worst}")
    print()


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
    #
    # hc_tour = hill_climb(g, m=HC_M, first_choice=False)[0]
    # draw(g, [hc_tour])


if __name__ == "__main__":
    print(f"TSP for {DATASET}\nLength of the optimal cycle: {OPT_PATH_LENGTH}\n")
    nodes, coords, dist, opt_path = read_dataset(DATASET)

    # draw_tours()

    # random()
    # weighted_random()

    hc()

    # hc_iterations()
