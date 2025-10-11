from timeit import default_timer as timer

from list0.algorithms.hill_climb import hill_climb
from list0.algorithms.random_tours import random_tour, weighted_random_tour
from list0.eval import generate, generate_hc
from list0.read_datasets import read_graph
from list0.visualization import draw

DATASET = "berlin52"
OPT_PATH_LENGTH = 7542
HC_N, HC_M = 10, 2


def random():
    bins0, best0, worst0 = generate(
        g=g, generate_tour=random_tour, opt_cost=OPT_PATH_LENGTH, n=100
    )
    print(bins0)
    print(f"Best: {best0}\nWorst:{worst0}")
    print()


def weighted_random():
    bins1, best1, worst1 = generate(
        g=g, generate_tour=weighted_random_tour, opt_cost=OPT_PATH_LENGTH, n=100
    )
    print(bins1)
    print(f"Best: {best1}\nWorst:{worst1}")
    print()


def hc():
    hc_start = timer()
    bins2, best2, worst2, iterations = generate_hc(
        g, m=HC_M, first_choice=False, opt_cost=OPT_PATH_LENGTH, n=HC_N
    )
    hc_end = timer()

    print(f"Hill climb N={HC_N}, M={HC_M} execution time {hc_end - hc_start}")
    print(f"Iterations {iterations}")
    print(bins2)
    print(f"Best: {best2}\nWorst:{worst2}")
    print()


def hc_iterations():
    """
    How does the cost change in consecutive iterations?
    """
    _, costs = hill_climb(g, m=HC_M, first_choice=False)
    print(f"COST | DIFF\n0: {costs[0]}, 0")
    for i in range(1, len(costs)):
        print(f"1: {costs[i]}, {costs[i - 1] - costs[i]}")


def draw_tours(g):
    draw(g, [opt_path])

    random_tour = weighted_random_tour(g, start=42)
    draw(g, [random_tour])

    hc_tour = hill_climb(g, m=HC_M, first_choice=False)[0]
    draw(g, [hc_tour])


if __name__ == "__main__":
    print(f"TSP for {DATASET}\nLength of the optimal cycle: {OPT_PATH_LENGTH}\n")
    g, opt_path = read_graph(dataset=DATASET)

    # draw_tours(g)

    random()
    weighted_random()
    hc()

    hc_iterations()
