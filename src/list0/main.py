from list0.algorithms.hill_climb import hill_climb
from list0.algorithms.random_tours import random_tour, weighted_random_tour
from list0.eval import generate
from list0.read_datasets import read_graph

if __name__ == "__main__":
    DATASET = "berlin52"
    OPT_PATH_LENGTH = 7542

    print(f"TSP for {DATASET}\nLength of the optimal cycle: {OPT_PATH_LENGTH}\n")
    g, opt_path = read_graph(dataset=DATASET)

    # EXERCISE 1
    # tour1 = random_tour(g)
    # draw(g, [opt_path])

    # EXERCISE 2

    bins0, best0, worst0 = generate(
        g=g, generate_tour=random_tour, opt_cost=OPT_PATH_LENGTH, n=100
    )
    print(bins0)
    print(f"Best: {best0}\nWorst:{worst0}")

    bins1, best1, worst1 = generate(
        g=g, generate_tour=weighted_random_tour, opt_cost=OPT_PATH_LENGTH, n=100
    )
    print(bins1)
    print(f"Best: {best1}\nWorst:{worst1}")

    def hill_climb_m(g):
        return hill_climb(g, m=3)[0]

    bins2, best2, worst2 = generate(
        g, generate_tour=hill_climb_m, opt_cost=OPT_PATH_LENGTH, n=10
    )
    print(bins2)
    print(f"Best: {best2}\nWorst:{worst2}")
