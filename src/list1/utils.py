import numpy as np


def binary_random(prob, rng):
    return rng.choice(a=[1, 0], p=[prob, 1 - prob])


def random_genes(prob_vector, rng):
    d = len(prob_vector)
    return rng.random(d) < prob_vector


def random_population(prob_vector, pop_size, rng):
    return np.array([random_genes(prob_vector, rng) for _ in range(pop_size)])
