import numpy as np

rng = np.random.default_rng()


def pbil(d, eval_f, pop_size, learn_rate, mut_prob, mut_shift, max_iter=200):
    prob_vector = np.full(d, fill_value=0.5, dtype=np.float32)

    for _ in range(max_iter):
        population = random_population(prob_vector, pop_size)
        ranks = np.array([eval_f(x) for x in population])

        best = population[ranks.argmax()]
        prob_vector = prob_vector * (1 - learn_rate) + best * learn_rate

        # Mutation
        for k in range(d):
            if rng.uniform() < mut_prob:
                prob_vector[k] = (
                    prob_vector[k] * (1 - mut_shift) + binary_random(0.5) * mut_shift
                )

    return prob_vector


def binary_random(prob):
    return rng.choice(a=[1, 0], p=[prob, 1 - prob])


def random_genes(prob_vector):
    d = len(prob_vector)
    return rng.random(d) < prob_vector


def random_population(prob_vector, pop_size):
    return np.array([random_genes(prob_vector) for _ in range(pop_size)])
