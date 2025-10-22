import numpy as np

from list1.utils import binary_random, random_population


def pbil(rng, d, eval_f, pop_size, learn_rate, mut_prob, mut_shift, max_iter=200):
    prob_vector = np.full(d, fill_value=0.5, dtype=np.float32)

    for _ in range(max_iter):
        population = random_population(prob_vector, pop_size, rng)
        ranks = np.array([eval_f(x) for x in population])

        best = population[ranks.argmax()]
        prob_vector = prob_vector * (1 - learn_rate) + best * learn_rate

        # Mutation
        for k in range(d):
            if rng.uniform() < mut_prob:
                prob_vector[k] = (
                    prob_vector[k] * (1 - mut_shift)
                    + binary_random(0.5, rng) * mut_shift
                )

    return prob_vector
