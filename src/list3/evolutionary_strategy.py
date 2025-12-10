import numpy as np


def init_population(pop_size, dim, bounds, rng, init_sigma=0.3):
    low, high = bounds
    x = rng.uniform(low, high, size=(pop_size, dim))
    sigma0 = init_sigma * (high - low)
    sigma = np.full_like(x, sigma0)
    return x, sigma


def evaluate_population(x, f_objective):
    return np.apply_along_axis(f_objective, 1, x)


def mutate(pop_x, sigma, tau, tau0, rng, min_sigma=1e-8):
    n, d = pop_x.shape

    eps0 = rng.normal(loc=0.0, scale=tau0, size=(n, 1))
    eps = rng.normal(loc=0.0, scale=tau, size=(n, d))

    new_sigma = sigma * np.exp(eps + eps0)
    new_sigma = np.maximum(new_sigma, min_sigma)

    noise = rng.normal(loc=0.0, scale=new_sigma)
    new_x = pop_x + noise
    return new_x, new_sigma


def es(
    f_objective: callable,
    dim: int,
    rng,
    mu: int = 15,
    lambda_: int = 100,
    bounds: tuple = (-5.0, 5.0),
    iters: int = 1000,
    plus: bool = True,
    tau: float | None = None,
    tau0: float | None = None,
):
    if tau is None:
        tau = 1.0 / np.sqrt(2.0 * np.sqrt(dim))
    if tau0 is None:
        tau0 = 1.0 / np.sqrt(2.0 * dim)

    x, sigma = init_population(mu, dim, bounds, rng)
    fitness = evaluate_population(x, f_objective)

    history = []
    low, high = bounds

    for _ in range(iters):
        parent_indices = rng.integers(0, mu, size=lambda_)
        parent_x = x[parent_indices]
        parent_sigma = sigma[parent_indices]

        child_x, child_sigma = mutate(parent_x, parent_sigma, tau, tau0, rng)
        child_x = np.clip(child_x, low, high)
        child_fitness = evaluate_population(child_x, f_objective)

        if plus:
            all_x = np.vstack([x, child_x])
            all_sigma = np.vstack([sigma, child_sigma])
            all_fitness = np.concatenate([fitness, child_fitness])

            best_idx = np.argsort(all_fitness)[:mu]
            x = all_x[best_idx]
            sigma = all_sigma[best_idx]
            fitness = all_fitness[best_idx]
        else:
            best_idx = np.argsort(child_fitness)[:mu]
            x = child_x[best_idx]
            sigma = child_sigma[best_idx]
            fitness = child_fitness[best_idx]

        history.append(fitness[0])

    best_idx = np.argmin(fitness)
    return x[best_idx], fitness[best_idx], np.array(history)
