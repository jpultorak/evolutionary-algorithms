from dataclasses import dataclass

import numpy as np


@dataclass
class Benchmark:
    name: str
    f_objective: callable
    bounds: tuple[float, float]
    optimum: float


def sphere(x):
    return np.sum(x**2)


def griewank(x):
    n = x.size
    sum_ = np.sum(x**2) / 4000.0
    indices = np.arange(1, n + 1, dtype=float)
    prod = np.prod(np.cos(x / np.sqrt(indices)))
    return 1.0 + sum_ - prod


def rastrigin(x):
    n = x.size
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))


def schwefel(x):
    n = x.size
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)


BENCHMARKS = {
    "sphere": Benchmark(
        name="Sphere",
        f_objective=sphere,
        bounds=(-5.12, 5.12),
        optimum=0.0,
    ),
    "griewank": Benchmark(
        name="Griewank",
        f_objective=griewank,
        bounds=(-600.0, 600.0),
        optimum=0.0,
    ),
    "rastrigin": Benchmark(
        name="Rastrigin",
        f_objective=rastrigin,
        bounds=(-5.12, 5.12),
        optimum=0.0,
    ),
    "schwefel": Benchmark(
        name="Schwefel",
        f_objective=schwefel,
        bounds=(-500.0, 500.0),
        optimum=0.0,
    ),
    "rosenbrock": Benchmark(
        name="Rosenbrock",
        f_objective=rosenbrock,
        bounds=(-5.0, 10.0),
        optimum=0.0,
    ),
}
