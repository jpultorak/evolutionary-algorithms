from dataclasses import dataclass

import numpy as np


@dataclass
class Benchmark:
    name: str
    f_objective: callable
    bounds: tuple[float, float]
    optimum: float
    dim: int | None  # None if the dimension is not fixed


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
        name="Sphere", f_objective=sphere, bounds=(-5.12, 5.12), optimum=0.0, dim=None
    ),
    "griewank": Benchmark(
        name="Griewank",
        f_objective=griewank,
        bounds=(-600.0, 600.0),
        optimum=0.0,
        dim=None,
    ),
    "rastrigin": Benchmark(
        name="Rastrigin",
        f_objective=rastrigin,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        dim=None,
    ),
    "schwefel": Benchmark(
        name="Schwefel",
        f_objective=schwefel,
        bounds=(-500.0, 500.0),
        optimum=0.0,
        dim=None,
    ),
    "rosenbrock": Benchmark(
        name="Rosenbrock",
        f_objective=rosenbrock,
        bounds=(-5.0, 10.0),
        optimum=0.0,
        dim=None,
    ),
}


@dataclass
class ConstrainedBenchmark:
    name: str
    f_objective: callable
    constraints_ineq: list[callable]
    bounds: tuple[float, float]
    optimum: float
    dim: int | None


def make_penalty_objective(
    problem: ConstrainedBenchmark,
):
    def penalized(x: np.ndarray) -> float:
        x_arr = np.asarray(x)
        value = problem.f_objective(x_arr)
        penalty = 0.0
        for g in problem.constraints_ineq:
            v = g(x_arr)
            if v > 1e-9:
                penalty += v * v
        return value + 1e6 * penalty

    return penalized


def constrained_to_unconstrained(
    problem: ConstrainedBenchmark,
) -> Benchmark:
    f_pen = make_penalty_objective(problem)
    return Benchmark(
        name=problem.name,
        f_objective=f_pen,
        bounds=problem.bounds,
        optimum=problem.optimum,
        dim=problem.dim,
    )


def make_g3() -> Benchmark:
    def f_min(x: np.ndarray):
        n = x.size
        return -(np.sqrt(n) ** n * np.prod(x))

    def g1(x: np.ndarray):
        return np.sum(x**2) - 1.0

    def g2(x: np.ndarray):
        return -(np.sum(x**2) - 1.0)

    optimum_min = -1.0

    constrained = ConstrainedBenchmark(
        name="G3",
        f_objective=f_min,
        constraints_ineq=[g1, g2],
        bounds=(0, 1),
        optimum=optimum_min,
        dim=None,
    )
    return constrained_to_unconstrained(constrained)


UNCONSTRAINED_BENCHMARKS = {
    "G3": make_g3(),
}

if __name__ == "__main__":
    g3_f = UNCONSTRAINED_BENCHMARKS["G3"].f_objective
    print(g3_f([1 / np.sqrt(2), 1 / np.sqrt(2)]))
