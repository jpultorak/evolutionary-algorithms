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


BENCHMARKS = {
    "sphere": Benchmark(
        name="Sphere",
        f_objective=sphere,
        bounds=(-5.12, 5.12),
        optimum=0.0,
    ),
}
