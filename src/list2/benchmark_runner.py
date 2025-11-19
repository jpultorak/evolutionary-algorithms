import numpy as np

from list2.benchmark import Benchmark
from list2.evolutionary_strategy import es


def run_es(
    benchmark: Benchmark,
    rng,
    dim,
    runs=10,
    plus=True,
    mu=15,
    lambda_=100,
    iters=1000,
    tau=None,
    tau0=None,
):
    best_values = []
    histories = []

    for _ in range(runs):
        best_x, best_f, history = es(
            f_objective=benchmark.f_objective,
            dim=dim,
            mu=mu,
            lambda_=lambda_,
            bounds=benchmark.bounds,
            iters=iters,
            plus=plus,
            rng=rng,
            tau=tau,
            tau0=tau0,
        )
        best_values.append(best_f)
        histories.append(history)

    best_values = np.array(best_values, dtype=float)

    return {
        "name": benchmark.name,
        "plus": plus,
        "optimum": benchmark.optimum,
        "dim": dim,
        "runs": runs,
        "mu": mu,
        "lambda_": lambda_,
        "iters": iters,
        "bounds": benchmark.bounds,
        "best_values": best_values,
        "mean_best": float(best_values.mean()),
        "std_best": float(best_values.std()),
        "all_runs_best": np.min(best_values),
        "histories": histories,
    }
