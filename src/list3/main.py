import numpy as np

from list3.benchmark import UNCONSTRAINED_BENCHMARKS
from list3.benchmark_runner import run_es


def run_benchmarks(rng, benchmarks):
    dim = 2
    runs = 5
    iters = 1000

    for plus in (True, False):
        scheme = "ES(μ+λ)" if plus else "ES(μ,λ)"
        print(f"\n=== {scheme} ===")

        for _, benchmark in benchmarks.items():
            res = run_es(
                benchmark=benchmark,
                rng=rng,
                dim=benchmark.dim if benchmark.dim is not None else dim,
                runs=runs,
                plus=plus,
                mu=15,
                lambda_=100,
                iters=iters,
            )

            print(
                f"{benchmark.name} | "
                f"all runs best={res['all_runs_best']:.6f} | "
                f"mean={res['mean_best']:.6f} | "
                f"std={res['std_best']:.6f} | "
                f"optimum={benchmark.optimum}"
            )


def main():
    rng = np.random.default_rng(42)

    # run_benchmarks(rng, BENCHMARKS)
    run_benchmarks(rng, UNCONSTRAINED_BENCHMARKS)


if __name__ == "__main__":
    main()
