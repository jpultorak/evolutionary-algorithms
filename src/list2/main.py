import numpy as np

from list2.benchmark import BENCHMARKS
from list2.benchmark_runner import run_es


def main():
    rng = np.random.default_rng(42)
    dim = 30
    runs = 5
    iters = 5000

    for plus in (True, False):
        scheme = "ES(μ+λ)" if plus else "ES(μ,λ)"
        print(f"\n=== {scheme} ===")

        for _, bench in BENCHMARKS.items():
            res = run_es(
                benchmark=bench,
                rng=rng,
                dim=dim,
                runs=runs,
                plus=plus,
                mu=15,
                lambda_=100,
                iters=iters,
            )

            print(
                f"{bench.name:10s} | "
                f"mean={res['mean_best']:.6f} | "
                f"std={res['std_best']:.6f} | "
                f"optimum={bench.optimum}"
            )


if __name__ == "__main__":
    main()
