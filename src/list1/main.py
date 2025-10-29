import numpy as np

from list1.algorithms.pbil import pbil
from list1.utils import random_population

DECEPTIVE_K = 8

INSTANCE_LENGTH = 50
POP_SIZE = 100
LEARN_RATE = 0.1
MUT_PROB = 0.02
MUT_SHIFT = 0.05
MAX_ITER = 400

rng = np.random.default_rng()


def f_one_max(x: np.ndarray):
    return np.sum(x)


def f_deceptive_one_max(x: np.ndarray):
    d = len(x)
    ones_count = np.sum(x)

    if ones_count == 0:
        return d + 1
    return ones_count


def opt_k_deceptive(d: int, k: int = DECEPTIVE_K) -> int:
    blocks, tail = divmod(d, k)
    return blocks * (k + 1) + ((tail + 1) if tail > 0 else 0)


def f_k_deceptive_one_max(x: np.ndarray):
    k = DECEPTIVE_K
    d = len(x)
    res = 0
    for i in range(0, d, k):
        r = min(d, i + k)
        total = r - i
        sub_x = x[i:r]

        ones_count = np.sum(sub_x == 1)
        if ones_count == 0:
            res += total + 1
        else:
            res += ones_count

    return res


def one_max():
    prob_vector = pbil(
        d=INSTANCE_LENGTH,
        eval_f=f_one_max,
        pop_size=POP_SIZE,
        learn_rate=LEARN_RATE,
        mut_prob=MUT_PROB,
        mut_shift=MUT_SHIFT,
        max_iter=MAX_ITER,
        rng=rng,
    )

    print(prob_vector)
    print()


def deceptive_one_max():
    prob_vector = pbil(
        d=INSTANCE_LENGTH,
        eval_f=f_deceptive_one_max,
        pop_size=POP_SIZE,
        learn_rate=LEARN_RATE,
        mut_prob=MUT_PROB,
        mut_shift=MUT_SHIFT,
        max_iter=MAX_ITER,
        rng=rng,
    )

    print(prob_vector)
    print()


def k_deceptive_one_max():
    prob_vector = pbil(
        d=INSTANCE_LENGTH,
        eval_f=f_k_deceptive_one_max,
        pop_size=POP_SIZE,
        learn_rate=LEARN_RATE,
        mut_prob=MUT_PROB,
        mut_shift=MUT_SHIFT,
        max_iter=MAX_ITER,
        rng=rng,
    )

    opt = opt_k_deceptive(INSTANCE_LENGTH, k=DECEPTIVE_K)
    print(f"OPTIMAL: {opt}")
    pop = random_population(prob_vector, 100, rng)
    rank = np.array([f_k_deceptive_one_max(x) for x in pop])
    best = rank.max()
    print(best)
    print_prob_vector(prob_vector)
    print()


def print_prob_vector(p, decimals=5):
    p = np.asarray(p, float)
    np.set_printoptions(suppress=True)
    for i in range(0, len(p), 10):
        chunk = p[i : i + 10]
        print(" ".join(f"{x:.{decimals}f}" for x in chunk))


def main():
    # one_max()
    # deceptive_one_max()
    k_deceptive_one_max()


if __name__ == "__main__":
    main()
