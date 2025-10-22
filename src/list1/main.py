import numpy as np

from list1.algorithms.pbil import pbil

DECEPTIVE_K = 3

INSTANCE_LENGTH = 10
POP_SIZE = 10
LEARN_RATE = 0.1
MUT_PROB = 0.02
MUT_SHIFT = 0.05
MAX_ITER = 200

rng = np.random.default_rng()


def f_one_max(x: np.ndarray):
    return np.sum(x)


def f_deceptive_one_max(x: np.ndarray):
    d = len(x)
    ones_count = np.sum(x)

    if ones_count == 0:
        return d + 1
    return ones_count


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

    print(prob_vector)
    print()


def main():
    one_max()
    deceptive_one_max()
    k_deceptive_one_max()


if __name__ == "__main__":
    main()
