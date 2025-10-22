import numpy as np

from list1.algorithms.pbil import pbil

INSTANCE_LENGTH = 10


def f_one_max(x: np.ndarray):
    return np.sum(x)


def f_deceptive_one_max(x: np.ndarray):
    d = len(x)
    ones_count = np.sum(x)

    if ones_count == 0:
        return d + 1
    return ones_count


def f_k_deceptive_one_max(x: np.ndarray, k):
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


def main():
    prob_vector = pbil(
        d=INSTANCE_LENGTH,
        eval_f=f_deceptive_one_max,
        pop_size=3,
        learn_rate=0.1,
        mut_prob=0.02,
        mut_shift=0.05,
        max_iter=200,
    )

    print(prob_vector)


if __name__ == "__main__":
    main()
