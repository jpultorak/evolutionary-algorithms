import numpy as np

from list1.algorithms.pbil import pbil

INSTANCE_LENGTH = 100


def f_one_max(x: np.ndarray):
    d = len(x)
    goal = np.ones(d, dtype=np.bool)
    return np.sum(x == goal)


def main():
    prob_vector = pbil(
        d=INSTANCE_LENGTH,
        eval_f=f_one_max,
        pop_size=100,
        learn_rate=0.1,
        mut_prob=0.02,
        mut_shift=0.05,
        max_iter=200,
    )

    print(prob_vector)


if __name__ == "__main__":
    main()
