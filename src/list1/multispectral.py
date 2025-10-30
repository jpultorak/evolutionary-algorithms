from pathlib import Path

import numpy as np

from list1.algorithms.pbil import pbil
from list1.utils import random_population

rng = np.random.default_rng()

DATASETS = Path(__file__).resolve().parents[2] / "datasets" / "list01"


def load_expert_labels():
    labels = np.loadtxt(DATASETS / "ImageExpertReduced.txt")
    labels = np.round(labels).astype(np.int8)
    return labels


def load_rule_outputs():
    rules = np.loadtxt(DATASETS / "ClassificationRules.txt")
    rules = np.round(rules).astype(np.int8)
    return rules


def eval_subset(mask: np.ndarray, rules: np.ndarray, labels: np.ndarray) -> int:
    rules = rules[mask]

    if len(rules) == 0:
        return 0

    counts1 = np.sum(rules == 1, axis=0)
    counts2 = np.sum(rules == 2, axis=0)
    counts3 = np.sum(rules == 3, axis=0)

    counts = np.vstack([counts1, counts2, counts3])
    pred = counts.argmax(axis=0) + 1

    return int((pred == labels).sum())


if __name__ == "__main__":
    expert_labels = load_expert_labels()

    rules = load_rule_outputs()

    def eval_f(mask):
        return eval_subset(mask, rules, expert_labels)

    d = rules.shape[0]
    prob_vector = pbil(
        rng=rng,
        d=d,
        eval_f=eval_f,
        pop_size=50,
        learn_rate=0.1,
        mut_prob=0.02,
        mut_shift=0.05,
        max_iter=200,
    )

    pop = random_population(prob_vector, 100, rng)
    ranks = np.array([eval_f(x) for x in pop])

    best_id = ranks.argmax()
    best = pop[best_id]
    best_val = ranks[best_id]

    print("BEST: ", best.astype(np.int8))
    print(f"BEST VAL: {best_val}")
    print(prob_vector)
