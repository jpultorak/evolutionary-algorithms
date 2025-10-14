import numpy as np


def tour_cost(dist: np.ndarray, path: np.ndarray):
    return np.sum(dist[path, np.roll(path, -1)], dtype=np.int32)
