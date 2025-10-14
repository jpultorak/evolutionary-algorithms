import matplotlib.pyplot as plt
import numpy as np


def draw_coords_tour(coords: np.ndarray, tours: list[np.ndarray]):
    plt.figure(figsize=(10, 10))
    x, y = coords[:, 0], coords[:, 1]
    plt.scatter(x, y)
    for tour in tours:
        cycle = np.append(tour, tour[0])
        plt.plot(x[cycle], y[cycle], linewidth=2)
    plt.show()
