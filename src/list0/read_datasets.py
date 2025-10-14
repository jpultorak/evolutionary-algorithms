from pathlib import Path

import numpy as np
import tsplib95

DATASETS = Path(__file__).resolve().parents[2] / "datasets"


def matrix_from_coords(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            d = np.hypot(dx, dy)
            w = int(np.floor(d + 0.5))
            dist[i, j] = w
            dist[j, i] = w
    return dist


def read_dataset(dataset):
    tsp_path = DATASETS / f"{dataset}.tsp"
    opt_path = DATASETS / f"{dataset}.opt.tour"

    problem = tsplib95.load(str(tsp_path))
    opt_tours = tsplib95.load(str(opt_path))

    assert problem.type == "TSP"
    assert problem.edge_weight_type == "EUC_2D"

    assert opt_tours.type == "TOUR"
    assert len(opt_tours.tours) == 1

    nodes = np.array(list(problem.get_nodes()), dtype=np.int32) - 1
    opt_path = np.array(opt_tours.tours[0], dtype=np.int32) - 1
    coords = np.array(list(problem.node_coords.values()), dtype=np.float64)

    dist = matrix_from_coords(coords)

    return nodes, coords, dist, opt_path
