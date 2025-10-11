from pathlib import Path

import tsplib95

from list0.utils import euc_2d

DATASETS = Path(__file__).resolve().parents[2] / "datasets"


def read_graph(dataset):
    tsp_path = DATASETS / f"{dataset}.tsp"
    opt_path = DATASETS / f"{dataset}.opt.tour"

    problem = tsplib95.load(str(tsp_path))
    problem_opt = tsplib95.load(str(opt_path))

    assert problem.type == "TSP"
    assert problem.edge_weight_type == "EUC_2D"

    assert problem_opt.type == "TOUR"
    assert len(problem_opt.tours) == 1

    assert problem.get_weight(1, 2) == euc_2d(
        problem.node_coords[1], problem.node_coords[2]
    )

    g = problem.get_graph()
    opt_path = problem_opt.tours[0]
    assert problem.get_weight(2, 1) == g[1][2]["weight"]

    return g, opt_path
