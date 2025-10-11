import matplotlib.pyplot as plt
import networkx as nx

from list0.utils import path_edges


def draw(g, paths):
    plt.figure(figsize=(15, 15))
    pos = {v: g.nodes[v]["coord"] for v in g.nodes}

    nx.draw_networkx_nodes(g, pos, node_size=60)

    for path in paths:
        nx.draw_networkx_edges(g, pos, edgelist=path_edges(path, cyclic=True), width=3)

    plt.show()
