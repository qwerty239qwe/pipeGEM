import numpy as np
import cobra
import networkx as nx

from pipeGEM.core import LP_Problem


class ModelGraph:
    def __init__(self,
                 model,
                 data=None,
                 occur_ratio=0.1,
                 met_color="blue",
                 rxn_color="red"):
        self.lp = LP_Problem(model)
        self.cofactor_index = self._check_cofactors(self.lp.S, occur_ratio=occur_ratio)
        self.met_color, self.rxn_color = met_color, rxn_color

    @staticmethod
    def _check_cofactors(s, occur_ratio=0.1):
        m, n = s.shape
        cofactor_index = np.where((s != 0).sum(axis=1) > n * occur_ratio)[0]
        return cofactor_index

    def get_bipartite_graph(self, is_directional=True):
        m, n = self.lp.S.shape
        met_names, rxn_names = self.lp.row_names, self.lp.col_names
        if is_directional:
            g = nx.DiGraph()
            g.add_nodes_from([m for m in met_names], color=self.met_color)
            g.add_nodes_from([r for r in rxn_names], color=self.rxn_color)
            revs = self.lp.get_rev()
            g.add_edges_from([(m, r, {"weight": 10000 if i in self.cofactor_index else 1})
                              if self.lp.S[i, j] < 0 else
                              (r, m, {"weight": 10000 if i in self.cofactor_index else 1})
                              for i, m in enumerate(met_names)
                              for j, r in enumerate(rxn_names) if self.lp.S[i, j] != 0])
            g.add_edges_from([(r, m, {"weight": 10000 if i in self.cofactor_index else 1})
                              if self.lp.S[i, j] > 0 else
                              (r, m, {"weight": 10000 if i in self.cofactor_index else 1})
                              for j, r in enumerate(rxn_names) if revs[j]
                              for i, m in enumerate(met_names) if self.lp.S[i, j] != 0])
        else:
            raise NotImplementedError
        return g

    @staticmethod
    def _get_shortest_path_list(g, shortest_path, met_names, rxn_names):
        nodes, edges = [], []
        for i, n in enumerate(shortest_path):
            if n in met_names or n in rxn_names:
                nodes.append(("met", n, g.nodes[n]))
            else:
                raise RuntimeError("This is not a valid node, should be one of the reactions or metabolites: ", n)
            if i != len(shortest_path) - 1:
                edges.append(g.edges[n][shortest_path[i+1]])
        return nodes, edges

    def plot_shortest(self, g, s, t):
        shortest_path = nx.shortest_path(g, weight="weight", source=s, target=t)
        met_names, rxn_names = self.lp.row_names, self.lp.col_names
        added_nodes, added_edges = self._get_shortest_path_list(g, shortest_path, met_names, rxn_names)
        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(added_nodes)
        new_graph.add_edges_from(added_edges)
