import numpy as np
import cobra
import networkx as nx

from pipeGEM.core._problem import LP_Problem


class ModelGraph:
    def __init__(self,
                 model,
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
        if is_directional:
            g = nx.DiGraph()
            g.add_nodes_from([f"m{i}" for i in range(m)], color=self.met_color)
            g.add_nodes_from([f"r{i}" for i in range(n)], color=self.rxn_color)
            revs = self.lp.get_rev()
            g.add_edges_from([(f"m{i}", f"n{i}", {"weight": 10000 if i in self.cofactor_index else 1})
                              if self.lp.S[i, j] < 0 else
                              (f"n{i}", f"m{j}", {"weight": 10000 if i in self.cofactor_index else 1})
                              for i in range(m)
                              for j in range(n) if self.lp.S[i, j] != 0])
            g.add_edges_from([(f"m{i}", f"n{i}", {"weight": 10000 if i in self.cofactor_index else 1})
                              if self.lp.S[i, j] > 0 else
                              (f"n{i}", f"m{j}", {"weight": 10000 if i in self.cofactor_index else 1})
                              for j in range(n) if revs[n]
                              for i in range(m) if self.lp.S[i, j] != 0])
        else:
            raise NotImplementedError
        return g

    def plot_shortest(self, g, s, t):
        shortest_path = nx.shortest_path(g, weight="weight", source=s, target=t)
