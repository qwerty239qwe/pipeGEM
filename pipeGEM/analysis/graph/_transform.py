from collections import OrderedDict

import numpy as np
import cobra
import networkx as nx
import matplotlib.pyplot as plt

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

    def _get_branch_nodes(self, rxns, skip_mets):
        met_names, rxn_names = self.lp.row_names, self.lp.col_names
        forward, backward = {}, {}
        for rxn in rxns:
            r_id = rxn_names.index(rxn)
            met_ids = np.nonzero(self.lp.S[:, r_id])[0]
            forward[rxn], backward[rxn] = [], []
            for m_id in met_ids:
                if met_names[m_id] not in skip_mets:
                    if self.lp.S[m_id, r_id] < 0:
                        backward[rxn] = met_names[m_id]
                    elif self.lp.S[r_id, m_id] > 0:
                        forward[rxn] = met_names[m_id]
                    else:
                        raise RuntimeError()
        return forward, backward

    @staticmethod
    def _get_shortest_path_list(g, shortest_path, met_names, rxn_names):
        m_nodes, r_nodes, edges = OrderedDict(), OrderedDict(), []
        node_order = {}
        for i, n in enumerate(shortest_path):
            if n in met_names:
                m_nodes[n] = g.nodes[n]
                node_order[n] = i
            elif n in rxn_names:
                r_nodes[n] = g.nodes[n]
                node_order[n] = i
            else:
                raise RuntimeError("This is not a valid node, should be one of the reactions or metabolites: ", n)
            if i < len(shortest_path) - 1:
                data = g.edges[n][shortest_path[i+1]]
                edges.append((n, shortest_path[i+1], data))
        return m_nodes, r_nodes, edges, node_order

    def plot_shortest(self, g, s, t):
        shortest_path = nx.shortest_path(g, weight="weight", source=s, target=t)
        met_names, rxn_names = self.lp.row_names, self.lp.col_names
        m_nodes, r_nodes, added_edges, node_order = self._get_shortest_path_list(g, shortest_path, met_names, rxn_names)
        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(m_nodes)
        new_graph.add_nodes_from(r_nodes)
        new_graph.add_edges_from(added_edges)
        pos = {name: (node_order[name], 0) for name in m_nodes}
        pos.update({name: (node_order[name], 0) for name in r_nodes})
        forward, backward = self._get_branch_nodes(r_nodes, m_nodes)
        new_graph.add_nodes_from([met for r, m in forward.items() for met in m])
        new_graph.add_nodes_from([met for r, m in backward.items() for met in m])
        new_graph.add_edges_from([(r, met) for r, m in forward.items() for met in m])
        new_graph.add_edges_from([(met, r) for r, m in backward.items() for met in m])
        pos.update({name: (node_order[name] - 1, 1 + i) for i, name in enumerate(backward)})
        pos.update({name: (node_order[name] + 1, 1 + i) for i, name in enumerate(forward)})
        nx.draw_networkx(new_graph, pos)
        plt.show()
