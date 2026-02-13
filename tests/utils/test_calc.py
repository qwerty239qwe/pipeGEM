"""Tests for pipeGEM/utils/_calc.py — Jaccard index calculation."""
from unittest.mock import MagicMock

import pytest

from pipeGEM.utils._calc import calc_jaccard_index


def _make_mock_model(reaction_ids, metabolite_ids, gene_ids):
    """Create a mock pipeGEM model with the required *_ids properties."""
    m = MagicMock()
    m.reaction_ids = list(reaction_ids)
    m.metabolite_ids = list(metabolite_ids)
    m.gene_ids = list(gene_ids)
    return m


class TestCalcJaccardIndex:

    def test_identical_models_jaccard_1(self):
        """Same model -> Jaccard = 1.0."""
        m = _make_mock_model(["R1", "R2"], ["m1", "m2"], ["g1", "g2"])
        assert calc_jaccard_index(m, m) == 1.0

    def test_disjoint_models_jaccard_0(self):
        """No shared components -> 0.0."""
        m1 = _make_mock_model(["R1", "R2"], ["m1", "m2"], ["g1", "g2"])
        m2 = _make_mock_model(["R3", "R4"], ["m3", "m4"], ["g3", "g4"])
        assert calc_jaccard_index(m1, m2) == 0.0

    def test_partial_overlap(self):
        """Known overlap -> exact value."""
        m1 = _make_mock_model(["R1", "R2", "R3"], ["m1", "m2", "m3"], ["g1"])
        m2 = _make_mock_model(["R1", "R2", "R4"], ["m1", "m4", "m5"], ["g1"])
        j = calc_jaccard_index(m1, m2)
        # Reactions: inter=2, union=4
        # Metabolites: inter=1, union=5
        # Genes: inter=1, union=1
        # Total: (2 + 1 + 1) / (4 + 5 + 1) = 4/10 = 0.4
        assert pytest.approx(j, abs=1e-10) == 4.0 / 10.0

    def test_subset_components_genes(self):
        """components=['genes'] only considers genes."""
        m1 = _make_mock_model(["R1"], ["m1"], ["g1"])
        m2 = _make_mock_model(["R2"], ["m2"], ["g1"])
        j = calc_jaccard_index(m1, m2, components=["genes"])
        assert j == 1.0

    def test_reactions_only(self):
        """components=['reactions'] only considers reactions."""
        m1 = _make_mock_model(["R1", "R2"], ["m1"], ["g1"])
        m2 = _make_mock_model(["R2", "R3"], ["m2"], ["g2"])
        j = calc_jaccard_index(m1, m2, components=["reactions"])
        # Reactions: {R1,R2} & {R2,R3} -> inter=1, union=3 -> j=1/3
        assert pytest.approx(j, abs=1e-10) == 1.0 / 3.0
