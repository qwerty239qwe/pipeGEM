"""Tests for helper classes in pipeGEM/analysis/_consistency.py.

Only tests FluxLogger, StoppingCriteria, and IsInSetStoppingCriteria
(not FASTCC/FVA which require a solver).
"""
import numpy as np
import pandas as pd
import pytest

from pipeGEM.analysis._consistency import (
    FluxLogger,
    StoppingCriteria,
    IsInSetStoppingCriteria,
)


# =====================================================================
# FluxLogger
# =====================================================================

class TestFluxLogger:

    def test_tic_increments(self):
        fl = FluxLogger()
        assert fl._iter == 0
        fl.tic()
        assert fl._iter == 1
        fl.tic()
        assert fl._iter == 2

    def test_add_stores(self):
        fl = FluxLogger()
        fl.tic()
        flux = pd.Series({"R1": 1.0, "R2": 2.0})
        fl.add("test", flux)
        assert len(fl._dfs) == 1

    def test_to_frame_returns_correct_df(self):
        fl = FluxLogger()
        fl.tic()
        flux1 = pd.Series({"R1": 1.0, "R2": 2.0})
        fl.add("LP7", flux1)
        fl.tic()
        flux2 = pd.Series({"R1": 3.0, "R2": 4.0})
        fl.add("LP3", flux2)
        df = fl.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_column_naming(self):
        """Column naming follows {name}_{iter} pattern."""
        fl = FluxLogger()
        fl.tic()  # iter=1
        flux = pd.Series({"R1": 1.0})
        fl.add("LP7", flux)
        fl.tic()  # iter=2
        fl.add("LP3", pd.Series({"R1": 2.0}))
        df = fl.to_frame()
        assert "LP7_1" in df.columns
        assert "LP3_2" in df.columns

    def test_empty_to_frame(self):
        """Empty FluxLogger should concat without error (empty list)."""
        fl = FluxLogger()
        # Calling to_frame on empty should raise or return empty
        # pd.concat([]) raises ValueError
        with pytest.raises(ValueError):
            fl.to_frame()


# =====================================================================
# StoppingCriteria
# =====================================================================

class TestStoppingCriteria:

    def test_base_check_raises_not_implemented(self):
        """Base class check() raises NotImplementedError."""
        sc = StoppingCriteria()
        with pytest.raises(NotImplementedError):
            sc.check(removed=[], kept=[])


# =====================================================================
# IsInSetStoppingCriteria
# =====================================================================

class TestIsInSetStoppingCriteria:

    def test_removed_intersects_ess_set_true(self):
        """removed ∩ ess_set -> True."""
        sc = IsInSetStoppingCriteria(ess_set={"R1", "R2", "R3"})
        assert sc.check(removed=["R1", "R5"], kept=["R4"]) is True

    def test_removed_disjoint_from_ess_set_false(self):
        """removed disjoint from ess_set -> False."""
        sc = IsInSetStoppingCriteria(ess_set={"R1", "R2", "R3"})
        assert sc.check(removed=["R4", "R5"], kept=["R6"]) is False

    def test_empty_removed_false(self):
        """Empty removed -> False."""
        sc = IsInSetStoppingCriteria(ess_set={"R1", "R2"})
        assert sc.check(removed=[], kept=["R1"]) is False
