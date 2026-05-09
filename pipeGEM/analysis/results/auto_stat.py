"""Result class for automated statistical testing."""
from typing import Union, Dict, Optional

import pandas as pd

from ._base import BaseAnalysis


class AutoStatResult(BaseAnalysis):
    """Result from :class:`~pipeGEM.analysis._auto_stat.AutoStatTest`.

    Attributes
    ----------
    main_result : pandas.DataFrame
        The primary test result (t-test / Mann-Whitney / ANOVA /
        Kruskal-Wallis).
    pairwise_result : pandas.DataFrame or None
        Pairwise comparison table (Tukey / Dunn), ``None`` for 2-group
        tests.
    """

    def add_result(self, result_dict: dict) -> None:
        self._result.update(result_dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def main_result(self) -> pd.DataFrame:
        return self._result.get("main_result", pd.DataFrame())

    @property
    def pairwise_result(self) -> Optional[pd.DataFrame]:
        return self._result.get("pairwise_result")

    @property
    def assumption_results(self) -> Dict:
        """Normality and homoscedasticity test results."""
        return {
            "normality": self._result.get("normality", {}),
            "homoscedasticity": self._result.get("homoscedasticity", {}),
        }

    @property
    def effect_sizes(self) -> Dict:
        """Effect sizes (Cohen's *d*, eta-squared, etc.)."""
        return self._result.get("effect_sizes", {})

    @property
    def corrected_p_values(self) -> Optional[pd.Series]:
        """Multiple-comparison corrected p-values."""
        return self._result.get("corrected_p_values")

    @property
    def sample_sizes(self) -> Dict:
        """Sample sizes per group."""
        return self._result.get("sample_sizes", {})

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, format: str = "text") -> Union[str, pd.DataFrame]:
        """Generate a publication-ready summary.

        Parameters
        ----------
        format : str
            ``"text"`` for a formatted string, ``"dataframe"`` for a
            :class:`~pandas.DataFrame`, ``"latex"`` for a LaTeX table.

        Returns
        -------
        str or pandas.DataFrame
        """
        lines = []
        log = self._log

        # Header
        lines.append("=" * 60)
        lines.append("AutoStatTest Summary")
        lines.append("=" * 60)
        lines.append(f"  Groups         : {log.get('n_groups')}")
        lines.append(f"  Parametric     : {log.get('parametric')}")
        lines.append(f"  Alpha          : {log.get('alpha')}")
        lines.append(f"  Correction     : {log.get('correction')}")

        # Sample sizes
        lines.append(f"  Sample sizes   : {self.sample_sizes}")

        # Assumptions
        norm = self.assumption_results["normality"]
        homosc = self.assumption_results["homoscedasticity"]
        lines.append("")
        lines.append("Assumption Tests:")
        if "is_normal" in norm:
            for g, is_n in norm["is_normal"].items():
                p = norm["p_value"].get(g, "N/A")
                lines.append(f"  Normality [{g}] : {'pass' if is_n else 'FAIL'} (p={p:.4g})" if isinstance(p, float) else f"  Normality [{g}] : {'pass' if is_n else 'FAIL'}")
        lines.append(f"  Equal var      : {'pass' if homosc.get('is_equal_var') else 'FAIL'} "
                      f"(p={homosc.get('p_value', 'N/A')})")

        # Main test
        lines.append("")
        lines.append("Main Test:")
        main = self.main_result
        if isinstance(main, pd.DataFrame) and not main.empty:
            lines.append(main.to_string(index=False))

        # Effect sizes
        if self.effect_sizes:
            lines.append("")
            lines.append("Effect Sizes:")
            for k, v in self.effect_sizes.items():
                lines.append(f"  {k} = {v:.4f}")

        # Pairwise
        pw = self.pairwise_result
        if pw is not None and not pw.empty:
            lines.append("")
            lines.append("Pairwise Comparisons:")
            lines.append(pw.to_string(index=False))

        lines.append("=" * 60)
        text = "\n".join(lines)

        if format == "text":
            return text
        elif format == "dataframe":
            return self.main_result
        elif format == "latex":
            return self.main_result.to_latex(index=False)
        return text
