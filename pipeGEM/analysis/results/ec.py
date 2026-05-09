"""Result classes for enzyme-constrained model analyses."""
import pandas as pd

from ._base import BaseAnalysis


class GECKOLightAnalysis(BaseAnalysis):
    """Result from GECKO-light analysis.

    Stores the modified flux bounds and kcat mapping produced by
    ``apply_gecko_light``.

    Parameters
    ----------
    log : dict
        Parameters used during the analysis (sigma, f_factor, ptot, etc.).

    Attributes
    ----------
    modified_bounds : dict[str, tuple[float, float]]
        Mapping of reaction IDs to their new ``(lower_bound, upper_bound)``
        after applying enzyme constraints.
    kcat_mapping : dict[str, float]
        Mapping of reaction IDs to the kcat values used.
    enzyme_usage : pandas.DataFrame
        DataFrame summarising enzyme usage per reaction.
    """
    RESULT_FIELDS = {
        "modified_bounds": {"required": True},
        "kcat_mapping": {"required": True},
        "enzyme_usage": {"required": True},
        "ec_model": {"required": True},
    }
    ALLOW_EXTRA_RESULT_FIELDS = False

    @classmethod
    def from_results(cls, log, modified_bounds, kcat_mapping, enzyme_usage, ec_model):
        result = cls(log=log)
        result.set_result(
            modified_bounds=modified_bounds,
            kcat_mapping=kcat_mapping,
            enzyme_usage=enzyme_usage,
            ec_model=ec_model,
        )
        result.validate_result()
        return result

    @property
    def ec_model(self):
        return self._result.get("ec_model")

    @property
    def modified_bounds(self):
        return self._result.get("modified_bounds", {})

    @property
    def kcat_mapping(self):
        return self._result.get("kcat_mapping", {})

    @property
    def enzyme_usage(self):
        return self._result.get("enzyme_usage", pd.DataFrame())

    @property
    def n_reactions_with_enzyme_data(self):
        return self._log.get("n_reactions_with_enzyme_data", 0)

    @property
    def n_bound_reductions(self):
        return self._log.get("n_bound_reductions", 0)


class GECKOFullAnalysis(BaseAnalysis):
    """Result from full GECKO analysis.

    Stores the enzyme-constrained model (ecModel) together with metadata
    about the protein pool and draw/arm reactions that were created.

    Parameters
    ----------
    log : dict
        Parameters used during the analysis.

    Attributes
    ----------
    ec_model : cobra.Model
        The enzyme-constrained COBRA model.
    protein_pool_id : str
        ID of the protein pool pseudo-metabolite.
    draw_reactions : list[str]
        IDs of the draw reactions added to the model.
    arm_reactions : list[str]
        IDs of the arm reactions added to the model.
    """
    RESULT_FIELDS = {
        "ec_model": {"required": True},
        "protein_pool_id": {"required": True},
        "draw_reactions": {"required": True},
        "arm_reactions": {"required": True},
    }
    ALLOW_EXTRA_RESULT_FIELDS = False

    @classmethod
    def from_results(cls, log, ec_model, protein_pool_id, draw_reactions, arm_reactions):
        result = cls(log=log)
        result.set_result(
            ec_model=ec_model,
            protein_pool_id=protein_pool_id,
            draw_reactions=draw_reactions,
            arm_reactions=arm_reactions,
        )
        result.validate_result()
        return result

    @property
    def ec_model(self):
        return self._result.get("ec_model")

    @property
    def protein_pool_id(self) -> str:
        return self._result.get("protein_pool_id", "prot_pool")

    @property
    def draw_reactions(self):
        return self._result.get("draw_reactions", [])

    @property
    def arm_reactions(self):
        return self._result.get("arm_reactions", [])

    @property
    def n_enzyme_constraints(self):
        return self._log.get("n_enzyme_constraints", self._log.get("n_constrained", 0))
