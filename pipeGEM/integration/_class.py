from pipeGEM.integration.continuous.GIMME import apply_GIMME
from pipeGEM.integration.continuous.Eflux import apply_EFlux
from pipeGEM.integration.continuous.SPOT import apply_SPOT
from pipeGEM.integration.continuous.RIPTiDe import apply_RIPTiDe_pruning, apply_RIPTiDe_sampling
from pipeGEM.integration.algo.rFASTCORMICS import apply_rFASTCORMICS
from pipeGEM.integration.algo.CORDA import apply_CORDA
from pipeGEM.integration.algo.mCADRE import apply_mCADRE
from pipeGEM.integration.algo.MBA import apply_MBA
from pipeGEM.integration.algo.INIT import apply_INIT
from pipeGEM.integration.algo.iMAT import apply_iMAT
from pipeGEM.integration.algo.FASTCORE import apply_FASTCORE
from pipeGEM.integration.algo.SWIFTCORE import swiftCore
from pipeGEM.integration.utils import parse_predefined_threshold
from pipeGEM.analysis.results._base import BaseAnalysis
from pipeGEM.integration.ec.gecko_light import apply_gecko_light
from pipeGEM.integration.ec.gecko_full import apply_gecko_full
from pipeGEM.utils import ObjectFactory


class Integrators(ObjectFactory):
    def __init__(self):
        super().__init__()


class GeneDataIntegrator:
    def __init__(self):
        pass

    def integrate(self, model, data, **kwargs):
        raise NotImplementedError()


class EnzymeDataIntegrator:
    def __init__(self):
        pass

    def integrate(self, model, data, **kwargs):
        raise NotImplementedError()


class RemovableGeneDataIntegrator(GeneDataIntegrator):
    def __init__(self):
        super(RemovableGeneDataIntegrator, self).__init__()
        self._model = None
        self._applied_model = None

    def integrate(self, model, data, **kwargs):
        self._model = model
        raise NotImplementedError()

    def apply(self, model, data, **kwargs):
        if self._applied_model is not None:
            raise RuntimeError("Integrator is already applied; call remove() before applying it again.")
        self._model = model
        model.__enter__()
        self._applied_model = model
        try:
            return self.integrate(model, data, **kwargs)
        except BaseException as exc:
            self.remove(type(exc), exc, exc.__traceback__)
            raise

    def remove(self, exc=None, value=None, tb=None, **kwargs):
        if self._applied_model is None:
            return
        self._applied_model.__exit__(exc, value, tb)
        self._applied_model = None


class GIMME(RemovableGeneDataIntegrator):
    def __init__(self):
        super(GIMME, self).__init__()

    def integrate(self, model, data, **kwargs):
        """
        Integrate the given data with the model.

        Parameters
        ----------
        model: cobra.Model or pipeGEM.Model
            The model to be integrated with the data
        data: GeneData
            Gene data used to determine the objective function of GIMME
        kwargs: dict
            Keyword arguments passed to apply_GIMME

        Returns
        -------
        result: GIMMEAnalysis
        """
        self._model = model
        return apply_GIMME(model=self._model,
                           rxn_expr_score=data.rxn_scores,
                           **kwargs)


class EFlux(RemovableGeneDataIntegrator):
    def __init__(self):
        super().__init__()

    def integrate(self, model, data, **kwargs):
        self._model = model
        return apply_EFlux(model=self._model,
                           rxn_expr_score=data.rxn_scores,
                           **kwargs)


class SPOT(RemovableGeneDataIntegrator):
    def __init__(self):
        super().__init__()

    def integrate(self, model, data, **kwargs):
        self._model = model
        return apply_SPOT(model=self._model, rxn_expr_score=data.rxn_scores, **kwargs)


class RIPTiDePruning(GeneDataIntegrator):
    def __init__(self):
        super().__init__()

    def integrate(self, model, data, **kwargs):
        return apply_RIPTiDe_pruning(model=model,
                                     rxn_expr_score=data.rxn_scores,
                                     **kwargs)


class RIPTiDeSampling(RemovableGeneDataIntegrator):
    def __init__(self):
        super().__init__()

    def integrate(self, model, data, **kwargs):
        self._model = model
        return apply_RIPTiDe_sampling(model=self._model,
                                      rxn_expr_score=data.rxn_scores,
                                      **kwargs)


class RIPTiDe(GeneDataIntegrator):
    def __init__(self):
        super().__init__()

    def integrate(self, model, data, **kwargs):
        pr_result = apply_RIPTiDe_pruning(model=model,
                                          rxn_expr_score=data.rxn_scores,
                                          **kwargs)
        sp_result = apply_RIPTiDe_sampling(model=pr_result.result_model,
                                           rxn_expr_score=data.rxn_scores,
                                           **kwargs)
        return sp_result


class rFASTCORMICS(GeneDataIntegrator):
    def __init__(self):
        super().__init__()

    def integrate(self, model, data, **kwargs):
        return apply_rFASTCORMICS(model=model,
                                  data=data,
                                  **kwargs)


def _core_reaction_ids(model, data, C, protected_rxns, predefined_threshold):
    if C is None:
        thresholds = parse_predefined_threshold(predefined_threshold, gene_data=data.gene_data,
                                               threshold_type_if_none="percentile", p=[25, 75])
        C = {r for r, score in data.rxn_scores.items() if score > thresholds["exp_th"]}
    core = set(C) | set(protected_rxns or [])
    unknown = core - {r.id for r in model.reactions}
    if unknown:
        raise ValueError(f"Unknown core reaction IDs: {sorted(unknown)}")
    return core


class FASTCORE(GeneDataIntegrator):
    def integrate(self, model, data, C=None, nonP=None, epsilon=1e-6, return_model=True,
                  protected_rxns=None, predefined_threshold=None, rxn_scaling_coefs=None, **kwargs):
        core = _core_reaction_ids(model, data, C, protected_rxns, predefined_threshold)
        return apply_FASTCORE(C=core, nonP=[] if nonP is None else nonP, model=model, epsilon=epsilon,
                              return_model=return_model, rxn_scaling_coefs=rxn_scaling_coefs, **kwargs)


class SWIFTCORE(GeneDataIntegrator):
    def integrate(self, model, data, C=None, protected_rxns=None, predefined_threshold=None,
                  rxn_scaling_coefs=None, **kwargs):
        if rxn_scaling_coefs is not None:
            raise ValueError("SWIFTCORE does not support rxn_scaling_coefs")
        core = _core_reaction_ids(model, data, C, protected_rxns, predefined_threshold)
        indices = [i for i, r in enumerate(model.reactions) if r.id in core]
        result_model = swiftCore(model, core_index=indices, **kwargs)
        kept = {r.id for r in result_model.reactions}
        result = BaseAnalysis(log={"method": "SWIFTCORE"})
        result.set_result(
            result_model=result_model,
            kept_rxn_ids=sorted(kept), removed_rxn_ids=sorted({r.id for r in model.reactions} - kept),
        )
        return result


class CORDA(GeneDataIntegrator):
    def __init__(self):
        super().__init__()

    def integrate(self, model, data, **kwargs):
        return apply_CORDA(model=model,
                           data=data,
                           **kwargs)


class mCADRE(GeneDataIntegrator):
    def __init__(self):
        super().__init__()

    def integrate(self, model, data, **kwargs):
        return apply_mCADRE(model=model,
                            data=data,
                            **kwargs)


class MBA(GeneDataIntegrator):
    def __init__(self):
        super().__init__()

    def integrate(self, model, data, **kwargs):
        return apply_MBA(model=model,
                         data=data,
                         **kwargs)


class INIT(GeneDataIntegrator):
    def __init__(self):
        super().__init__()

    def integrate(self, model, data, **kwargs):
        return apply_INIT(model=model,
                          data=data,
                          **kwargs)


class iMAT(GeneDataIntegrator):
    def __init__(self):
        super().__init__()

    def integrate(self, model, data, **kwargs):
        return apply_iMAT(model=model,
                          data=data,
                          **kwargs)


class GECKOLight(EnzymeDataIntegrator):
    def integrate(self, model, data, **kwargs):
        return apply_gecko_light(model=model, enzyme_data=data, **kwargs)


class GECKOFull(EnzymeDataIntegrator):
    def integrate(self, model, data, **kwargs):
        return apply_gecko_full(model=model, enzyme_data=data, **kwargs)


integrator_factory = Integrators()
integrator_factory.register("GIMME", GIMME)
integrator_factory.register("EFlux", EFlux)
integrator_factory.register("SPOT", SPOT)
integrator_factory.register("FASTCORE", FASTCORE)
integrator_factory.register("SWIFTCORE", SWIFTCORE)
integrator_factory.register("RIPTiDePruning", RIPTiDePruning)
integrator_factory.register("RIPTiDeSampling", RIPTiDeSampling)
integrator_factory.register("RIPTiDe", RIPTiDe)
integrator_factory.register("rFASTCORMICS", rFASTCORMICS)
integrator_factory.register("CORDA", CORDA)
integrator_factory.register("mCADRE", mCADRE)
integrator_factory.register("MBA", MBA)
integrator_factory.register("INIT", INIT)
integrator_factory.register("iMAT", iMAT)

enzyme_integrator_factory = Integrators()
enzyme_integrator_factory.register("GECKOLight", GECKOLight)
enzyme_integrator_factory.register("GECKOFull", GECKOFull)
