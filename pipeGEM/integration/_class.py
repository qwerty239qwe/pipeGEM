from pipeGEM.integration.continuous.GIMME import apply_GIMME
from pipeGEM.integration.continuous.Eflux import apply_EFlux
from pipeGEM.integration.continuous.SPOT import apply_SPOT
from pipeGEM.integration.continuous.RIPTiDe import apply_RIPTiDe_pruning, apply_RIPTiDe_sampling
from pipeGEM.integration.algo.rFASTCORMICS import apply_rFASTCORMICS
from pipeGEM.integration.algo.CORDA import apply_CORDA
from pipeGEM.integration.algo.mCADRE import apply_mCADRE
from pipeGEM.utils import ObjectFactory


class Integrators(ObjectFactory):
    def __init__(self):
        super().__init__()


class GeneDataIntegrator:
    def __init__(self):
        pass

    def integrate(self, model, data, **kwargs):
        raise NotImplementedError()


class RemovableGeneDataIntegrator(GeneDataIntegrator):
    def __init__(self):
        super(RemovableGeneDataIntegrator, self).__init__()
        self._model = None

    def integrate(self, model, data, **kwargs):
        self._model = model
        raise NotImplementedError()

    def apply(self, **kwargs):
        self._model.__enter__()
        self.integrate(**kwargs)

    def remove(self, exc, value, tb, **kwargs):
        self._model.__exit__(exc, value, tb)


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


integrator_factory = Integrators()
integrator_factory.register("GIMME", GIMME)
integrator_factory.register("EFlux", EFlux)
integrator_factory.register("RIPTiDePruning", RIPTiDePruning)
integrator_factory.register("RIPTiDeSampling", RIPTiDeSampling)
integrator_factory.register("RIPTiDe", RIPTiDe)
integrator_factory.register("rFASTCORMICS", rFASTCORMICS)
integrator_factory.register("CORDA", CORDA)
integrator_factory.register("mCADRE", mCADRE)
