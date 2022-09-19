from pipeGEM.integration.continuous.GIMME import apply_GIMME
from pipeGEM.integration.continuous.Eflux import apply_EFlux
from pipeGEM.integration.continuous.SPOT import apply_SPOT
from pipeGEM.integration.algo.rFASTCORMICS import apply_rFASTCORMICS
from pipeGEM.integration.continuous.RIPTiDe import apply_RIPTiDe_pruning, apply_RIPTiDe_sampling
from pipeGEM.integration.algo.CORDA import apply_CORDA
from pipeGEM.utils import ObjectFactory


class Integrators(ObjectFactory):
    def __init__(self):
        super().__init__()


class GeneDataIntegrator:
    def __init__(self, model):
        self._model = model

    def integrate(self, data, **kwargs):
        raise NotImplementedError()


class RemovableGeneDataIntegrator(GeneDataIntegrator):
    def __init__(self, model):
        super(RemovableGeneDataIntegrator, self).__init__(model)

    def integrate(self, data, **kwargs):
        raise NotImplementedError()

    def apply(self, **kwargs):
        self._model.__enter__()
        self.integrate(**kwargs)

    def remove(self, exc, value, tb, **kwargs):
        self._model.__exit__(exc, value, tb)


class GIMME(RemovableGeneDataIntegrator):
    def __init__(self, model):
        super(GIMME, self).__init__(model)

    def integrate(self, data, **kwargs):
        return apply_GIMME(model=self._model, rxn_expr_score=data.rxn_scores, **kwargs)


class EFlux(RemovableGeneDataIntegrator):
    def __init__(self, model):
        super().__init__(model)

    def integrate(self, data, **kwargs):
        return apply_EFlux(model=self._model, rxn_expr_score=data.rxn_scores, **kwargs)


class SPOT(RemovableGeneDataIntegrator):
    def __init__(self, model):
        super().__init__(model)

    def integrate(self, data, **kwargs):
        return apply_SPOT(model=self._model, rxn_expr_score=data.rxn_scores, **kwargs)


class RIPTiDePruning(GeneDataIntegrator):
    def __init__(self, model):
        super().__init__(model)

    def integrate(self, data, **kwargs):
        return apply_RIPTiDe_pruning(model=self._model, rxn_expr_score=data.rxn_scores, **kwargs)


class RIPTiDeSampling(RemovableGeneDataIntegrator):
    def __init__(self, model):
        super().__init__(model)

    def integrate(self, data, **kwargs):
        return apply_RIPTiDe_sampling(model=self._model, rxn_expr_score=data.rxn_scores, **kwargs)


class RIPTiDe(GeneDataIntegrator):
    def __init__(self, model):
        super().__init__(model)

    def integrate(self, data, **kwargs):
        pr_result = apply_RIPTiDe_pruning(model=self._model, rxn_expr_score=data.rxn_scores, **kwargs)
        sp_result = apply_RIPTiDe_sampling(model=pr_result.model, rxn_expr_score=data.rxn_scores, **kwargs)
        return sp_result


class rFASTCORMICS(GeneDataIntegrator):
    def __init__(self, model):
        super().__init__(model)

    def integrate(self, data, **kwargs):
        return apply_rFASTCORMICS(model=self._model,
                                  data=data,
                                  **kwargs)


class CORDA(GeneDataIntegrator):
    def __init__(self, model):
        super().__init__(model)

    def integrate(self, data, **kwargs):
        return apply_CORDA(model=self._model,
                           data=data,
                           **kwargs)



integrator_factory = Integrators()
integrator_factory.register("GIMME", GIMME)
integrator_factory.register("EFlux", EFlux)
integrator_factory.register("SPOT", SPOT)
integrator_factory.register("RIPTiDePruning", RIPTiDePruning)
integrator_factory.register("RIPTiDeSampling", RIPTiDeSampling)
integrator_factory.register("RIPTiDe", RIPTiDe)
integrator_factory.register("rFASTCORMICS", rFASTCORMICS)
integrator_factory.register("CORDA", CORDA)