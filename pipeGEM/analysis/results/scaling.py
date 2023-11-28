from ._base import *
import scipy
from tqdm import tqdm


class ModelScalingResult(BaseAnalysis):
    def __init__(self, log):
        super(ModelScalingResult, self).__init__(log=log)
        self._result_saving_params["diff_A"] = {"fm_name": "SparseArrayFloat"}
        self._result_saving_params["decimals"] = {"fm_name": "SparseArrayFloat"}

    def reverse_scaling(self, model):
        model = model.copy()
        rxn_ids_in_model = [r.id for r in model.reactions]
        met_ids_in_model = [m.id for m in model.metabolites]

        for ri, rid in tqdm(enumerate(self._result["rxn_index"])):
            if rid not in rxn_ids_in_model:
                continue
            rxn = model.reactions.get_by_id(rid)
            involved_met_ids = scipy.sparse.find(self._result["diff_A"][:, ri] != 0)[0]

            assert all([self._result["met_index"][mi] in met_ids_in_model
                        for mi in involved_met_ids]), \
                "This method assumes all the metabolites in this model's reactions found in the rescaling step are still in the model."\
                f"\nMissing mets of {rxn.id}: {[mid for mi, mid in enumerate(self._result['met_index']) if mi in involved_met_ids if mid not in met_ids_in_model]}"

            rxn.add_metabolites({
                self._result["met_index"][mi]: np.round(rxn.metabolites[model.metabolites.get_by_id(self._result["met_index"][mi])] -
                                                        self._result["diff_A"][mi, ri], decimals=self._result["decimals"][mi, ri])
                for mi in involved_met_ids
            }, combine=False)

            rxn.lower_bound /= self._result["rxn_scaling_factor"][rid]
            rxn.upper_bound /= self._result["rxn_scaling_factor"][rid]

        return model