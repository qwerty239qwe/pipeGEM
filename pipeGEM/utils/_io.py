import re
from pathlib import Path
from typing import List, Any, Union, Optional
from os import PathLike
from warnings import warn
from functools import reduce

import pandas as pd
import numpy as np
import cobra
import cobra.manipulation
from cobra.io.mat import create_mat_dict, _cell

try:
    import scipy.sparse as scipy_sparse
    import scipy.io as scipy_io
except ImportError:
    scipy_sparse = None
    scipy_io = None


__all__ = ("model_to_excel", "model_to_mat", "load_test_model", "save_model", "load_model")


def model_to_excel(model, file_name):
    rxns = pd.DataFrame({"ID": [r.id for r in model.reactions],
                         "name": [r.name for r in model.reactions],
                         "reaction": [r.reaction for r in model.reactions],
                         "upper_bound": [r.upper_bound for r in model.reactions],
                         "lower_bound": [r.lower_bound for r in model.reactions],
                         "subsystem": [r.subsystem for r in model.reactions],
                         "GPR": [r.gene_reaction_rule for r in model.reactions],
                         "is_objective": [r.objective_coefficient for r in model.reactions],})

    mets = pd.DataFrame({"ID": [m.id for m in model.metabolites],
                         "formula": [m.formula for m in model.metabolites],
                         "name": [m.name for m in model.metabolites],
                         "compartment": [m.compartment for m in model.metabolites],
                         "charge": [m.charge for m in model.metabolites],
                         "reaction": [', '.join([r.id for r in m.reactions]) for m in model.metabolites],})
    gens = pd.DataFrame({"ID": [g.id for g in model.genes],
                         "name": [g.name for g in model.genes]})

    writer = pd.ExcelWriter(file_name)
    gens.to_excel(writer, sheet_name='Genes')
    mets.to_excel(writer, sheet_name='Metabolites')
    rxns.to_excel(writer, sheet_name='Reactions')
    writer.save()


def _grr_to_rule_field(unified_grr, gene_name_map_to_idx):
    pattern = re.compile(r'\d+')
    rep_str = re.sub(r"and", '&', re.sub(r"or", '|', unified_grr))
    rep_str = re.sub(pattern, lambda x: gene_name_map_to_idx.get(x.group()), rep_str)
    return rep_str


# modified function that convert cobra.model to matlab file
def model_to_mat(model, file_name, varname=None):
    """Save the cobra model as a .mat file.

    This .mat file can be used directly in the MATLAB version of COBRA.

    Parameters
    ----------
    model : cobra.core.Model.Model object
        The model to save
    file_name : str or file-like object
        The file to save to
    varname : string
       The name of the variable within the workspace
    """
    if not scipy_io:
        raise ImportError('load_matlab_model requires scipy')

    if varname is None:
        varname = str(model.id) \
            if model.id is not None and len(model.id) > 0 \
            else "exported_model"
    mat = create_mat_dict(model)
    gene_name_map_to_idx = {g.id: f'x({i + 1})' for i, g in enumerate(model.genes)}
    mat['rules'] = _cell([_grr_to_rule_field(grr, gene_name_map_to_idx)
                          for grr in model.reactions.list_attr('gene_reaction_rule')])
    scipy_io.savemat(file_name, {varname: mat},
                     appendmat=True, oned_as="column")


def load_test_model(name: str = "e_coli_core.xml",
                    use_gene_level: bool = True,
                    model_path: Optional[PathLike] = None) -> cobra.Model:
    """
    Get a cobra.Model in the core folder

    Parameters
    ----------
    name: str
        file name of the cobra.Model
    use_gene_level: bool
        If true, use gene entrez id (without transcript part)
    model_path: Optional[PathLike]
        The path of model directory, default: ../../assets/core from the folder the package is located

    Returns
    -------
    cobra.Model
        The requested cobra.Model
    """
    fn_path = Path(name)
    if model_path is None:
        model_path = Path(__file__).parent.parent / Path("assets/core")
    if fn_path.suffix == ".xml":
        mod = cobra.io.read_sbml_model(str(model_path / fn_path))
    elif fn_path.suffix == ".mat":
        mod = cobra.io.load_matlab_model(model_path / fn_path)
    else:
        raise ValueError()
    if use_gene_level:
        cobra.manipulation.modify.rename_genes(mod, {g.id: g.id[:g.id.index(".")] if "." in g.id else g.id
                                                     for g in mod.genes})
    return mod


def save_model(model: cobra.Model,
               output_file_name: Union[str, PathLike],
               output_extension: str) -> None:
    """
    Save a cobra.Model

    Parameters
    ----------
    model: cobra.Model
        Saved cobra.Model
    output_file_name: Union[str, PathLike]
        Saved core' file name
    output_extension: str
        File extension, choices: .mat, .json, .xml, .yaml
    Returns
    -------
    None
    """
    if isinstance(output_file_name, str):
        output_file_name = Path(output_file_name)

    if output_extension == ".mat":
        cobra.io.save_matlab_model(model, output_file_name.with_suffix(output_extension))
    elif output_extension == ".json":
        cobra.io.json.save_json_model(model, output_file_name.with_suffix(output_extension))
    elif output_extension == ".xml":
        cobra.io.write_sbml_model(model, output_file_name.with_suffix(output_extension))
    elif output_extension == ".yaml":
        cobra.io.yaml.save_yaml_model(model, output_file_name.with_suffix(output_extension))
    else:
        raise ValueError(f"Invalid file extension: {output_extension}")


def load_model(model_file_path: str) -> cobra.Model:
    """

    Parameters
    ----------
    model_file_path

    Returns
    -------
    cobra.Model
    """
    fn_path = Path(model_file_path)
    if fn_path.suffix == ".xml":
        return cobra.io.read_sbml_model(str(fn_path))
    elif fn_path.suffix == ".mat":
        return cobra.io.load_matlab_model(fn_path)
    elif fn_path.suffix == ".json":
        return cobra.io.load_json_model(fn_path)
    elif fn_path.suffix == ".yaml":
        return cobra.io.load_yaml_model(fn_path)
    else:
        raise ValueError(f"Invalid file extension: {fn_path.suffix}")


def sheet_to_comp(model, excel_file_name, raise_err=False):
    mets = pd.read_excel(excel_file_name, sheet_name="metabolites")
    rxns = pd.read_excel(excel_file_name, sheet_name="reactions")

    exist_mets = [m.id for m in model.metabolites]
    exist_rxns = [m.id for m in model.metabolites]
    added_mets = []
    added_rxns = []
    for m in mets.iterrows():
        if raise_err:
            assert m[1]['ID'] not in exist_mets, f"The metabolite {m[1]['ID']} is already in the model."
        new_met = cobra.Metabolite(id=m[1]["ID"],
                                   formula=m[1]["formula"],
                                   name=m[1]["name"],
                                   compartment=m[1]["compartment"])
        if new_met in exist_mets:
            warn(f"The metabolite {m[1]['ID']} is already in the model.")
        else:
            added_mets.append(new_met)
    model.add_metabolites(added_mets)

    for r in rxns.iterrows():
        if raise_err:
            assert r[1]['ID'] not in exist_mets, f"The reaction {r[1]['ID']} is already in the model."
        added_rxn = cobra.Reaction(id=r[1]["ID"],
                                   name=r[1]["name"],
                                   lower_bound=r[1]["lower_bound"],
                                   upper_bound=r[1]["upper_bound"])
        GPR = str(r[1]["GPR"]) if r[1]["GPR"] is not None else ""
        added_rxn.gene_reaction_rule = GPR
        if added_rxn in exist_rxns:
            warn(f"The reaction {r[1]['ID']} is already in the model.")
        else:
            added_rxns.append(added_rxn)
            model.add_reactions([added_rxn])

            added_rxn.build_reaction_from_string(r[1]["reaction"])