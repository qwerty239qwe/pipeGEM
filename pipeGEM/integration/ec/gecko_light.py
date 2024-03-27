
from pipeGEM.analysis import timing


def _check_gene_and_enzymes(model, enzyme_data):
    gid_in_enzyme_data = {g.id: g.id in enzyme_data for g in model.genes}

    return [g for g, isin in gid_in_enzyme_data.items() if not isin]


@timing
def apply_gecko_light(model,
                      enzyme_data):
    genes_not_in_enzyme_data = _check_gene_and_enzymes(model, enzyme_data)
    if len(genes_not_in_enzyme_data) != 0:
        raise ValueError(f"Genes {genes_not_in_enzyme_data} are not in the enzyme data. "
                         f"Please add them to the enzyme data first.")