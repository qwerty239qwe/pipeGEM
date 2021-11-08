
def calc_jaccard_index(model_a, model_b, components='all'):
    if components == 'all':
        components = ['genes', 'reactions', 'metabolites']
    union_components = {'genes': set(model_a.gene_ids) |
                                 set(model_b.gene_ids),
                        'reactions': set(model_a.reaction_ids) |
                                     set(model_b.reaction_ids),
                        'metabolites': set(model_a.metabolite_ids) |
                                       set(model_b.metabolite_ids)}
    intersect_components = {'genes': set(model_a.gene_ids) &
                                     set(model_b.gene_ids),
                            'reactions': set(model_a.reaction_ids) &
                                         set(model_b.reaction_ids),
                            'metabolites': set(model_a.metabolite_ids) &
                                           set(model_b.metabolite_ids)}

    return sum([len(intersect_components[c]) for c in components]) / \
           sum([len(union_components[c]) for c in components])