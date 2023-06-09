from textwrap import dedent


_docs = {"template": dedent("""
    template:
        Generate template toml files for pipelines' configs. 
        Name of the pipeline is needed."""),
         "integration": dedent("""
    integration: 
        Perform complete gene data integration pipeline.
        This needs users to determine gene_data, model_testing, 
        threshold, mapping, and integration configs."""),
         "model_processing": dedent("""
    model_processing:
        Perform model testing pipeline.
        This needs users to determine model_testing configs."""),


         }


def get_help_doc(names):
    if names == "all":
        names = list(_docs.keys())

    return "\n\n".join([_docs[i] for i in names])
