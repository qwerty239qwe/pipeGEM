from textwrap import dedent


PROGRAM_DESCRIPTION = dedent("""\
    pipeGEM — Processing and integrating data with genome-scale metabolic models (GEM).

    Available subcommands:
      template    Generate template TOML config files for a pipeline
      integrate   Run the full gene data integration pipeline
      process     Run model processing (rescaling, consistency, task testing)
      threshold   Compute expression thresholds from gene data
      flux        Run flux analysis on a group of models
      compare     Compare a group of metabolic models

    Use 'pipeGEM <subcommand> --help' for details on each subcommand.
""")


_docs = {
    "template": dedent("""\
        Generate template TOML config files for a pipeline.

        Required: -p/--pipeline (pipeline name), -o/--output (output directory).
        Example:  pipeGEM template -p integration -o ./configs/"""),

    "integration": dedent("""\
        Run the full gene data integration pipeline.

        Required: -g (gene data config), -t (model config), -r (threshold config),
                  -m (mapping config), -i (integration config).
        Example:  pipeGEM integrate -g gene.toml -t model.toml -r thresh.toml -m map.toml -i int.toml"""),

    "model_processing": dedent("""\
        Run model processing: rescaling, consistency testing, and metabolic task testing.

        Required: -t (model config).
        Example:  pipeGEM process -t model.toml"""),

    "get_threshold": dedent("""\
        Compute expression thresholds from gene data.

        Required: -g (gene data config), -r (threshold config).
        Example:  pipeGEM threshold -g gene.toml -r thresh.toml"""),

    "do_flux_analysis": dedent("""\
        Run flux analysis on a group of models.

        Required: -f (flux analysis config), -t (multi-model config).
        Optional: -g (gene data), -r (threshold), -m (mapping), -i (integration).
        Example:  pipeGEM flux -f flux.toml -t model.toml"""),

    "do_model_comparison": dedent("""\
        Compare a group of metabolic models (component counts, Jaccard, PCA).

        Required: -c (comparison config).
        Example:  pipeGEM compare -c comparison.toml"""),
}


def get_help_doc(names):
    if names == "all":
        names = list(_docs.keys())

    return "\n\n".join([_docs[i] for i in names])
