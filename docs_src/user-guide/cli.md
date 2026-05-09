# Command Line Interface

The `pipeGEM` executable provides pipeline-oriented workflows based on TOML configuration files.

## Subcommands

```bash
pipeGEM --help
pipeGEM template -p integration -o ./
pipeGEM process -t configs/model.toml
pipeGEM threshold -g configs/gene_data.toml -r configs/threshold.toml
pipeGEM integrate -g configs/gene_data.toml -t configs/model.toml -r configs/threshold.toml -m configs/mapping.toml -i configs/integration.toml
pipeGEM flux -f configs/flux_analysis/pFBA.toml -t configs/model.toml
pipeGEM compare -c configs/comparison.toml
```

Add `--dry-run` to validate configuration files and print the planned work without executing the pipeline.

## Configuration workflow

Start from generated templates instead of writing TOML from scratch:

```bash
pipeGEM template -p integration -o ./configs
```

Then edit paths and algorithm parameters in the generated files. Use relative paths when the configs will be committed with an analysis project, and absolute paths when running one-off local jobs.

Before a long run, validate the files:

```bash
pipeGEM integrate \
  -g configs/gene_data_conf.toml \
  -t configs/model_conf.toml \
  -r configs/threshold_conf.toml \
  -m configs/mapping_conf.toml \
  -i configs/integration_conf.toml \
  --dry-run
```

## Migration from legacy commands

Legacy calls using `-n` are still parsed:

```bash
pipeGEM -n integration -g configs/gene_data.toml -t configs/model.toml
```

New automation should use subcommands because the `-n` style is deprecated.

See [CLI API](../api/cli.md).
