# pipeGEM

pipeGEM provides Python and CLI workflows for working with genome-scale metabolic models. It wraps common model operations, expression-data integration, medium handling, metabolic tasks, flux analysis, and model comparison around COBRA models.

## Where to start

- New installation: [Installation](getting-started/installation.md)
- First model workflow: [Quickstart](getting-started/quickstart.md)
- Notebook walkthroughs: [Tutorials](tutorials/index.md)
- Task-oriented reference: [User guide](user-guide/model-group.md)
- Python objects and functions: [API reference](api/index.md)

## Common workflows

1. Load a COBRA model with `pipeGEM.load_model`.
2. Wrap it in `pipeGEM.Model` for annotation, data, task, and analysis helpers.
3. Add gene expression data with `pipeGEM.data.GeneData`.
4. Add medium constraints with `pipeGEM.data.MediumData` or the built-in medium catalog.
5. Run thresholding, model extraction, task testing, or flux analysis.
6. Compare multiple models with `pipeGEM.Group`.

## What changed from the old docs?

The Sphinx documentation has moved to MkDocs Material. The old topics are still present, but the navigation is organized by workflow:

- `quickstart.html` is now [Getting Started > Quickstart](getting-started/quickstart.md).
- `tutorial/index.html` is now [Tutorials](tutorials/index.md).
- `api/index.html` is now [API Reference](api/index.md).
- Sphinx internals such as `genindex`, `modindex`, and `_sources` are no longer generated.

The CLI examples now use the current subcommand form, such as `pipeGEM template`, `pipeGEM process`, and `pipeGEM integrate`. The old `-n <pipeline>` form remains available for compatibility but is deprecated.

## Build these docs locally

The documentation build is driven by the `doc` optional dependency group in `pyproject.toml`:

```bash
uv run --extra doc mkdocs build --strict -d ./docs
```

Read the Docs uses the same command and publishes the generated HTML from its configured output directory.
