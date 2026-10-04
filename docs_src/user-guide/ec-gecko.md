# Enzyme-constrained models (GECKO)

GECKO limits reaction flux using enzyme kinetics and protein availability.

| Method | Constraints | Use when |
| --- | --- | --- |
| `GECKOLight` | Independent capacity for each reaction, in both directions | You want simple flux bounds |
| `GECKOFull` | Shared enzyme budgets and a total protein pool | Several reactions compete for the same protein |

## Prepare the data

`enzymes.csv` uses gene IDs as its first column. `Reaction` must match model reaction IDs; genes must match their reaction's GPR. Replace these sample IDs with yours.

```csv
gene,Protein,Reaction,Kcat,MW
g1,P1,R1,1.0,50000
g2,P2,R2,2.0,72000
```

`proteins.csv` contains one abundance per protein ID:

```csv
protein,abundance
P1,0.0001
P2,0.0001
```

Units: **Kcat in s⁻¹**, **MW in Da (g/mol)**, **abundance in mmol/gDW**. Protein IDs must match between files; duplicate abundance IDs raise `ValueError`.

## Build and solve

```python
import cobra
import pandas as pd
from pipeGEM import Model
from pipeGEM.data import EnzymeData
from pipeGEM.data.data import ProteinAbundanceData

model = Model(model=cobra.io.read_sbml_model("model.xml"))
enzymes = EnzymeData(
    pd.read_csv("enzymes.csv", index_col=0),
    prot_id_col="Protein", rxn_id_col="Reaction",
)
model.add_enzyme_data(enzymes, run_DLKcat=False)
model.add_protein_abundance_data(
    "measured", ProteinAbundanceData(pd.read_csv("proteins.csv", index_col=0)),
)
result = model.integrate_enzyme_data(
    "measured", method="GECKOFull", sigma=0.5, ptot=0.5, f_factor=0.5,
)
solution = result.ec_model.optimize()
print(solution.status, solution.objective_value)
result.save("gecko_result")
```

Change `method` to `"GECKOLight"` for independent bounds. Both methods copy the model by default. Full GECKO renames constrained reversible reactions with `_F_` and `_R_` prefixes.

## Interpret the constraints

- **Light:** each reaction receives `kcat × abundance × sigma × 3600`, even when several reactions use the same protein.
- **Full:** reactions share a protein draw capped at `abundance × sigma`. Total pool capacity is `ptot × f_factor × sigma`; `ptot` is in g/gDW and `f_factor` is the metabolic-enzyme fraction.
- **Missing abundance:** omit `"measured"` from the integration call to use fallback constraints. Missing IDs or NaN also use the fallback: the total pool for full GECKO, or the coarse `ptot × f_factor` abundance scale for light.
- **Zero abundance:** prevents enzyme usage. Light raises `ValueError` if the resulting capacity conflicts with mandatory flux; full GECKO may solve as infeasible.

Alignment retains finite positive kcats and molecular weights, selecting the lowest `MW/kcat` enzyme per reaction. Alternative isoenzymes and enzyme complexes are not explicitly represented.

## Fill missing kcats

Before `model.add_enzyme_data(...)`, optionally run:

```python
from pipeGEM.integration.ec import auto_parameterize

enzymes = auto_parameterize(model, enzymes, fill_missing="median")
```

Use `"geometric_mean"` for geometric filling. Invalid values count as missing; fills use finite positive donors. With no valid donors, the logged fallback is **1.0 s⁻¹**, an assumed placeholder.
