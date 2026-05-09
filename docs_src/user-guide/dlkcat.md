# DLKcat

DLKcat support is available under `pipeGEM.extensions.DLKcat` for kcat prediction workflows.

Install optional dependencies before importing the extension:

```bash
pip install -e ".[dlkcat]"
```

Then use the extension APIs:

```python
from pipeGEM.extensions.DLKcat import predict_Kcat

predictions = predict_Kcat(...)
```

The extension requires PyTorch and RDKit. Keep DLKcat imports isolated from general workflows if those optional dependencies are not installed.
