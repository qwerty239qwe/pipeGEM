from ._builders import (ModelGroupBuilders, BatchBuilders, GroupBuilder,
                        ComplementBatchBuilder, ComparableBatchBuilder,
                        get_group)
from functools import partial


_registered_batches = {"complement": ComplementBatchBuilder,
                       "comparable": ComparableBatchBuilder}
_registered_groups = {"named": GroupBuilder}

groups = ModelGroupBuilders()
batches = BatchBuilders()

for name, builder in _registered_batches.items():
    batches.register(name, builder())

for name, builder in _registered_groups.items():
    groups.register(name, builder())


_complement_batch = batches.get("complement")  # only init once

group = partial(get_group, builder=groups, complement_batch=_complement_batch)