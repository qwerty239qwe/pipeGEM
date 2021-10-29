# from ._builders import (ModelBuilders, ModelBuilder,
#                         ModelGroupBuilders, GroupBuilder, ComplementGroupBuilder,
#                         BatchBuilders, ComplementBatchBuilder, ComparableBatchBuilder,
#                         get_group, get_batch, get_model,
#                         NameManager)
# from functools import partial
#
#
# _registered_batches = {"complement": ComplementBatchBuilder,
#                        "batch": ComparableBatchBuilder}
# _registered_groups = {"group": GroupBuilder,
#                       "complement": ComplementGroupBuilder}
# _registered_models = {"model": ModelBuilder}
#
# models = ModelBuilders()
# groups = ModelGroupBuilders()
# batches = BatchBuilders()
#
# for name, builder in _registered_batches.items():
#     batches.register(name, builder())
#
# _name_manager = NameManager()
# _complement_batch = batches.get("complement", name_manager=_name_manager)  # only init once
#
# for name, builder in _registered_groups.items():
#     groups.register(name, builder())
#
# for name, builder in _registered_models.items():
#     models.register(name, builder())
#
#
# group = partial(get_group, name_manager=_name_manager, builders=groups, builder_name="group",
#                 complement_batch=_complement_batch)
# batch = partial(get_batch, name_manager=_name_manager, builders=batches, builder_name="batch",
#                 complement_batch=_complement_batch)
# model = partial(get_model, name_manager=_name_manager, builders=models, builder_name="model",
#                 complement_batch=_complement_batch, complement_group=_complement_batch.complement_group)