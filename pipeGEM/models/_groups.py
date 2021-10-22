from ._models import ModelGroup, Batch


class Group(ModelGroup):

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, batch):
        assert isinstance(batch, Batch)
        self._batch = batch
        for mod in self._named_models:
            mod.batch = batch

    def _add_model(self, model):
        super(Group, self)._add_model(model)
        model.group = self
        model.batch = self._batch

    def leave_batch(self):
        super().leave_batch()
        for mod in self._named_models:
            mod.leave_batch()


class VirtualGroup(ModelGroup):
    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, batch):
        assert isinstance(batch, Batch)
        self._batch = batch


class ComplementGroup(ModelGroup):
    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, batch):
        assert isinstance(batch, Batch)
        self._batch = batch
        for mod in self._named_models:
            mod.batch = batch