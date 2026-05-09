# Metabolic Tasks

Metabolic tasks describe functions a model should be able to perform, such as producing a metabolite or carrying flux through support reactions.

The main task classes are available under `pipeGEM.analysis.tasks`:

- `Task`
- `TaskContainer`
- `TaskHandler`

```python
from pipeGEM.analysis.tasks import TaskContainer

tasks = TaskContainer.load("tasks.json")
model.add_tasks(tasks)
result = model.test_tasks()
```

Task results help identify missing capabilities, compare model variants, and guide gap filling.

See [Tasks API](../api/tasks.md).
