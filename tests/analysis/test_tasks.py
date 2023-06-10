from pipeGEM.analysis.tasks import TaskHandler
from pipeGEM import Model


def test_task_handler(ecoli_core, ecoli_Tasks):
    model = Model(name_tag="ecoli", model=ecoli_core)
    model.add_tasks(name="task", tasks=ecoli_Tasks)
    task_analysis = model.test_tasks("task", model_compartment_parenthesis="_{}",
                                     solver="glpk")
    task_analysis.save("./task")
