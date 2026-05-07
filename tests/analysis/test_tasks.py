import cobra

from pipeGEM.analysis.tasks import TASKS_MOUSE_FILE_PATH, Task, TaskContainer, TaskHandler
from pipeGEM import Model


def test_bundled_mouse_task_path_exists():
    assert TASKS_MOUSE_FILE_PATH.is_file()


def test_task_handler_uses_input_knockout_type(monkeypatch):
    model = cobra.Model("boundary_knockout_test")
    glc = cobra.Metabolite("glc__D_e", formula="C6H12O6", compartment="e")
    o2 = cobra.Metabolite("o2_e", formula="O2", compartment="e")
    model.add_metabolites([glc, o2])
    model.add_boundary(glc, type="exchange", lb=-10, ub=1000)
    model.add_boundary(o2, type="exchange", lb=-10, ub=1000)

    task = Task(
        should_fail=False,
        in_mets=[],
        out_mets=[],
        knockout_input_flag=True,
        knockout_output_flag=False,
        ko_input_type="organic",
        ko_output_type="all",
    )
    handler = TaskHandler(model, TaskContainer({"organic_input": task}))
    observed_bounds = {}

    def fake_test_one_task(self, task, model, *args, **kwargs):
        observed_bounds["glc"] = model.reactions.EX_glc__D_e.lower_bound
        observed_bounds["o2"] = model.reactions.EX_o2_e.lower_bound
        return {"Passed": True, "Should fail": False, "Status": "not analyzed"}

    monkeypatch.setattr(TaskHandler, "test_one_task", fake_test_one_task)

    handler.test_tasks(get_support_rxns=False)

    assert observed_bounds == {"glc": 0, "o2": -10}


def test_task_handler(ecoli_core, ecoli_Tasks):
    model = Model(name_tag="ecoli", model=ecoli_core)
    model.add_tasks(name="task", tasks=ecoli_Tasks)
    task_analysis = model.test_tasks("task", model_compartment_parenthesis="_{}",
                                     solver="glpk")
    assert task_analysis is not None
    assert hasattr(task_analysis, "result")
    assert isinstance(task_analysis.result, dict)
