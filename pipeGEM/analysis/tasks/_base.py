import json
import re
from typing import Dict, Any, Union, List, Optional, Literal
from pathlib import Path

import pandas as pd
import numpy as np
import cobra
from cobra.flux_analysis.parsimonious import pfba
from cobra.exceptions import Infeasible, OptimizationError

from pipeGEM.utils import get_organic_exs
from pipeGEM.analysis import flux_analyzers, TaskAnalysis
from ._var import TASKS_FILE_PATH


class Task:
    """
    Single metabolic task.

    Attributes
    ----------
    system : str
        The name of system that this task is in.
        Ex: Energy metabolism
    subsystem : str
        The name of subsystem that this task is in.
        Ex: OXYDATIVE PHOSPHORYLATION
    description : str
        A brief description about this task
    should_fail : bool
        Does this task should fail to pass or not.
    in_mets : List[Dict[str, Any]]
        The constraints(lower_bound, upper_bound) of influx metabolites
        Example :
            {'glc_D': {'lb': 0, 'ub': 1000, 'compartment': 'c'}, 'ATP': {'lb': 0, 'ub': 1000, 'compartment': 'c'}}
    out_mets : List[Dict[str, Any]]
        The constraints(lower_bound, upper_bound) of outflux metabolites
        Example :
            {'glc_D': {'lb': 0, 'ub': 1000, 'compartment': 'c'}, 'ATP': {'lb': 0, 'ub': 1000, 'compartment': 'c'}}
    """
    def __init__(self,
                 should_fail: bool,
                 in_mets: List[Dict[str, Any]],
                 out_mets: List[Dict[str, Any]],
                 knockout_input_flag: bool = True,
                 knockout_output_flag: bool = True,
                 ko_input_type: Literal["all", "organic"] = "all",
                 ko_output_type: Literal["all", "organic"] = "all",
                 compartment_parenthesis: str = "[{}]",
                 met_id_str: str = "met_id",
                 lower_bound_str: str = "lb",
                 upper_bound_str: str = "ub",
                 compartment_str: str = "compartment",
                 system: str = None,
                 subsystem: str = None,
                 description: str = None,
                 annotation: str = None):
        """
        Initialize a metabolic task.

        Parameters
        ----------
        should_fail : bool
            Indicates if the task is expected to fail (e.g., cannot carry flux).
        in_mets : List[Dict[str, Any]]
            List of dictionaries defining input metabolites and their bounds.
        out_mets : List[Dict[str, Any]]
            List of dictionaries defining output metabolites and their bounds.
        knockout_input_flag : bool, optional
            Whether to knock out (set bounds to 0) boundary reactions for inputs
            not specified in `in_mets`. Default is True.
        knockout_output_flag : bool, optional
            Whether to knock out boundary reactions for outputs not specified
            in `out_mets`. Default is True.
        ko_input_type : Literal["all", "organic"], optional
            Type of input boundary reactions to knock out ('all' or 'organic').
            Default is "all".
        ko_output_type : Literal["all", "organic"], optional
            Type of output boundary reactions to knock out ('all' or 'organic').
            Default is "all".
        compartment_parenthesis : str, optional
            Format string for compartment identifiers (e.g., "[{}]"). Default is "[{}]".
        met_id_str : str, optional
            Dictionary key for metabolite ID within `in_mets`/`out_mets`. Default is "met_id".
        lower_bound_str : str, optional
            Dictionary key for lower bound. Default is "lb".
        upper_bound_str : str, optional
            Dictionary key for upper bound. Default is "ub".
        compartment_str : str, optional
            Dictionary key for compartment ID. Default is "compartment".
        system : str, optional
            Broad functional category of the task. Default is None.
        subsystem : str, optional
            Specific metabolic pathway or subsystem. Default is None.
        description : str, optional
            Textual description of the task. Default is None.
        annotation : str, optional
            Additional annotations for the task. Default is None.
        """
        self.system = system
        self.subsystem = subsystem
        self.description = description
        self.should_fail = should_fail
        self.in_mets, self.out_mets = in_mets, out_mets
        self.annotation = annotation
        self.compartment_parenthesis = compartment_parenthesis
        self.met_id_str = met_id_str
        self.lower_bound_str = lower_bound_str
        self.upper_bound_str = upper_bound_str
        self.compartment_str = compartment_str
        self.knockout_input_flag, self.knockout_output_flag = knockout_input_flag, knockout_output_flag
        self.ko_input_type, self.ko_output_type = ko_input_type, ko_output_type

    def to_dict(self) -> dict:
        """Convert the Task object attributes to a dictionary."""
        return {"system": self.system,
                "subsystem": self.subsystem,
                "description": self.description,
                "annotation": self.annotation,
                "in_mets": self.in_mets,
                "out_mets": self.out_mets,
                "knockout_input_flag": self.knockout_input_flag,
                "knockout_output_flag": self.knockout_output_flag,
                "should_fail": self.should_fail,
                }

    def __str__(self):
        des = f"system : {self.system}\n" \
              f"subsystem : {self.subsystem}\n" \
              f"description : {self.description}\n" \
              f"should fail : {self.should_fail}"

        return des + \
               '\n----- input metabolites -----\n' + self._make_table("in") + \
               '\n----- output metabolites -----\n' + self._make_table("out")

    def _make_table(self, which="in"):
        met_len = max(max([len(m[self.met_id_str]) for m in (self.in_mets if which == "in" else self.out_mets)]),
                      6)
        lb_len = max([len(str(int(m[self.lower_bound_str])))
                      for m in (self.in_mets if which == "in" else self.out_mets)]) + 4
        ub_len = max([len(str(int(m[self.upper_bound_str])))
                      for m in (self.in_mets if which == "in" else self.out_mets)]) + 4
        comp_len = max(max([len(m[self.compartment_str]) for m in (self.in_mets if which == "in" else self.out_mets)]),
                       4)
        pad = 3
        rows = [f"|{met[self.met_id_str]:>{met_len+pad}s}" \
                f"|{met[self.lower_bound_str]:>{lb_len+pad}.3f}" \
                f"|{met[self.upper_bound_str]:>{ub_len+pad}.3f}" \
                f"|{met[self.compartment_str]:>{comp_len+pad}s}|"
                for met in (self.in_mets if which == "in" else self.out_mets)]

        columns = ["Met ID", "LB", "UB", "comp"]
        header = f"|{columns[0]:>{met_len+pad}s}|{columns[1]:>{lb_len+pad}s}|{columns[2]:>{ub_len+pad}s}|{columns[3]:>{comp_len+pad}s}|"
        body = "\n".join(rows)
        horiz_line = "\n" + "-" * len(header) + "\n"
        return horiz_line + header+ horiz_line +body + horiz_line

    def _substitute_compartment(self, met_id, compr):
        cp = self.compartment_parenthesis
        return met_id + cp.format(compr)

    def mets_in_model(self,
                      model,
                      all_mets_in_model = None) -> bool:
        """
        A function to check if all the tasks' metabolites are in the model

        Parameters
        ----------
        model: cobra.Model or pg.Model
            Checked metabolic model
        all_mets_in_model: optional, list of str
            All the metabolites' ID. Input to prevent to traverse the metabolite list in the model repeatedly.
        Returns
        -------
        are_in_model: bool
            A bool indicates if the metabolites are all in the model

        """
        if not all_mets_in_model:
            all_mets_in_model = [m.id for m in model.metabolites]
        all_fine = True
        for met in self.in_mets:
            met = f"{met[self.met_id_str]}{self.compartment_parenthesis[0]}{met[self.compartment_str]}{self.compartment_parenthesis[1]}"
            if met not in all_mets_in_model:
                print(f"{met} is not in the model")
                all_fine = False
        return all_fine

    @staticmethod
    def setup_support_flux_exp(model,
                               rxn_fluxes: Dict[str, float]):
        """
        Set up model objective and bounds based on supporting reaction fluxes.

        Used internally for testing task sinks. Modifies the model in place.

        Parameters
        ----------
        model : cobra.Model
            The metabolic model to modify.
        rxn_fluxes : Dict[str, float]
            A dictionary mapping reaction IDs to their flux values obtained
            from a previous task simulation (e.g., pFBA). Reaction bounds
            are adjusted slightly around these fluxes, and the objective
            is set to maximize flux through these reactions (weighted by sign).
        """
        obj_dic = {}

        for k, v in rxn_fluxes.items():
            # Set bounds slightly away from the calculated flux to ensure feasibility
            # while constraining the solution space near the pFBA result.
            model.reactions.get_by_id(k).bounds = (-1000, max(-1e-6, v)) if v < 0 else (min(1e-6, v), 1000)
            obj_dic[model.reactions.get_by_id(k)] = 1 if v > 0 else -1
        model.objective = obj_dic

    def assign(self,
               model,
               all_mets_in_model = None,
               add_output_rxns = True,
               add_input_rxns = True,
               loose_input=False,
               loose_output=False,
               met_scaling_coefs=None):
        """
        Assign this task to a model and do the test

        Parameters
        ----------
        model: cobra.Model or pg.Model
            Assigned metabolic model
        all_mets_in_model: optional, list of str
            All the metabolites in the model
        loose_input: bool, default: False
            To loose the input constraints to the maximum (0, 1000)
        loose_output: bool, default: False
            To loose the output constraints to the maximum (0, 1000)
        Returns
        -------

        """
        if met_scaling_coefs is None:
            met_scaling_coefs = {}

        if not all_mets_in_model:
            all_mets_in_model = [m.id for m in model.metabolites]
        dummy_rxn_list, obj_rxn_list = [], []
        all_met_exist = True
        for mets, coef in zip([self.in_mets, self.out_mets], [1, -1]):
            for met in mets:
                met_id = self._substitute_compartment(met[self.met_id_str], met[self.compartment_str])
                dummy_rxn = cobra.Reaction('input_{}'.format(met_id) if coef == 1 else 'output_{}'.format(met_id))
                if met_id in all_mets_in_model:
                    scaling_factor = met_scaling_coefs[met_id] if met_id in met_scaling_coefs else 1

                    dummy_rxn.add_metabolites({
                        model.metabolites.get_by_id(met_id): coef / scaling_factor
                    })
                    if coef == -1 and loose_output:
                        dummy_rxn.lower_bound, dummy_rxn.upper_bound = 0, 1000
                    elif coef == 1 and loose_input:
                        dummy_rxn.lower_bound, dummy_rxn.upper_bound = 0, 1000
                    else:
                        dummy_rxn.lower_bound, dummy_rxn.upper_bound = met[self.lower_bound_str], met[self.upper_bound_str]

                    if coef == 1 and add_input_rxns:
                        dummy_rxn_list.append(dummy_rxn)
                    if coef == -1:
                        if add_output_rxns:
                            dummy_rxn_list.append(dummy_rxn)
                        obj_rxn_list.append(dummy_rxn)
                else:
                    print(met_id, 'not exists in the model')
                    all_met_exist = False
        if all_met_exist:
            model.add_reactions(dummy_rxn_list)

        return all_met_exist, dummy_rxn_list, obj_rxn_list


def get_met_prod_task(met_id: str,
                      comp: str = "c") -> Task:
    """
    Create a simple metabolic task for producing a single metabolite.

    This task defines the production of a specified metabolite from any
    available precursors in the model, with no specific inputs defined
    and boundary reactions generally open (unless knockout flags are changed later).

    Parameters
    ----------
    met_id: str
        The metabolite's ID of the created Task
    comp: str
        The compartment of the metabolite
    Returns
    -------
    task: Task
    """
    return Task(subsystem="",
                should_fail=False,
                in_mets=[],
                out_mets=[{"met_id": met_id, "lb": 0, "ub": 1000, "compartment": comp}],
                knockout_input_flag=False,
                knockout_output_flag=False)


class TaskContainer:
    """
    A container to hold and manage multiple Task objects.

    Attributes
    ----------
    tasks : Dict[str, Task]
        A dictionary where keys are task IDs (str) and values are Task objects.
    ALLOW_BATCH_CHANGED_ATTR : List[str]
        Class attribute listing Task attributes that can be modified in batch
        using `set_all_mets_attr`.
    """
    ALLOW_BATCH_CHANGED_ATTR = ["compartment_parenthesis",
                                "lower_bound_str",
                                "upper_bound_str",
                                "compartment_str"]

    def __init__(self,
                 tasks: Optional[Dict[str, Task]] = None):
        """
        Initialize the TaskContainer.

        Parameters
        ----------
        tasks : Optional[Dict[str, Task]], optional
            A dictionary of tasks to initialize the container with.
            Keys are task IDs, values are Task objects. Defaults to None,
            creating an empty container.

        Raises
        ------
        TypeError
            If `tasks` is provided and is not a dictionary.
        """
        if tasks is not None and not isinstance(tasks, dict):
            raise TypeError("tasks must be a dict with strings as keys and Task as values")
        self.tasks: Dict[str, Task] = {} if tasks is None else tasks

    def __len__(self) -> int:
        """Return the number of tasks in the container."""
        return len(self.tasks)

    def __contains__(self, item: str) -> bool:
        """Check if a task ID exists in the container."""
        return item in self.tasks

    def __repr__(self) -> str:
        """Return a string representation showing the number of tasks."""
        return str(len(self.tasks)) + " tasks in this container"

    def __getitem__(self, item: str) -> Task:
        """Retrieve a task by its ID."""
        return self.tasks[item]

    def __setitem__(self, key: str, value: Task):
        """Add or update a task in the container."""
        if not isinstance(value, Task):
            raise TypeError("Value must be a Task object.")
        self.tasks[key] = value

    def __add__(self, other: 'TaskContainer') -> 'TaskContainer':
        """
        Combine two TaskContainers.

        Parameters
        ----------
        other : TaskContainer
            Another TaskContainer to add.

        Returns
        -------
        TaskContainer
            A new TaskContainer containing tasks from both containers.

        Raises
        ------
        AssertionError
            If there are overlapping task IDs between the two containers.
        """
        assert len(set(other.tasks.keys()) & set(self.tasks.keys())) == 0, \
            f"task id collision: {set(other.tasks.keys()) & set(self.tasks.keys())}"
        new_task = {}
        new_task.update(self.tasks) # Start with self's tasks
        new_task.update(other.tasks) # Add/overwrite with other's tasks
        return self.__class__(new_task)

    def items(self):
        """Return an iterator over the (task_id, Task) items."""
        return self.tasks.items()

    def subset(self, items: List[str]) -> 'TaskContainer':
        """
        Create a new TaskContainer containing only specified task IDs.

        Parameters
        ----------
        items : List[str]
            A list of task IDs to include in the subset.

        Returns
        -------
        TaskContainer
            A new container with the subset of tasks.
        """
        return self.__class__(tasks={name: task for name, task in self.tasks.items() if name in items})

    @classmethod
    def load(cls,
             file_path: Union[str, Path] = TASKS_FILE_PATH) -> 'TaskContainer':
        """
        Load tasks from a JSON file.

        Searches for the file first at the provided path, then in the default
        'tasks' directory relative to the package root if the path is not absolute.

        Parameters
        ----------
        file_path : Union[str, Path], optional
            Path to the JSON file containing task definitions.
            Defaults to the path defined by `TASKS_FILE_PATH`.

        Returns
        -------
        TaskContainer
            A new container populated with tasks from the file.
        """
        tasks_folder = Path(__file__).parent.parent.parent.parent / "tasks"
        file_path = Path(file_path) # Ensure it's a Path object
        if (not file_path.is_file()) and (tasks_folder / file_path).is_file():
            file_path = (tasks_folder / file_path)

        with open(file_path) as json_file:
            data = json.load(json_file)
            tasks = {ids: Task(**obj) for ids, obj in data.items()}
        return cls(tasks)

    def save(self, file_path: Union[str, Path]):
        """
        Save the tasks in the container to a JSON file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path where the JSON file will be saved.
        """
        data = {tid: tobj.to_dict() for tid, tobj in self.tasks.items()}
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4) # Add indent for readability

    def set_all_mets_attr(self, attr_name: str, new_value: Any):
        """
        Set a specific attribute for all Task objects within the container.

        Only attributes listed in `ALLOW_BATCH_CHANGED_ATTR` can be modified.

        Parameters
        ----------
        attr_name : str
            The name of the Task attribute to modify.
        new_value : Any
            The new value to assign to the attribute.

        Raises
        ------
        AssertionError
            If `attr_name` is not in `ALLOW_BATCH_CHANGED_ATTR`.
        """
        assert attr_name in self.ALLOW_BATCH_CHANGED_ATTR, \
            f"attr_name must be one of: {self.ALLOW_BATCH_CHANGED_ATTR}"
        for tid, tobj in self.tasks.items():
            setattr(tobj, attr_name, new_value)


class TaskHandler:
    """
    Handles the execution and analysis of metabolic tasks on a given model.

    Attributes
    ----------
    model : cobra.Model
        The metabolic model on which tasks will be tested.
    tasks : TaskContainer
        The container holding the metabolic tasks to be tested.
    """
    def __init__(self,
                 model: cobra.Model,
                 tasks_path_or_container: Union[TaskContainer, str, Path],
                 model_compartment_parenthesis: str = "[{}]"
                 ):
        """
        Initialize the TaskHandler.

        Parameters
        ----------
        model : cobra.Model
            The metabolic model to use for task testing.
        tasks_path_or_container : Union[TaskContainer, str, Path]
            Either a TaskContainer object or a path to a JSON file
            defining the tasks.
        model_compartment_parenthesis : str, optional
            The format string used for compartment identifiers in the model's
            metabolite IDs (e.g., "[{}]" for "met_c[c]"). Defaults to "[{}]".
        """
        self.model = model
        self.tasks = self._init_task_container(tasks_path_or_container,
                                               compartment_patenthesis=model_compartment_parenthesis)

    @staticmethod
    def _init_task_container(task_container: Union[TaskContainer, str, Path],
                             compartment_patenthesis: str) -> TaskContainer:
        """Load or prepare the TaskContainer, setting compartment format."""
        if isinstance(task_container, str) or isinstance(task_container, Path):
            task_container_obj = TaskContainer.load(task_container)
            task_container_obj.set_all_mets_attr("compartment_parenthesis", compartment_patenthesis)
            return task_container_obj
        task_container.set_all_mets_attr("compartment_parenthesis", compartment_patenthesis)
        return task_container

    def _get_sink_result(self, sol_df, dummy_rxns, fail_threshold):
        passed_rxns = sol_df[(abs(sol_df['fluxes']) >= fail_threshold)].index.to_list()
        dummy_rxn_id_list = [r.id for r in dummy_rxns]
        return {"rxn_supps": [rxn for rxn in passed_rxns if rxn not in dummy_rxn_id_list]}

    def _get_test_result(self, sol_df, dummy_rxns, fail_threshold) -> dict:
        passed_rxns = sol_df[(abs(sol_df['fluxes']) >= fail_threshold)].index.to_list()
        dummy_rxn_id_list = [r.id for r in dummy_rxns]
        sup_rxns = [rxn for rxn in passed_rxns if rxn not in dummy_rxn_id_list]
        return {"task_support_rxns": sup_rxns,
                "task_support_rxn_fluxes": dict(zip(sup_rxns,
                                                    sol_df.loc[sup_rxns, "fluxes"].values))}

    def test_one_task(self,
                      task,
                      model,
                      all_mets_in_model,
                      method,
                      method_kws,
                      solver,
                      fail_threshold: float,
                      n_additional_path: int = 0,
                      **kwargs) -> dict:
        """
        Test a single metabolic task on a given model context.

        Applies task constraints (inputs, outputs, knockouts), runs flux
        analysis (e.g., pFBA), and determines if the task passes based on
        feasibility and expected outcome (`should_fail`). Optionally identifies
        supporting reactions.

        Parameters
        ----------
        task : Task
            The metabolic task object to test.
        model : cobra.Model
            The model context (potentially modified with knockouts) to test the task on.
        all_mets_in_model : List[str]
            A list of all metabolite IDs present in the original model.
        method : str
            Flux analysis method to use (e.g., "pFBA", "FBA").
        method_kws : Optional[Dict]
            Keyword arguments for the chosen flux analysis method.
        solver : str
            The LP solver to use (e.g., "gurobi", "cplex").
        fail_threshold : float
            Flux threshold below which a reaction is considered inactive.
        n_additional_path : int, optional
            Number of additional paths (minimal flux solutions) to find for
            identifying alternative supporting reactions. Default is 0.
        **kwargs
            Additional keyword arguments passed to `task.assign`.

        Returns
        -------
        dict
            A dictionary containing results: 'Passed', 'Should fail',
            'Missing mets', 'Status', 'Obj_value', 'Obj_rxns', 'system',
            'subsystem', 'description', 'annotation', and optionally
            'task_support_rxns' and 'task_support_rxn_fluxes'.
        """
        all_met_exist, dummy_rxns, obj_rxns = task.assign(model, all_mets_in_model, **kwargs)
        assert n_additional_path >= 0, "n_additional_path cannot be negative."

        if not all_met_exist:
            # Task cannot be tested if required metabolites are missing
            return {'Passed': False,
                    'Should fail': task.should_fail,
                    'Missing mets': True,
                    'Status': 'infeasible', 'Obj_value': 0, "Obj_rxns": [r.id for r in obj_rxns]}

        all_supp_rxns = set()
        test_results = {}
        n_pos_supps = 0
        true_status = "not analyzed"
        obj_val = np.nan
        for i in range(n_additional_path + 1):
            if i == 0:
                if method == "pFBA":
                    model.objective = {rxn: 1 for rxn in obj_rxns}
            elif i > 0:
                model.objective = {model.reactions.get_by_id(rxn): 1
                                   for rxn in all_supp_rxns}
                method = "FBA"

            analyzer = flux_analyzers[method](model=model, solver=solver)
            #sol = model.optimize(raise_error=False, objective_sense="minimize")
            #true_status = sol.status
            try:
                if i == 0:
                    result = analyzer.analyze(**(method_kws if method_kws is not None else {}))
                    test_results["task_support_rxns"] = []
                    test_results["task_support_rxn_fluxes"] = []
                    true_status = result.solution.status
                    obj_val = result.solution.objective_value
                else:
                    result = analyzer.analyze(objective_sense="minimize",
                                              **(method_kws if method_kws is not None else {}))
                test_result = self._get_test_result(result.flux_df, dummy_rxns, fail_threshold)
                all_supp_rxns |= set(test_result["task_support_rxns"])
                if n_pos_supps == len(all_supp_rxns):
                    break
                n_pos_supps = len(all_supp_rxns)
                test_results["task_support_rxns"].append(test_result["task_support_rxns"])
                test_results["task_support_rxn_fluxes"].append(test_result["task_support_rxn_fluxes"])

            except Infeasible:
                true_status = "infeasible"
                print("Got an infeasible result")
                break

        return {'Passed': (((true_status == 'optimal') != task.should_fail) and all_met_exist),
                'Should fail': task.should_fail,
                'Missing mets': not all_met_exist,
                'Status': true_status,
                'Obj_value': obj_val,
                "Obj_rxns": [r.id for r in obj_rxns],
                "system": task.system,
                "subsystem": task.subsystem,
                "description": task.description,
                "annotation": task.annotation,
                **test_results}

    @staticmethod
    def _test_task_sinks_utils(ID, task, model, rxn_fluxes):
        with model:
            task.setup_support_flux_exp(model, rxn_fluxes=rxn_fluxes)
            try:
                sol = pfba(model)
            except Infeasible:
                sol = None
                print(f"Task {ID} cannot support tasks' metabolites")
            except OptimizationError:
                sol = None
                print(f"Weird result happened when testing Task {ID}")

        return sol

    def test_task_sink_one_path(self, ID, task, model, fail_threshold, rxn_fluxes) -> dict:
        supp_sol = self._test_task_sinks_utils(ID, task, model, rxn_fluxes)
        supp_status = supp_sol.status if supp_sol is not None else "infeasible"
        if supp_status == "optimal":
            status = "optimal"
            result = self._get_sink_result(supp_sol.to_frame(), [], fail_threshold)
        else:
            status = "support rxns infeasible"
            result = {}
        return {"Sink Status": status, "Passed": status == "optimal", **result}

    def test_task_sinks(self, ID, task, model, fail_threshold, rxn_fluxes) -> dict:
        output_dic = {"Sink Status": [], "Passed": False}
        for rxn_f in rxn_fluxes:
            one_sink_result = self.test_task_sink_one_path(ID, task, model, fail_threshold,
                                                           rxn_fluxes=rxn_f)
            for k, v in one_sink_result.items():
                if k == "Passed":
                    # Overall 'Passed' is True if any path passes
                    output_dic[k] = (output_dic[k] or v)
                elif k not in output_dic:
                    # Initialize list for other keys
                    output_dic[k] = [v]
                else:
                    output_dic[k].append(v)
        return output_dic

    def test_tasks(self,
                   method="pFBA",
                   method_kws=None,
                   solver="gurobi",
                   get_support_rxns=True,
                   task_ids="all",
                   verbosity=0,
                   fail_threshold=1e-6,
                   n_additional_path=0,
                   met_scaling_coefs: Optional[Dict] = None,
                   log: Optional[Dict] = None) -> TaskAnalysis:
        """
        Test a set of metabolic tasks on the model.

        Iterates through specified tasks, applies constraints, runs flux
        analysis, and aggregates results into a TaskAnalysis object.

        Parameters
        ----------
        method : str, optional
            Flux analysis method used for testing each task (e.g., "pFBA").
            Default is "pFBA".
        method_kws : Optional[Dict], optional
            Keyword arguments for the flux analysis method. Default is None.
        solver : str, optional
            LP solver to use. Default is "gurobi".
        get_support_rxns : bool, optional
            If True, attempts to identify reactions supporting passed tasks
            by analyzing flux distributions and testing sinks. Default is True.
        task_ids : Union[str, List[str]], optional
            Specific task IDs to test. If "all" (default), tests all tasks
            in the container.
        verbosity : int, optional
            Level of printed output (0: silent, 1: summary, 2: detailed). Default is 0.
        fail_threshold : float, optional
            Flux threshold for determining reaction activity. Default is 1e-6.
        n_additional_path : int, optional
            Number of additional paths to find for supporting reactions. Default is 0.
        met_scaling_coefs : Optional[Dict], optional
            Dictionary mapping metabolite IDs to scaling coefficients, used if
            the model stoichiometry has been rescaled. Default is None.
        log : Optional[Dict], optional
            Additional information to store in the TaskAnalysis log. Default is None.

        Returns
        -------
        TaskAnalysis
            An object containing the aggregated results (DataFrame, score) and logs.
        """
        # Define boundary reactions based on type
        boundary = [r.id for r in self.model.exchanges] + \
                   [r.id for r in self.model.demands] + \
                   [r.id for r in self.model.sinks]
        org_boundary = get_organic_exs(self.model, [], [])
        ko_types = {"all": boundary, "organic": org_boundary}

        task_info = {}
        all_mets_in_model = [m.id for m in self.model.metabolites]
        for ID, task in self.tasks.items():
            if task_ids != "all" and ID not in task_ids:
                continue
            with self.model as model:
                if task.knockout_output_flag:
                    for r in ko_types[task.ko_output_type]:
                        model.reactions.get_by_id(r).upper_bound = 0
                if task.knockout_input_flag:
                    for r in ko_types[task.ko_output_type]:
                        model.reactions.get_by_id(r).lower_bound = 0
                if verbosity >= 2:
                    print(f'Checking Task {ID}')
                elif verbosity >= 1:
                    print(f'Task {ID} - ', end='')
                # add dummy reactions to the model
                task_info[ID] = self.test_one_task(task=task,
                                                   model=model,
                                                   all_mets_in_model=all_mets_in_model,
                                                   method=method,
                                                   method_kws=method_kws,
                                                   solver=solver,
                                                   fail_threshold=fail_threshold,
                                                   n_additional_path=n_additional_path,
                                                   met_scaling_coefs=met_scaling_coefs)
            if task_info[ID]['Passed'] and not task.should_fail and get_support_rxns:
                sup_exp_result = self.test_task_sinks(ID, task, self.model,
                                                      fail_threshold,
                                                      rxn_fluxes=task_info[ID]["task_support_rxn_fluxes"])
                task_info[ID].update(sup_exp_result)
            if verbosity >= 2:
                print('status: ', task_info[ID]['Status'],
                      'should fail: ', task.should_fail,
                      'Passed: ', task_info[ID]['Passed'])
            elif verbosity >= 1:
                print('Passed' if task_info[ID]['Passed'] else 'Failed')
        result_df = pd.DataFrame(data=task_info).T
        score = len(self.tasks) if task_ids == "all" else len(task_ids)
        for ID, info in task_info.items():
            if not info['Passed']:
                if verbosity >= 1:
                    print(f'Task {ID} is not correct')
                score -= 1
        print(f'score of the model: {score} / {len(self.tasks) if task_ids == "all" else len(task_ids)}')
        log = {"method": method,
               "method_kws": method_kws,
               **(log if log is not None else {})}

        task_result = TaskAnalysis(log=log)
        task_result.add_result(dict(result_df=result_df,
                                    score=score))
        return task_result


def table_to_container(df,
                       met_id_map=None,
                       id_name="ID",
                       should_fail_name="SHOULD FAIL",
                       desc_name="DESCRIPTION",
                       in_met_name="IN",
                       in_met_lb="IN LB",
                       in_met_ub="IN UB",
                       out_met_name="OUT",
                       out_met_lb="OUT LB",
                       out_met_ub="OUT UB",
                       sys_name="SYSTEM",
                       sub_sys_name="SUBSYSTEM",
                       compartment_format: str = r"\[(.?)\]" # Raw string for regex
                       ) -> TaskContainer:
    """
    Construct a TaskContainer from a pandas DataFrame.

    Parses a DataFrame where rows define aspects of metabolic tasks (inputs,
    outputs, metadata) and converts it into a TaskContainer object. Assumes
    a specific table structure defined by the column name parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing task definitions.
    met_id_map : Optional[Dict], optional
        A dictionary to map metabolite IDs found in the table to different IDs
        if needed. Default is None.
    id_name : str, optional
        Column name for the unique Task ID. Default is "ID".
    should_fail_name : str, optional
        Column name indicating if the task should fail (boolean). Default is "SHOULD FAIL".
    desc_name : str, optional
        Column name for the task description. Default is "DESCRIPTION".
    in_met_name : str, optional
        Column name for input metabolite IDs (including compartment). Default is "IN".
    in_met_lb : str, optional
        Column name for input metabolite lower bound. Default is "IN LB".
    in_met_ub : str, optional
        Column name for input metabolite upper bound. Default is "IN UB".
    out_met_name : str, optional
        Column name for output metabolite IDs (including compartment). Default is "OUT".
    out_met_lb : str, optional
        Column name for output metabolite lower bound. Default is "OUT LB".
    out_met_ub : str, optional
        Column name for output metabolite upper bound. Default is "OUT UB".
    sys_name : str, optional
        Column name for the task system. Default is "SYSTEM".
    sub_sys_name : str, optional
        Column name for the task subsystem. Default is "SUBSYSTEM".
    compartment_format : str, optional
        Regular expression to extract the compartment ID from metabolite strings
        (e.g., in `in_met_name`, `out_met_name`). Default captures a single
        character within square brackets: r"\\[(.?)\\]".

    Returns
    -------
    TaskContainer
        A container populated with Task objects derived from the DataFrame.
    """
    df = df.copy()
    # Forward fill Task IDs to associate multiple rows (e.g., multiple inputs/outputs) with the same task
    df[id_name] = df[id_name].fillna(method='ffill')
    task_d = {}

    for i in df[id_name].unique():
        sub_df = df[df[id_name] == i]
        task_names = {"system": sys_name, "subsystem": sub_sys_name, "description": desc_name,
                      "should_fail": should_fail_name, "annotation": None}
        task_kws = {}
        in_mets, out_mets = [], []
        for ind, row in sub_df.iterrows():
            for t, v in task_names.items():
                if v is not None and v in row and pd.notna(row[v]):
                    task_kws[t] = row[v]
                elif t == "should_fail":
                    task_kws[t] = False
            if pd.notna(row[in_met_name]):
                in_met_id_matches = [r for r in re.finditer(compartment_format, row[in_met_name])]
                in_met_id_match = max(in_met_id_matches, key=lambda x: x.start())
                met_id = met_id_map[row[in_met_name][:in_met_id_match.start()]] \
                    if met_id_map is not None else row[in_met_name][:in_met_id_match.start()]
                in_mets.append({"met_id": met_id,
                                "lb": row[in_met_lb] if pd.notna(row[in_met_lb]) else 0,
                                "ub": row[in_met_ub] if pd.notna(row[in_met_ub]) else 1000,
                                "compartment": re.findall(compartment_format, row[in_met_name])[0]})
            if pd.notna(row[out_met_name]):
                out_met_id_matches = [r for r in re.finditer(compartment_format, row[out_met_name])]
                out_met_id_match = max(out_met_id_matches, key=lambda x: x.start())
                met_id = met_id_map[row[out_met_name][:out_met_id_match.start()]] \
                    if met_id_map is not None else row[out_met_name][:out_met_id_match.start()]
                out_mets.append({"met_id": met_id,
                                 "lb": row[out_met_lb] if pd.notna(row[out_met_lb]) else 0,
                                 "ub": row[out_met_ub] if pd.notna(row[out_met_ub]) else 1000,
                                 "compartment": re.findall(compartment_format, row[out_met_name])[0]})
        task_kws["in_mets"] = in_mets
        task_kws["out_mets"] = out_mets
        task_d[i] = Task(**task_kws)

    return TaskContainer(task_d)
