import json
import re
from typing import Dict, Any, Union, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import cobra
from cobra.flux_analysis.parsimonious import pfba
from cobra.exceptions import Infeasible

from pipeGEM.integration.mapping import Expression
from pipeGEM.analysis import flux_analyzers, TaskAnalysis
from ._var import TASKS_FILE_PATH


class Task:
    def __init__(self,
                 should_fail: bool,
                 in_mets: List[Dict[str, Any]],
                 out_mets: List[Dict[str, Any]],
                 knockout_input_flag: bool = True,
                 knockout_output_flag: bool = True,
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

    def to_dict(self):
        return {"system": self.system,
                "subsystem": self.subsystem,
                "in_mets": self.in_mets,
                "out_mets": self.out_mets,
                "knockout_input_flag": self.knockout_input_flag,
                "knockout_output_flag": self.knockout_output_flag,
                "should_fail": self.should_fail,
                "description": self.description,
                "annotation": self.annotation}

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

    def setup_support_flux_exp(self,
                               model,
                               rxn_fluxes):
        obj_dic = {}

        for k, v in rxn_fluxes.items():
            model.reactions.get_by_id(k).bounds = (-1000, max(-1e-6, v)) if v < 0 else (min(1e-6, v), 1000)
            obj_dic[model.reactions.get_by_id(k)] = 1 if v > 0 else -1
        model.objective = obj_dic

    def assign(self,
               model,
               all_mets_in_model = None,
               add_output_rxns = True,
               add_input_rxns = True,
               loose_input=False,
               loose_output=False):
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
        if not all_mets_in_model:
            all_mets_in_model = [m.id for m in model.metabolites]
        dummy_rxn_list, obj_rxn_list = [], []
        all_met_exist = True
        for mets, coef in zip([self.in_mets, self.out_mets], [1, -1]):
            for met in mets:
                met_id = self._substitute_compartment(met[self.met_id_str], met[self.compartment_str])
                dummy_rxn = cobra.Reaction('input_{}'.format(met_id) if coef == 1 else 'output_{}'.format(met_id))
                if met_id in all_mets_in_model:
                    dummy_rxn.add_metabolites({
                        model.metabolites.get_by_id(met_id): coef
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

    Parameters
    ----------
    met_id: str
        The metabolite's ID of the created Task
    comp: str
        The compartment of the metabolite
    Returns
    -------

    """
    return Task(subsystem="",
                should_fail=False,
                in_mets=[],
                out_mets=[{"met_id": met_id, "lb": 0, "ub": 1000, "compartment": comp}],
                knockout_input_flag=False,
                knockout_output_flag=False)


class TaskContainer:
    ALLOW_BATCH_CHANGED_ATTR = ["compartment_parenthesis", "lower_bound_str", "upper_bound_str", "compartment_str"]

    def __init__(self, tasks = None):
        self.tasks = {} if tasks is None else tasks

    def __len__(self):
        return len(self.tasks)

    def __contains__(self, item):
        return item in self.tasks

    def __repr__(self):
        return str(len(self.tasks)) + " tasks in this container"

    def __getitem__(self, item):
        return self.tasks[item]

    def __setitem__(self, key, value):
        self.tasks[key] = value

    def __add__(self, other):
        assert len(set(other.tasks.keys()) & set(self.tasks.keys())) == 0, \
            f"task id collision: {set(other.tasks.keys()) & set(self.tasks.keys())}"
        new_task = {}
        new_task.update(other.tasks)
        new_task.update(self.tasks)
        return self.__class__(new_task)

    def items(self):
        return self.tasks.items()

    def subset(self, items):
        return self.__class__(tasks={name: task for name, task in self.tasks.items() if name in items})

    @classmethod
    def load(cls, file_path=TASKS_FILE_PATH):
        with open(file_path) as json_file:
            data = json.load(json_file)
            tasks = {ids: Task(**obj) for ids, obj in data.items()}
        return cls(tasks)

    def save(self, file_path):
        data = {tid: tobj.to_dict() for tid, tobj in self.tasks.items()}
        with open(file_path, "w") as f:
            json.dump(data, f)

    def set_all_mets_attr(self, attr_name, new):
        assert attr_name in self.ALLOW_BATCH_CHANGED_ATTR, "attr_name should be chose from: " + str(self.ALLOW_BATCH_CHANGED_ATTR)
        for tid, tobj in self.tasks.items():
            setattr(tobj, attr_name, new)


class TaskHandler:
    def __init__(self,
                 model,
                 tasks_path_or_container: Union[TaskContainer, str],
                 model_compartment_parenthesis: str = "[{}]"
                 ):
        self.model = model
        self.tasks = self._init_task_container(tasks_path_or_container,
                                               compartment_patenthesis=model_compartment_parenthesis)

    @staticmethod
    def _init_task_container(task_container, compartment_patenthesis):
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

    def test_one_task(self, task, model, all_mets_in_model, method, method_kws, solver, fail_threshold, **kwargs):
        all_met_exist, dummy_rxns, obj_rxns = task.assign(model, all_mets_in_model, **kwargs)
        if not all_met_exist:
            return {'Passed': False,
                    'Should fail': task.should_fail,
                    'Missing mets': True,
                    'Status': 'infeasible', 'Obj_value': 0, "Obj_rxns": obj_rxns}
        if method == "pFBA":
            model.objective = {rxn: 1 for rxn in obj_rxns}

        analyzer = flux_analyzers[method](model=model, solver=solver)
        sol = model.optimize(raise_error=False, objective_sense="minimize")
        true_status = sol.status
        if true_status == 'optimal':
            try:
                result = analyzer.analyze(**(method_kws if method_kws is not None else {}))
                result = self._get_test_result(result.result, dummy_rxns, fail_threshold)
            except Infeasible:
                true_status = "infeasible"
                result = {}
                print("get an unexpected result")
        else:
            result = {}

        return {'Passed': (((true_status == 'optimal') != task.should_fail) and all_met_exist),
                'Should fail': task.should_fail,
                'Missing mets': not all_met_exist,
                'Status': true_status,
                'Obj_value': sol.objective_value,
                "Obj_rxns": obj_rxns, **result}

    @staticmethod
    def _test_task_sinks_utils(ID, task, model, rxn_fluxes):
        with model:
            task.setup_support_flux_exp(model, rxn_fluxes=rxn_fluxes)
            try:
                sol = pfba(model)
            except Infeasible:
                sol = None
                print(f"Task {ID} cannot support tasks' metabolites")
        return sol

    def test_task_sinks(self, ID, task, model, fail_threshold, rxn_fluxes):
        supp_sol = self._test_task_sinks_utils(ID, task, model, rxn_fluxes)
        supp_status = supp_sol.status if supp_sol is not None else "infeasible"
        if supp_status == "optimal":
            status = "optimal"
            result = self._get_sink_result(supp_sol.to_frame(), [], fail_threshold)
        else:
            status = "support rxns infeasible"
            result = {}
        return {"Sink Status": status, "Passed": status == "optimal", **result}

    def test_tasks(self,
                   method="pFBA",
                   method_kws=None,
                   solver="gurobi",
                   get_support_rxns=True,
                   task_ids="all",
                   verbosity=0,
                   fail_threshold=1e-6,
                   log=None):
        # maybe needs some modifications
        boundary = [r.id for r in self.model.exchanges] + \
                   [r.id for r in self.model.demands] + \
                   [r.id for r in self.model.sinks]
        task_info = {}
        all_mets_in_model = [m.id for m in self.model.metabolites]
        for ID, task in self.tasks.items():
            if task_ids != "all" and ID not in task_ids:
                continue
            with self.model as model:
                for r in boundary:
                    if task.knockout_output_flag:
                        model.reactions.get_by_id(r).upper_bound = 0
                    if task.knockout_input_flag:
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
                                                   fail_threshold=fail_threshold)
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
        task_result.add_result(result_df=result_df, score=score)
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
                       compartment_format="\[(.?)\]"
                       ):
    # a helper function to construct TaskContainer from a dataframe, the table could be
    df = df.copy()
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
