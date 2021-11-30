import json
from typing import Dict, Any, Union, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import cobra
from cobra.exceptions import Infeasible

from pipeGEM.analysis import FluxAnalyzer
from pipeGEM.integration.utils import find_exp_threshold
from pipeGEM.integration.mapping import Expression
from pipeGEM.analysis.tasks import *
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

    def assign(self,
               model,
               all_mets_in_model = None,
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
                    dummy_rxn_list.append(dummy_rxn)
                    if coef == -1:
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
        return TaskContainer(new_task)

    def items(self):
        return self.tasks.items()

    def subset(self, items):
        return TaskContainer(tasks={name: task for name, task in self.tasks.items() if name in items})

    @classmethod
    def load(cls, file_path=TASKS_FILE_PATH):
        with open(file_path) as json_file:
            data = json.load(json_file)
            tasks = {ids: Task(**obj) for ids, obj in data.items()}
        return cls(tasks)

    def save(self, file_path):
        data = {tid: tobj.to_dict() for tid, tobj in self.tasks.items()}
        print(json.dumps(data, indent=4))
        with open(file_path, "w") as f:
            json.dump(data, f)

    def set_all_mets_attr(self, attr_name, new):
        assert attr_name in self.ALLOW_BATCH_CHANGED_ATTR, "attr_name should be chose from: " + str(self.ALLOW_BATCH_CHANGED_ATTR)
        for tid, tobj in self.tasks.items():
            setattr(tobj, attr_name, new)


class TaskHandler:
    def __init__(self,
                 model: cobra.Model,
                 tasks_path_or_container: Union[TaskContainer, str],
                 model_compartment_parenthesis: str,
                 method: str = 'pFBA',
                 constr: str = 'GIMME',
                 method_kwargs: dict = None,
                 constr_kwargs: dict = None,
                 solver="glpk"
                 ):
        self.model = model
        self.tasks = self._init_task_container(tasks_path_or_container,
                                               compartment_patenthesis=model_compartment_parenthesis)
        self.analyzer = FluxAnalyzer(model=self.model, solver=solver)

        self._method = method
        self._method_kwargs = {} if method_kwargs is None else method_kwargs

        self._constr = constr
        self._constr_kwargs = {} if constr_kwargs is None else constr_kwargs

        self._result_df = pd.DataFrame(columns=['Passed', 'Should fail',
                                                'Missing mets', 'Status', 'Obj_value'])
        self._expr_th, self._non_expr_th, self._rxn_score = 0, 0, {}

    @property
    def result_df(self):
        return self._result_df

    @staticmethod
    def _init_task_container(task_container, compartment_patenthesis):
        if isinstance(task_container, str) or isinstance(task_container, Path):
            task_container_obj = TaskContainer.load(task_container)
            task_container_obj.set_all_mets_attr("compartment_parenthesis", compartment_patenthesis)
            return task_container_obj
        return task_container

    def change_constr(self, new_constr_name, **kwargs):
        self._constr = new_constr_name
        self._constr_kwargs = kwargs

    def update_gene_data(self,
                         gene_data_dict: Union[dict, pd.Series],
                         transform=np.log2,
                         threshold=1e-4):
        if isinstance(gene_data_dict, pd.Series):
            gene_data_dict = {gid: gexp
                              for gid, gexp in
                              zip(gene_data_dict.index, gene_data_dict.values)}

        self._expr_th, self._non_expr_th, self._rxn_score = find_exp_threshold(self.model,
                                                                               gene_data_dict,
                                                                               transform, threshold)
        self.analyzer.rxn_expr_score = self._rxn_score
        if self._constr == "GIMME":
            self._constr_kwargs['rxn_expression_dict'] = self._rxn_score
            self._constr_kwargs['low_exp'], self._constr_kwargs['high_exp'] = self._non_expr_th, self._expr_th

    def update_thresholds(self,
                          express_thres,
                          non_express_thres,
                          rxn_scores):
        self._expr_th = express_thres if express_thres is not None else self._expr_th
        self._non_expr_th = non_express_thres if non_express_thres is not None else self._non_expr_th
        self._rxn_score = rxn_scores if rxn_scores is not None else self._rxn_score
        if self._constr == "GIMME":
            self._constr_kwargs['rxn_expression_dict'] = self._rxn_score
            self._constr_kwargs['low_exp'], self._constr_kwargs['high_exp'] = self._non_expr_th, self._expr_th

    def _get_test_result(self, ID, sol_df, dummy_rxns):
        raise NotImplementedError

    def test_one_task(self, ID, task, model, all_mets_in_model):
        all_met_exist, dummy_rxns, obj_rxns = task.assign(model, all_mets_in_model)
        if not all_met_exist:
            return {'Passed': False, 'Should fail': task.should_fail,
                    'Missing mets': True, 'Status': 'infeasible', 'Obj_value': 0, "Obj_rxns": obj_rxns}
        if self._method == "pFBA":
            model.objective = {rxn: 1 for rxn in obj_rxns}
        sol = model.optimize()
        true_status = sol.status
        if true_status != 'infeasible':
            try:
                kws = {}
                kws.update(self._constr_kwargs)
                kws.update(self._method_kwargs)
                self.analyzer.do_analysis(method=self._method,
                                          constr=self._constr,
                                          **kws)
                sol_df = self.analyzer.get_flux(method=self._method,
                                                constr=self._constr,
                                                keep_rc=False)
                self._get_test_result(ID, sol_df, dummy_rxns)
            except Infeasible:
                true_status = "infeasible"
                print("get an unexpected result")

        return {'Passed': (((true_status != 'infeasible') != task.should_fail) and all_met_exist),
                'Should fail': task.should_fail,
                'Missing mets': not all_met_exist, 'Status': true_status,
                'Obj_value': sol.objective_value,
                "Obj_rxns": obj_rxns}

    def test(self, verbosity=0):
        # maybe needs some modifications
        boundary = [r for r in self.model.exchanges] + \
                   [r for r in self.model.demands] + \
                   [r for r in self.model.sinks]
        task_info = {}
        all_mets_in_model = [m.id for m in self.model.metabolites]
        for ID, task in self.tasks.items():
            with self.analyzer.model as model:
                for r in boundary:
                    if task.knockout_output_flag:
                        r.upper_bound = 0
                    if task.knockout_input_flag:
                        r.lower_bound = 0
                if verbosity >= 2:
                    print(f'Checking Task {ID}')
                elif verbosity >= 1:
                    print(f'Task {ID} - ', end='')
                # add dummy reactions to the model
                task_info[ID] = self.test_one_task(ID, task, model, all_mets_in_model)

                if verbosity >= 2:
                    print('status: ', task_info[ID]['Status'],
                          'should fail: ', task.should_fail,
                          'Passed: ', task_info[ID]['Passed'])
                elif verbosity >= 1:
                    print('Passed' if task_info[ID]['Passed'] else 'Failed')
        self._result_df = pd.DataFrame(data=task_info).T
        score = len(self.tasks)

        for ID, info in task_info.items():
            if not info['Passed']:
                if verbosity >= 1:
                    print(f'Task {ID} is not correct')
                score -= 1
        print(f'score of the model: {score} / {len(self.tasks)}')


class TaskTester(TaskHandler):
    def __init__(self,
                 model: cobra.Model,
                 task_container: Union[str, TaskContainer] = TASKS_FILE_PATH,
                 model_compartment_parenthesis="[{}]",
                 method='pFBA',
                 constr='GIMME',
                 fail_tol=1e-6,
                 method_kwargs: dict = None,
                 constr_kwargs: dict = None
                 ):

        super().__init__(model=model,
                         tasks_path_or_container=task_container,
                         model_compartment_parenthesis=model_compartment_parenthesis,
                         method=method,
                         constr=constr,
                         method_kwargs=method_kwargs,
                         constr_kwargs=constr_kwargs
                         )
        self.tol = fail_tol
        if method == 'pFBA':
            self._method_kwargs['fraction_of_optimum'] = 1.0

        self._passed_rxns: Union[Dict[str, List[str]], dict] = {}
        self._tasks_scores = {}
        self._passed_tasks_list = []

    @property
    def passed_rxns(self):
        return self._passed_rxns

    @property
    def tasks_scores(self):
        return self._tasks_scores

    @property
    def passed_tasks_list(self):
        return self._passed_tasks_list

    def get_all_passed_rxns(self) -> list:
        return list(set([r for ind in self._passed_tasks_list for r in self._passed_rxns[ind]]))

    def _get_test_result(self, ID, sol_df, dummy_rxns) -> None:
        passed_rxns = sol_df[(abs(sol_df['fluxes']) >= self.tol)].index.to_list()
        dummy_rxn_id_list = [r.id for r in dummy_rxns]
        self._passed_rxns[ID] = [rxn for rxn in passed_rxns if rxn not in dummy_rxn_id_list]

    def map_expr_to_tasks(self,
                          expression_threshold = None,
                          non_expression_threshold = None,
                          rxn_scores = None,
                          coef=0.8,
                          **kwargs) -> None:
        if expression_threshold is not None:
            self._expr_th = expression_threshold
        if non_expression_threshold is not None:
            self._non_expr_th = non_expression_threshold
        if rxn_scores is not None:
            self._rxn_score = rxn_scores
        if self._passed_rxns is not None:
            self._tasks_scores = dict()
            for ID, rxns in self._passed_rxns.items():
                grr_rxns = [v for v in rxns if self._rxn_score[v] != 0 and not np.isnan(self._rxn_score[v])]
                if len(grr_rxns) > 0:
                    self._tasks_scores[ID] = sum([self._rxn_score[v]
                                                  if v in self._rxn_score and self._rxn_score[v] != -1 else 0
                                                  for v in grr_rxns]) / len(grr_rxns)
                else:
                    # always passed if reactions in a task are not related to genes
                    self._tasks_scores[ID] = self._expr_th + 1
        else:
            raise ValueError('Need to test the template model first! Use .test() to do so')

        self._passed_tasks_list = [ID for ID, exp_score in self._tasks_scores.items()
                                   if exp_score - self._non_expr_th >= coef * (self._expr_th - self._non_expr_th)]


class TaskValidator(TaskHandler):
    def __init__(self,
                 model,
                 task_container: Union[str, TaskContainer] = TASKS_FILE_PATH,
                 model_compartment_parenthesis="[{}]",
                 method: str = "pFBA",
                 constr: str = "None",
                 method_kwargs: dict = None,
                 constr_kwargs: dict = None,
                 ):
        super().__init__(model=model,
                         tasks_path_or_container=task_container,
                         model_compartment_parenthesis=model_compartment_parenthesis,
                         method=method,
                         constr=constr,
                         method_kwargs=method_kwargs,
                         constr_kwargs=constr_kwargs
                         )
        if method == 'pFBA':
            self._method_kwargs['fraction_of_optimum'] = 1.0
        self._flux_dfs = {}

    @property
    def flux_dfs(self):
        return self._flux_dfs

    def _get_test_result(self, ID, sol_df, dummy_rxns):
        dummy_rxn_id_list = [r.id for r in dummy_rxns]
        df = sol_df.loc[[r for r in sol_df.index.to_list()
                         if r not in dummy_rxn_id_list], :]
        self._flux_dfs[ID] = df


def get_task_protection_rxns(ref_model,
                             sample_names: List[str],
                             expr_threshold_dic: Dict[str, float],
                             non_expr_threshold_dic: Dict[str, float],
                             expression_dic: Dict[str, Expression],
                             constr="default",
                             task_file_path=TASKS_FILE_PATH,
                             model_compartment_format="[{}]",
                             added_protected_rxns: List[str] = None,
                             plot_file_name: Optional[str] = None,
                             score_z_score=0,
                             task_kwargs=None) -> Dict[str, Dict[str, List[str]]]:
    if task_kwargs is None:
        task_kwargs = {}
    protected_rxns = {sample: [] for sample in sample_names}
    task_scores = {}
    constr = "None" if constr is None else constr
    model_tester = TaskTester(ref_model,
                              constr=constr,
                              task_container=task_file_path,
                              model_compartment_parenthesis=model_compartment_format,
                              **task_kwargs)
    if constr not in ["GIMME"]:
        model_tester.test()
    for sample in sample_names:
        model_tester.update_thresholds(express_thres=expr_threshold_dic[sample],
                                       non_express_thres=non_expr_threshold_dic[sample],
                                       rxn_scores=expression_dic[sample].rxn_scores)
        if constr in ["GIMME"]:
            model_tester.test()
        model_tester.map_expr_to_tasks()
        protected_rxns[sample].extend(model_tester.get_all_passed_rxns())
        task_scores[sample] = model_tester.tasks_scores
        if added_protected_rxns is not None:
            protected_rxns[sample].extend(added_protected_rxns)
    if plot_file_name:
        score_data = pd.DataFrame(task_scores).fillna(-1)
        subsystem_dict = {ID: task.subsystem
                          for ID, task in model_tester.tasks.items()}
        # plot_clustermap(score_data,
        #                 rxn_subsystem=subsystem_dict,  # TODO rename plot_clustermap's arg
        #                 file_name = plot_file_name,
        #                 z_score=score_z_score,
        #                 )
    else:
        score_data = None
        subsystem_dict = None

    return {"protected_rxns": protected_rxns,
            "score_data": score_data,
            "subsystem_dict": subsystem_dict}