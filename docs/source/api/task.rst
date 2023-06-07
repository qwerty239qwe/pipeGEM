==============================
Task (functionality test)
==============================

.. currentmodule:: pipeGEM

Task object
------------

.. autosummary::
   .. toctree:: pipeGEM/

   analysis.tasks.Task
   analysis.tasks.Task.to_dict
   analysis.tasks.Task.assign
   analysis.tasks.Task.setup_support_flux_exp
   analysis.tasks.get_met_prod_task

Task Container
---------------

.. autosummary::
   .. toctree:: pipeGEM/

   analysis.tasks.TaskContainer
   analysis.tasks.TaskContainer.__getitem__
   analysis.tasks.TaskContainer.__setitem__
   analysis.tasks.TaskContainer.__add__
   analysis.tasks.TaskContainer.items
   analysis.tasks.TaskContainer.subset
   analysis.tasks.TaskContainer.load
   analysis.tasks.TaskContainer.save
   analysis.tasks.TaskContainer.set_all_mets_attr


Task Handler
---------------

.. autosummary::
   .. toctree:: pipeGEM/

   analysis.tasks.TaskHandler
   analysis.tasks.TaskHandler.test_one_task
   analysis.tasks.TaskHandler.test_task_sinks
   analysis.tasks.TaskHandler.test_tasks