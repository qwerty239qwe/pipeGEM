======
Model
======

.. currentmodule:: pipeGEM

Constructor
-----------

.. autosummary::
   .. toctree:: pipeGEM/

   Model

Attributes
----------

.. autosummary::
   .. toctree:: pipeGEM/

   Model.annotation
   Model.reaction_ids
   Model.gene_ids
   Model.metabolite_ids
   Model.cobra_model
   Model.subsystems
   Model.gene_data
   Model.aggregated_gene_data
   Model.medium_data
   Model.tasks

Indexing / Getting information
-------------------------------

.. autosummary::
   .. toctree:: pipeGEM/

   Model.__getattr__
   Model.get_rxn_info

Utilities
---------

.. autosummary::
   .. toctree:: pipeGEM/

   Model.copy
   Model.rename
   Model.save_model

Adding data / tasks
---------

.. autosummary::
   .. toctree:: pipeGEM/

   Model.add_annotation
   Model.add_gene_data
   Model.set_gene_data
   Model.add_medium_data
   Model.add_tasks

Applying Data / Test Model
----------------------------

.. autosummary::
   .. toctree:: pipeGEM/

   Model.apply_medium
   Model.integrate_gene_data
   Model.test_tasks
   Model.get_activated_task_sup_rxns
   Model.check_rxn_scales
   Model.check_consistency

Flux Analysis
---------------

.. autosummary::
   .. toctree:: pipeGEM/

   Model.do_flux_analysis
   Model.do_ko_analysis