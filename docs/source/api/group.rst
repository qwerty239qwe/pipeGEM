======
Group
======

.. currentmodule:: pipeGEM

Constructor
-----------

.. autosummary::
   .. toctree:: pipeGEM/

   Group

Attributes
----------

.. autosummary::
   .. toctree:: pipeGEM/

   Group.annotation
   Group.reaction_ids
   Group.gene_ids
   Group.metabolite_ids
   Group.subsystems
   Group.gene_data

Indexing / Getting information
-------------------------------

.. autosummary::
   .. toctree:: pipeGEM/

   Group.__getattr__
   Group.index
   Group.items
   Group.get_rxn_info
   Group.get_info

Utilities
---------

.. autosummary::
   .. toctree:: pipeGEM/

   Group.aggregate_models
   Group.rename
   Group.compare
   Group.add_annotation


Flux Analysis
---------------

.. autosummary::
   .. toctree:: pipeGEM/

   Group.do_flux_analysis
   Group.do_ko_analysis


Model comparison
---------------

.. autosummary::
   .. toctree:: pipeGEM/

   Group.compare


Model comparison results
---------------

.. autosummary::
   .. toctree:: pipeGEM/

   analysis.ComponentNumberAnalysis
   analysis.ComponentNumberAnalysis.plot
   analysis.ComponentComparisonAnalysis
   analysis.ComponentComparisonAnalysis.plot
   analysis.PCA_Analysis
   analysis.PCA_Analysis.plot