======
Data
======

.. currentmodule:: pipeGEM

GeneData
~~~~~~~~
Constructor
-----------

.. autosummary::
   .. toctree:: pipeGEM/

   data.GeneData

Attributes
----------

.. autosummary::
   .. toctree:: pipeGEM/

   data.GeneData.rxn_scores
   data.GeneData.transformed_gene_data

Methods
--------

.. autosummary::
   .. toctree:: pipeGEM/
   data.GeneData.align
   data.GeneData.calc_rxn_score_stat
   data.GeneData.apply
   data.GeneData.aggregate

MediumData
~~~~~~~~~~~

Constructor
-----------

.. autosummary::
   .. toctree:: pipeGEM/

   data.MediumData
   data.MediumData.from_file

Methods
--------

.. autosummary::
   .. toctree:: pipeGEM/
   data.MediumData.align
   data.MediumData.apply

Fetching
~~~~~~~~
.. autosummary::
   .. toctree:: pipeGEM/

   load_remote_model
   load_model
   data.list_models
   data.fetch_brenda_data
   data.load_HPA_data
   data.fetch_HPA_data
   data.fetch_KEGG_gene_data
   data.fetch_KEGG_gene_list

Synthesis
~~~~~~~~~
.. autosummary::
   .. toctree:: pipeGEM/
   data.get_syn_gene_data


