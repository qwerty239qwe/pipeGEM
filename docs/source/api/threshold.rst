==========
Threshold
==========

.. currentmodule:: pipeGEM

Collection
-----------
.. autosummary::
   .. toctree:: pipeGEM/

   analysis.threshold_finders

Calling thresholding
-----------------------
.. autosummary::
   .. toctree:: pipeGEM/

   data.GeneData.get_threshold
   data.GeneData.assign_local_threshold
   data.find_local_threshold

Thresholding operators
-----------------------
.. autosummary::
   .. toctree:: pipeGEM/

   analysis.rFASTCORMICSThreshold
   analysis.PercentileThreshold
   analysis.LocalThreshold

Threshold analysis result
-----------------------

.. autosummary::
   .. toctree:: pipeGEM/

   analysis.LocalThresholdAnalysis
   analysis.LocalThresholdAnalysis.plot
   analysis.PercentileThresholdAnalysis
   analysis.PercentileThresholdAnalysis.plot
   analysis.rFASTCORMICSThresholdAnalysis
   analysis.rFASTCORMICSThresholdAnalysis.plot