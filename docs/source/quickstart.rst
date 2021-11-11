Installation
-------------

To use pipeGEM, first install it using pip:

.. code-block:: console

    $ pip install pipeGEM


QuickStart
-------------

To do model comparison using pipeGEM, create a Group object and call do_analysis method.
Then use plotting functions to visualize the results

.. code-block:: python

    import pipeGEM as pg
    from pipeGEM.utils import load_model

    model_1 = load_model("your_model_path_1")  # cobra.Model
    model_2 = load_model("your_model_path_2")
    group = pg.Group({"model1": model_1, "model2": model_2})

    # get basic info
    group.get_info()

    # Model comparison
    group.plot_components()

    # Do and plot pFBA result
    group.do_analysis(method="pFBA", constr="default")
    group.plot_flux(method="pFBA", constr="default", rxn_ids=["your rxn ids"])

To run existing or create a new pipeline, use the classes defined in pipeGEM.pipeline:

.. code-block:: python

    import pandas as pd
    from pipeGEM.pipeline.algo.constraints import GIMME
    from pipeGEM.utils import load_model

    # run GIMME algo
    template_model = load_model("your_template_model_path")
    rnaseq_data = pd.read_csv("your_dataset_path")

    pipeline = GIMME(rnaseq_data)
    pipeline(template_model)  # run the pipeline

    group = pipeline.output["models"]  # this is a Group contains context-specific Models
    group.plot_model_heatmap()  # visualize model distances

.. code-block:: python

    from pipeGEM.pipeline import *
    from pipeGEM.pipeline.preprocessing import *
    from pipeGEM.pipeline.task import *
    from pipeGEM.pipeline.threshold import *

    # creating new pipeLine
    class MyPipeLine(Pipeline):
        def __init__(self, data_df):
            super().__init__()
            self.data_df = data_df
            self.gene_dataset = GeneDataSet(data_df)
            self.threshold = BimodalThreshold()
            ...

        def run(self):
            threshold_dict = self.threshold.run(self.data_df)
            expression_dict = self.gene_dataset()
            ...
            self.output = ...
            return self.output

    template_model = load_model("your_template_model_path")
    rnaseq_data = pd.read_csv("your_dataset_path")
    my_pipeline = MyPipeLine(rnaseq_data)
    my_pipeline(template_model)
