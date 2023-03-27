import pandas as pd
import numpy as np
import os
from pipeGEM.plotting.scatter import plot_PCA_loading


def test_plot_PCA_loading():
    # Create a sample dataframe
    component_df = pd.DataFrame({'PC1': np.random.rand(50),
                                 'PC2': np.random.rand(50),
                                 'feature': ['f{}'.format(i) for i in range(50)]})

    # Test basic function call
    plot = plot_PCA_loading(component_df)
    assert isinstance(plot, dict)

    # Test number of features plotted
    plot = plot_PCA_loading(component_df, n_feature=10)
    assert len(plot['g'].get_axes()[0].lines) == 10

    # Test figure title
    plot = plot_PCA_loading(component_df, fig_title='Test Figure Title')
    assert plot['g'].get_axes()[0].get_title() == 'Test Figure Title'

    # Test x and y labels
    plot = plot_PCA_loading(component_df, x_label='X Label', y_label='Y Label')
    assert plot['g'].get_axes()[0].get_xlabel() == 'X Label'
    assert plot['g'].get_axes()[0].get_ylabel() == 'Y Label'

    # Test file save
    plot = plot_PCA_loading(component_df, file_name='test.png', dpi=100)
    assert plot['g'].get_axes()[0].get_title() == None # title should not be set
    assert os.path.exists('test.png')
    os.remove('test.png')