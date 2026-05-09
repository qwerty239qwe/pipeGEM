"""Comprehensive tests for the pipeGEM plotting module.

Tests cover: utility functions, individual plot functions, plotter classes,
and result-class plot interfaces.  All tests use the ``Agg`` backend to
avoid GUI windows.
"""
import itertools
import warnings
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from pipeGEM.plotting._utils import (
    _save_fig, _get_subsystem_ticks, save_fig, handle_colors, extract_kws,
    format_file_name, _set_default_ax, draw_significance,
)
from pipeGEM.plotting._class import (
    BasePlotter, FBAPlotter, pFBAPlotter, FVAPlotter, SamplingPlotter,
    rFastCormicThresholdPlotter, PercentileThresholdPlotter,
    LocalThresholdPlotter, ComponentNumberPlotter, ComponentComparisonPlotter,
    DimReductionPlotter, HeatmapPlotter, CorrelationPlotter, DataCatPlotter,
)
from pipeGEM.plotting._flux import (
    plot_fba, plot_fva, plot_sampling_df,
    plot_sampling_displot, plot_sampling_catplot,
    plot_one_sampling, plot_sampling,
)
from pipeGEM.plotting.scatter import (
    plot_PCA, plot_2D_PCA_score, plot_3D_PCA_score,
    plot_PCA_screeplot, plot_PCA_loading, plot_embedding, plot_Eflux_scatter,
    plot_volcano,
)
from pipeGEM.plotting.curve import (
    plot_rFastCormic_thresholds, plot_percentile_thresholds,
)
from pipeGEM.plotting.categorical import (
    plot_data_cat, plot_model_components, plot_local_threshold_boxplot,
)
from pipeGEM.plotting.heatmap import (
    _get_qual_size, _resolve_palette, _parse_one_axis_colors, _parse_colors,
    plot_heatmap, plot_clustermap, _modify_clustermap_for_subsys,
)
from pipeGEM.plotting._prep import (
    prep_flux_df, prep_fva_plotting_data, filter_fva_df,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture()
def fba_flux_df():
    """Minimal FBA flux DataFrame."""
    return pd.DataFrame({
        "Reaction": ["R1", "R1", "R2", "R2", "R3", "R3"],
        "fluxes": [1.0, 2.0, 0.0, 0.0, 3.5, 4.5],
        "reduced_costs": [0.0] * 6,
        "model": ["m1", "m2", "m1", "m2", "m1", "m2"],
    })


@pytest.fixture()
def fva_flux_df():
    """Minimal FVA flux DataFrame."""
    return pd.DataFrame({
        "Reaction": ["R1", "R1", "R2", "R2"],
        "minimum": [-1.0, -0.5, 0.0, 0.0],
        "maximum": [1.0, 0.8, 0.0, 0.0],
        "model": ["m1", "m2", "m1", "m2"],
    })


@pytest.fixture()
def pca_data():
    """Minimal PCA result dict."""
    np.random.seed(42)
    n_samples = 6
    pc_df = pd.DataFrame(
        np.random.randn(3, n_samples),
        index=["PC1", "PC2", "PC3"],
        columns=[f"s{i}" for i in range(n_samples)],
    )
    exp_var_df = pd.DataFrame({"exp_var": [0.5, 0.3, 0.2]}, index=["PC1", "PC2", "PC3"])
    component_df = pd.DataFrame({
        "PC1": np.random.randn(20),
        "PC2": np.random.randn(20),
    }, index=[f"f{i}" for i in range(20)])
    return {"PC": pc_df, "exp_var": exp_var_df, "components": component_df}


@pytest.fixture()
def groups_dict():
    return {"g1": ["s0", "s1", "s2"], "g2": ["s3", "s4", "s5"]}


# ===================================================================
# _utils tests
# ===================================================================

class TestSaveFig:
    def test_save_simple_name(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "test.png"
        _save_fig(str(out), prefix="", dpi=50, g=fig)
        assert out.exists()

    def test_save_with_prefix(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        _save_fig(str(tmp_path / "plot.png"), prefix="FBA_", dpi=50, g=fig)
        assert (tmp_path / "FBA_plot.png").exists()

    def test_save_with_subdirectory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        _save_fig(str(sub / "result.png"), prefix="", dpi=50, g=fig)
        assert (sub / "result.png").exists()

    def test_save_without_g_uses_plt(self, tmp_path):
        plt.figure()
        plt.plot([0, 1], [0, 1])
        out = tmp_path / "current.png"
        _save_fig(str(out), prefix="", dpi=50)
        assert out.exists()

    def test_prefix_none_treated_as_empty(self, tmp_path):
        fig, ax = plt.subplots()
        out = tmp_path / "p.png"
        _save_fig(str(out), prefix=None, dpi=50, g=fig)
        assert out.exists()


class TestHandleColors:
    def test_returns_enough_colors(self):
        colors = handle_colors(palette="deep", n_colors_used=3)
        assert len(colors) >= 3

    def test_switches_palette_when_too_few(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            colors = handle_colors(palette="deep", n_colors_used=100)
            assert len(colors) >= 100
            assert len(w) == 1
            assert "changed" in str(w[0].message).lower()

    def test_no_warning_when_disabled(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            handle_colors(palette="deep", n_colors_used=100, warn_when_switch=False)
            assert len(w) == 0

    def test_non_qualitative_palette_no_crash(self):
        colors = handle_colors(palette="Spectral", n_colors_used=5)
        assert len(colors) >= 5

    def test_non_qualitative_palette_viridis(self):
        colors = handle_colors(palette="viridis", n_colors_used=20)
        assert len(colors) >= 20


class TestExtractKws:
    def test_pops_existing_keys(self):
        kws = {"a": 1, "b": 2, "c": 3}
        result = extract_kws(kws, ["a", "b"], default_kws={"a": 10, "b": 20})
        assert result == {"a": 1, "b": 2}
        assert "a" not in kws
        assert "b" not in kws
        assert kws == {"c": 3}

    def test_uses_defaults_when_missing(self):
        kws = {"c": 3}
        result = extract_kws(kws, ["a"], default_kws={"a": 99})
        assert result == {"a": 99}


class TestFormatFileName:
    def test_simple_format(self):
        kws = {"file_name": "out.png"}
        result = format_file_name("{file_name}", kws)
        assert result == "out.png"

    def test_multi_field_format(self):
        kws = {"method": "FBA", "ext": "png"}
        result = format_file_name("{method}_result.{ext}", kws)
        assert result == "FBA_result.png"

    def test_missing_key_returns_none(self):
        kws = {}
        result = format_file_name("{file_name}", kws)
        assert result is None


class TestSetDefaultAx:
    def test_basic(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax = _set_default_ax(ax, x_label="X", y_label="Y", title="T")
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert ax.get_title() == "T"


class TestDrawSignificance:
    def test_horizontal_line(self):
        fig, ax = plt.subplots()
        ax.bar([0, 1], [5, 10])
        draw_significance(ax, x_pos_list=[0, 1], y_pos_list=[12, 12], num_stars=2)
        assert len(ax.lines) > 0

    def test_vertical_line(self):
        fig, ax = plt.subplots()
        ax.barh([0, 1], [5, 10])
        draw_significance(ax, x_pos_list=[12, 12], y_pos_list=[0, 1], num_stars=3)
        assert len(ax.lines) > 0

    def test_invalid_line_raises(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            draw_significance(ax, [0, 1], [0, 1], 1)  # diagonal


# ===================================================================
# _prep tests
# ===================================================================

class TestPrep:
    def test_prep_flux_df(self):
        df = pd.DataFrame({
            "model": ["m1", "m2"],
            "R1": [1.0, 2.0],
            "R2": [3.0, 4.0],
        })
        model_df, flux_df = prep_flux_df(df, ["R1", "R2"])
        assert "model" in model_df.columns
        assert list(flux_df.columns) == ["R1", "R2"]

    def test_prep_flux_df_with_dict(self):
        df = pd.DataFrame({"R1": [1.0], "R2": [2.0]})
        _, flux_df = prep_flux_df(df, {"R1": "Reaction_A"})
        assert "Reaction_A" in flux_df.columns

    def test_prep_fva_plotting_data(self):
        df = pd.DataFrame({
            "Reaction": ["R1", "R2"],
            "minimum": [-1.0, 0.0],
            "maximum": [1.0, 0.5],
        })
        melted = prep_fva_plotting_data(df)
        assert "Flux" in melted.columns
        assert "Stats" in melted.columns

    def test_filter_fva_df(self):
        df = pd.DataFrame({
            "Reaction": ["R1", "R2", "R3"],
            "minimum": [-1.0, -0.0001, 0.0],
            "maximum": [1.0, 0.0001, 0.0],
        })
        filtered = filter_fva_df(df, threshold=0.001, verbosity=0)
        assert len(filtered) == 1
        assert filtered.iloc[0]["Reaction"] == "R1"


# ===================================================================
# Flux plotting tests
# ===================================================================

class TestPlotFBA:
    def test_basic_plot(self, fba_flux_df):
        result = plot_fba(fba_flux_df, rxn_ids=["R1", "R3"])
        assert "g" in result
        assert hasattr(result["g"], "figure") or isinstance(result["g"], plt.Figure)

    def test_filters_zeros(self, fba_flux_df):
        result = plot_fba(fba_flux_df, rxn_ids=["R1", "R2", "R3"], filter_all_zeros=True)
        assert "g" in result

    def test_with_group_by(self, fba_flux_df):
        result = plot_fba(fba_flux_df, rxn_ids=["R1"], group_by="model")
        assert "g" in result

    def test_horizontal(self, fba_flux_df):
        result = plot_fba(fba_flux_df, rxn_ids=["R1"], vertical=False)
        assert "g" in result

    def test_with_title(self, fba_flux_df):
        result = plot_fba(fba_flux_df, rxn_ids=["R1"], fig_title="Test FBA")
        assert "g" in result

    def test_index_as_reaction_id(self):
        df = pd.DataFrame({"fluxes": [1.0, 2.0]}, index=["R1", "R2"])
        result = plot_fba(df, rxn_ids=["R1"])
        assert "g" in result


class TestPlotFVA:
    def test_basic_plot(self, fva_flux_df):
        result = plot_fva(fva_flux_df, rxn_ids=["R1"])
        assert "g" in result
        assert isinstance(result["g"], plt.Figure)

    def test_filter_zeros(self, fva_flux_df):
        result = plot_fva(fva_flux_df, rxn_ids=["R1", "R2"], filter_all_zeros=True)
        assert "g" in result


class TestPlotSamplingDf:
    def test_displot(self):
        df = pd.DataFrame({
            "R1": np.random.randn(30),
            "group": ["g1"] * 15 + ["g2"] * 15,
        })
        result = plot_sampling_df(df, rxn_id="R1", kind="kde",
                                   group_by="group", plotting_type="displot")
        assert "g" in result

    def test_catplot(self):
        df = pd.DataFrame({
            "R1": np.random.randn(30),
            "group": ["g1"] * 15 + ["g2"] * 15,
        })
        result = plot_sampling_df(df, rxn_id="R1", kind="box",
                                   group_by="group", plotting_type="catplot")
        assert "g" in result

    def test_invalid_type_raises(self):
        df = pd.DataFrame({"R1": [1], "g": ["a"]})
        with pytest.raises(AssertionError):
            plot_sampling_df(df, "R1", "kde", "g", plotting_type="invalid")


# ===================================================================
# Curve / threshold tests
# ===================================================================

class TestPlotRFastCormicThresholds:
    def test_basic(self):
        x = np.linspace(0, 10, 100)
        y = np.exp(-0.5 * (x - 5) ** 2)
        result = plot_rFastCormic_thresholds(x, y, exp_th=6.0, nonexp_th=4.0)
        assert "g" in result
        assert isinstance(result["g"], plt.Figure)

    def test_with_curves(self):
        x = np.linspace(0, 10, 100)
        y = np.exp(-0.5 * (x - 5) ** 2)
        right_c = np.exp(-0.5 * (x - 7) ** 2) * 0.5
        left_c = np.exp(-0.5 * (x - 3) ** 2) * 0.5
        result = plot_rFastCormic_thresholds(x, y, 6.0, 4.0,
                                              right_c=right_c, left_c=left_c)
        assert isinstance(result["g"], plt.Figure)


class TestPlotPercentileThresholds:
    def test_float_threshold(self):
        data = pd.Series(np.random.randn(200))
        result = plot_percentile_thresholds(data, exp_th=0.5)
        assert "g" in result
        assert isinstance(result["g"], plt.Figure)

    def test_series_threshold(self):
        data = pd.Series(np.random.randn(200))
        th = pd.Series({"p25": -0.5, "p75": 0.5})
        result = plot_percentile_thresholds(data, exp_th=th)
        assert isinstance(result["g"], plt.Figure)

    def test_list_threshold(self):
        data = pd.Series(np.random.randn(200))
        result = plot_percentile_thresholds(data, exp_th=[-0.5, 0.0, 0.5])
        assert isinstance(result["g"], plt.Figure)

    def test_old_typo_param_still_works(self):
        data = pd.Series(np.random.randn(200))
        result = plot_percentile_thresholds(data, exp_th=0.5, palatte="muted")
        assert isinstance(result["g"], plt.Figure)

    def test_new_palette_param(self):
        data = pd.Series(np.random.randn(200))
        result = plot_percentile_thresholds(data, exp_th=0.5, palette="muted")
        assert isinstance(result["g"], plt.Figure)


# ===================================================================
# Categorical tests
# ===================================================================

class TestPlotDataCat:
    def test_basic(self):
        df = pd.DataFrame({
            "id": ["A", "A", "B", "B"],
            "val": [1, 2, 3, 4],
            "grp": ["x", "y", "x", "y"],
        })
        result = plot_data_cat(df, id_col="id", val_col="val", ids=["A", "B"])
        assert "g" in result

    def test_with_hue(self):
        df = pd.DataFrame({
            "id": ["A", "A", "B", "B"],
            "val": [1, 2, 3, 4],
            "grp": ["x", "y", "x", "y"],
        })
        result = plot_data_cat(df, id_col="id", val_col="val",
                                ids=["A", "B"], group_col="grp")
        assert "g" in result


class TestPlotModelComponents:
    def test_basic(self):
        df = pd.DataFrame({
            "group": ["g1", "g1", "g2", "g2"] * 3,
            "component": ["n_rxns"] * 4 + ["n_mets"] * 4 + ["n_genes"] * 4,
            "number": np.random.randint(10, 100, 12),
        })
        result = plot_model_components(df, order=["g1", "g2"])
        assert "g" in result
        assert isinstance(result["g"], plt.Figure)


class TestPlotLocalThresholdBoxplot:
    def test_basic(self):
        genes = ["g1", "g2"]
        groups = ["A", "B"]
        group_dic = {"A": ["s1", "s2"], "B": ["s3", "s4"]}
        data = pd.DataFrame(
            np.random.randn(2, 4),
            index=genes,
            columns=["s1", "s2", "s3", "s4"],
        )
        local_th = pd.DataFrame(
            [[0.5, 0.3], [0.4, 0.2]], index=genes, columns=groups
        )
        global_on = pd.Series({"A": 0.8, "B": 0.7})
        global_off = pd.Series({"A": -0.5, "B": -0.4})
        result = plot_local_threshold_boxplot(
            data, genes, groups, group_dic, local_th, global_on, global_off
        )
        assert "g" in result
        assert isinstance(result["g"], plt.Figure)

    def test_no_duplicate_legends(self):
        genes = ["g1", "g2"]
        groups = ["A", "B"]
        group_dic = {"A": ["s1", "s2"], "B": ["s3", "s4"]}
        data = pd.DataFrame(
            np.random.randn(2, 4), index=genes, columns=["s1", "s2", "s3", "s4"]
        )
        local_th = pd.DataFrame([[0.5, 0.3], [0.4, 0.2]], index=genes, columns=groups)
        global_on = pd.Series({"A": 0.8, "B": 0.7})
        global_off = pd.Series({"A": -0.5, "B": -0.4})
        result = plot_local_threshold_boxplot(
            data, genes, groups, group_dic, local_th, global_on, global_off
        )
        fig = result["g"]
        ax = fig.get_axes()[0]
        legend = ax.get_legend()
        if legend is not None:
            labels = [t.get_text() for t in legend.get_texts()]
            assert labels.count("Global-on threshold") <= 1
            assert labels.count("Global-off threshold") <= 1
            assert labels.count("Local threshold") <= 1


# ===================================================================
# Heatmap tests
# ===================================================================

class TestParseOneAxisColors:
    def test_string_palette(self):
        groups = pd.DataFrame({
            "treatment": ["a", "b", "a"],
        }, index=["s1", "s2", "s3"])
        result = _parse_one_axis_colors(groups, color_palette="deep")
        assert result.shape == groups.shape
        assert all(isinstance(v, tuple) for v in result["treatment"])

    def test_list_palette(self):
        groups = pd.DataFrame({
            "treatment": ["a", "b", "a"],
            "cell": ["x", "y", "x"],
        }, index=["s1", "s2", "s3"])
        result = _parse_one_axis_colors(groups, color_palette=["deep", "muted"])
        assert result.shape == groups.shape

    def test_dict_palette(self):
        groups = pd.DataFrame({
            "treatment": ["a", "b"],
        }, index=["s1", "s2"])
        result = _parse_one_axis_colors(groups,
                                         color_palette={"treatment": "deep"})
        assert result.shape == groups.shape


class TestPlotHeatmap:
    def test_basic(self):
        data = pd.DataFrame(np.random.randn(5, 5),
                             index=[f"r{i}" for i in range(5)],
                             columns=[f"c{i}" for i in range(5)])
        result = plot_heatmap(data)
        assert "g" in result
        assert isinstance(result["g"], plt.Figure)

    def test_with_title(self):
        data = pd.DataFrame(np.random.randn(5, 5))
        result = plot_heatmap(data, fig_title="Test Heatmap")
        assert "g" in result

    def test_with_row_groups(self):
        data = pd.DataFrame(np.random.randn(4, 4),
                             index=["a", "b", "c", "d"],
                             columns=["x", "y", "z", "w"])
        row_groups = pd.DataFrame({"grp": ["g1", "g1", "g2", "g2"]},
                                   index=["a", "b", "c", "d"])
        result = plot_heatmap(data, row_groups=row_groups)
        assert "g" in result


# ===================================================================
# Scatter / PCA tests
# ===================================================================

class TestPlot2DPCAScore:
    def test_basic(self, pca_data, groups_dict):
        result = plot_2D_PCA_score(pca_data["PC"], groups=groups_dict)
        assert "g" in result
        assert isinstance(result["g"], plt.Figure)

    def test_with_exp_var(self, pca_data, groups_dict):
        result = plot_2D_PCA_score(pca_data["PC"], groups=groups_dict,
                                    exp_var_df=pca_data["exp_var"])
        assert "g" in result
        fig = result["g"]
        ax = fig.get_axes()[0]
        assert "Principal Component 1" in ax.get_xlabel()
        assert "variance" in ax.get_xlabel().lower()

    def test_labels_not_empty_when_no_exp_var(self, pca_data, groups_dict):
        result = plot_2D_PCA_score(pca_data["PC"], groups=groups_dict)
        ax = result["g"].get_axes()[0]
        assert ax.get_xlabel() == "Principal Component 1"
        assert ax.get_ylabel() == "Principal Component 2"


class TestPlot3DPCAScore:
    def test_basic(self, pca_data, groups_dict):
        result = plot_3D_PCA_score(pca_data["PC"], groups=groups_dict)
        assert "g" in result

    def test_with_exp_var(self, pca_data, groups_dict):
        result = plot_3D_PCA_score(pca_data["PC"], groups=groups_dict,
                                    exp_var_df=pca_data["exp_var"])
        assert "g" in result


class TestPlotPCAScreeplot:
    def test_basic(self, pca_data):
        result = plot_PCA_screeplot(pca_data["exp_var"])
        assert "g" in result
        ax = result["g"].get_axes()[0]
        assert "Principal Component" in ax.get_xlabel()
        assert "Explained" in ax.get_ylabel()


class TestPlotPCALoading:
    def test_basic(self, pca_data):
        result = plot_PCA_loading(pca_data["components"])
        assert "g" in result

    def test_n_feature(self, pca_data):
        result = plot_PCA_loading(pca_data["components"], n_feature=5)
        ax = result["g"].get_axes()[0]
        assert len(ax.lines) == 5


class TestPlotEmbedding:
    def test_basic(self):
        df = pd.DataFrame({
            "embedding 1": np.random.randn(6),
            "embedding 2": np.random.randn(6),
        }, index=[f"s{i}" for i in range(6)])
        groups = {"g1": ["s0", "s1", "s2"], "g2": ["s3", "s4", "s5"]}
        result = plot_embedding(df, groups=groups)
        assert "g" in result

    def test_groups_none_defaults(self):
        df = pd.DataFrame({
            "embedding 1": [1.0, 2.0],
            "embedding 2": [3.0, 4.0],
        }, index=["s0", "s1"])
        result = plot_embedding(df, groups=None)
        assert "g" in result


class TestPlotEfluxScatter:
    def test_returns_dict(self):
        r_exp = {"R1": 1.0, "R2": 2.0}
        r_bound = {"R1": [0.0, 5.0], "R2": [0.5, 3.0]}
        result = plot_Eflux_scatter(r_exp, r_bound)
        assert isinstance(result, dict)
        assert "g" in result
        assert isinstance(result["g"], plt.Figure)


class TestPlotPCA:
    def test_basic(self, pca_data, groups_dict):
        result = plot_PCA(pca_data, groups=groups_dict)
        assert isinstance(result, dict)

    def test_3d(self, pca_data, groups_dict):
        result = plot_PCA(pca_data, groups=groups_dict, plot_2D=False)
        assert isinstance(result, dict)

    def test_with_loading(self, pca_data, groups_dict):
        result = plot_PCA(pca_data, groups=groups_dict, plot_loading=True)
        assert isinstance(result, dict)


# ===================================================================
# BasePlotter class tests
# ===================================================================

class TestBasePlotter:
    def test_plot_func_not_implemented(self):
        plotter = BasePlotter()
        with pytest.raises(NotImplementedError):
            plotter.plot()

    def test_save_fig_method(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "test.png"
        BasePlotter._save_fig(str(out), prefix="", dpi=50, g=fig)
        assert out.exists()

    def test_extract_kws(self):
        kws = {"a": 1, "b": 2}
        result = BasePlotter.extract_kws(kws, ["a"], {"a": 10})
        assert result == {"a": 1}
        assert "a" not in kws

    def test_format_file_name(self):
        kws = {"file_name": "output.png"}
        name = BasePlotter.format_file_name("{file_name}", kws)
        assert name == "output.png"

    def test_format_file_name_none_when_missing(self):
        name = BasePlotter.format_file_name("{file_name}", {})
        assert name is None


class TestFBAPlotter:
    def test_plot(self, fba_flux_df):
        plotter = FBAPlotter(dpi=50)
        result = plotter.plot(flux_df=fba_flux_df, rxn_ids=["R1", "R3"])
        assert result is not None

    def test_save_to_file(self, fba_flux_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        plotter = FBAPlotter(dpi=50)
        # method is needed for the default name_format "{method}_result.png"
        plotter.plot(flux_df=fba_flux_df, rxn_ids=["R1"],
                     file_name="test.png", method="FBA")
        assert (tmp_path / "FBA_FBA_result.png").exists()


class TestFVAPlotter:
    def test_plot(self, fva_flux_df):
        plotter = FVAPlotter(dpi=50)
        result = plotter.plot(flux_df=fva_flux_df, rxn_ids=["R1"])
        assert result is not None


class TestPFBAPlotter:
    def test_prefix(self):
        plotter = pFBAPlotter()
        assert plotter.prefix == "pFBA_"


class TestPercentileThresholdPlotter:
    def test_plot(self):
        plotter = PercentileThresholdPlotter(dpi=50)
        data = pd.Series(np.random.randn(100))
        result = plotter.plot(data=data, exp_th=0.5)
        assert result is not None


class TestRFastCormicThresholdPlotter:
    def test_plot(self):
        plotter = rFastCormicThresholdPlotter(dpi=50)
        x = np.linspace(0, 10, 50)
        y = np.exp(-0.5 * (x - 5) ** 2)
        result = plotter.plot(x=x, y=y, exp_th=6.0, nonexp_th=4.0)
        assert result is not None


class TestCorrelationPlotter:
    def test_plot(self):
        data = pd.DataFrame(np.random.randn(5, 5)).corr()
        plotter = CorrelationPlotter(dpi=50)
        result = plotter.plot(result=data)
        assert result is not None


# ===================================================================
# Additional _utils coverage
# ===================================================================

class TestGetSubsystemTicks:
    def test_basic(self):
        data = pd.DataFrame(
            np.random.randn(6, 3),
            index=["r1", "r2", "r3", "r4", "r5", "r6"],
            columns=["c1", "c2", "c3"],
        )
        rxn_subsystem = {
            "r1": "glycolysis", "r2": "glycolysis", "r3": "TCA",
            "r4": "TCA", "r5": "PPP", "r6": "PPP",
        }
        result_data, subsystems, ticks_pos = _get_subsystem_ticks(data, rxn_subsystem)
        assert len(subsystems) == 3
        assert len(ticks_pos) == 3
        assert "_subsystem" not in result_data.columns
        assert "_index" not in result_data.columns
        assert result_data.shape == (6, 3)

    def test_single_subsystem(self):
        data = pd.DataFrame(np.random.randn(3, 2), index=["a", "b", "c"])
        rxn_sub = {"a": "X", "b": "X", "c": "X"}
        result_data, subsystems, ticks_pos = _get_subsystem_ticks(data, rxn_sub)
        assert subsystems == ["X"]
        assert len(ticks_pos) == 1


class TestSaveFigDecorator:
    def test_decorator_without_file_name(self):
        @save_fig
        def _plot():
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            return {"g": fig}

        result = _plot()
        assert "g" in result

    def test_decorator_with_file_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        @save_fig
        def _plot():
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            return {"g": fig}

        result = _plot(file_name="output.png")
        assert (tmp_path / "output.png").exists()

    def test_decorator_with_prefix_and_dpi(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        @save_fig(prefix="TEST_", dpi=50)
        def _plot():
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            return {"g": fig}

        result = _plot(file_name="result.png")
        assert (tmp_path / "TEST_result.png").exists()

    def test_decorator_returns_none_when_func_returns_none(self):
        @save_fig
        def _plot():
            return None

        result = _plot()
        assert result is None

    def test_decorator_with_name_format_from_func(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        @save_fig
        def _plot():
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            return {"g": fig, "name_format": "{method}_out.png", "method": "FBA"}

        result = _plot(file_name="ignored.png")
        assert (tmp_path / "FBA_out.png").exists()


class TestFormatFileNameExtra:
    def test_name_format_in_plotting_kws_overrides(self):
        kws = {"name_format": "{method}.png", "method": "FBA"}
        result = format_file_name("{file_name}", kws)
        assert result == "FBA.png"


# ===================================================================
# Additional _prep coverage
# ===================================================================

class TestFilterFvaDfVerbosity:
    def test_verbosity_logs(self):
        df = pd.DataFrame({
            "Reaction": ["R1", "R2"],
            "minimum": [-1.0, -0.0001],
            "maximum": [1.0, 0.0001],
        })
        filtered = filter_fva_df(df, threshold=0.001, verbosity=1)
        assert len(filtered) == 1
        assert filtered.iloc[0]["Reaction"] == "R1"


# ===================================================================
# Additional _flux coverage
# ===================================================================

class TestPlotFBAExtra:
    def test_verbosity_logs(self, fba_flux_df):
        result = plot_fba(fba_flux_df, rxn_ids=["R1", "R2", "R3"],
                          filter_all_zeros=True, verbosity=1)
        assert "g" in result

    def test_fva_with_name_format_kwargs(self, fva_flux_df):
        result = plot_fva(fva_flux_df, rxn_ids=["R1"],
                          name_format="{method}_result.png", method="FVA")
        assert "name_format" in result
        assert "method" in result


class TestPlotSamplingDisplotExtra:
    def test_with_limits(self):
        df = pd.DataFrame({
            "R1": np.random.randn(30),
            "group": ["g1"] * 15 + ["g2"] * 15,
        })
        facet = plot_sampling_displot(
            df, rxn_id="R1", kind="kde", group_by="group",
            vertical=True, y_lim=(0, 5), x_lim=(-3, 3),
        )
        assert facet is not None

    def test_with_stat_analysis_placeholder(self):
        df = pd.DataFrame({
            "R1": np.random.randn(30),
            "group": ["g1"] * 15 + ["g2"] * 15,
        })
        facet = plot_sampling_displot(
            df, rxn_id="R1", kind="kde", group_by="group",
            vertical=True, stat_analysis="placeholder",
        )
        assert facet is not None

    def test_horizontal(self):
        df = pd.DataFrame({
            "R1": np.random.randn(30),
            "group": ["g1"] * 15 + ["g2"] * 15,
        })
        facet = plot_sampling_displot(
            df, rxn_id="R1", kind="kde", group_by="group",
            vertical=False,
        )
        assert facet is not None


class TestPlotSamplingCatplotExtra:
    def test_with_limits(self):
        df = pd.DataFrame({
            "R1": np.random.randn(30),
            "group": ["g1"] * 15 + ["g2"] * 15,
        })
        facet = plot_sampling_catplot(
            df, rxn_id="R1", kind="box", group_by="group",
            group_order=["g1", "g2"], vertical=True,
            y_lim=(0, 5), x_lim=(-1, 2),
        )
        assert facet is not None

    def test_with_stat_analysis_significant(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "R1": np.concatenate([np.random.randn(30) + 5, np.random.randn(30)]),
            "group": ["g1"] * 30 + ["g2"] * 30,
        })
        # Mock stat_analysis with a result_df
        stat_mock = MagicMock()
        stat_mock.result_df = pd.DataFrame({
            "label": ["R1"],
            "A": ["g1"],
            "B": ["g2"],
            "adjusted_p_value": [0.001],
        })
        facet = plot_sampling_catplot(
            df, rxn_id="R1", kind="box", group_by="group",
            group_order=["g1", "g2"], vertical=True,
            stat_analysis=stat_mock,
        )
        assert facet is not None

    def test_with_stat_analysis_horizontal(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "R1": np.concatenate([np.random.randn(30) + 5, np.random.randn(30)]),
            "group": ["g1"] * 30 + ["g2"] * 30,
        })
        stat_mock = MagicMock()
        stat_mock.result_df = pd.DataFrame({
            "label": ["R1"],
            "A": ["g1"],
            "B": ["g2"],
            "adjusted_p_value": [0.001],
        })
        facet = plot_sampling_catplot(
            df, rxn_id="R1", kind="box", group_by="group",
            group_order=["g1", "g2"], vertical=False,
            stat_analysis=stat_mock,
        )
        assert facet is not None

    def test_with_stat_analysis_not_significant(self):
        df = pd.DataFrame({
            "R1": np.random.randn(30),
            "group": ["g1"] * 15 + ["g2"] * 15,
        })
        stat_mock = MagicMock()
        stat_mock.result_df = pd.DataFrame({
            "label": ["R1"],
            "A": ["g1"],
            "B": ["g2"],
            "adjusted_p_value": [0.5],
        })
        facet = plot_sampling_catplot(
            df, rxn_id="R1", kind="box", group_by="group",
            group_order=["g1", "g2"], vertical=True,
            stat_analysis=stat_mock,
        )
        assert facet is not None

    def test_stat_analysis_missing_comparison_raises(self):
        df = pd.DataFrame({
            "R1": np.random.randn(30),
            "group": ["g1"] * 15 + ["g2"] * 15,
        })
        stat_mock = MagicMock()
        stat_mock.result_df = pd.DataFrame({
            "label": ["R_OTHER"],
            "A": ["g1"],
            "B": ["g2"],
            "adjusted_p_value": [0.001],
        })
        with pytest.raises(ValueError):
            plot_sampling_catplot(
                df, rxn_id="R1", kind="box", group_by="group",
                group_order=["g1", "g2"], vertical=True,
                stat_analysis=stat_mock,
            )


class TestPlotOneSampling:
    """Tests for the deprecated plot_one_sampling function."""

    def test_kde(self):
        df = pd.DataFrame({
            "R1": np.random.randn(30),
            "grp": ["g1"] * 15 + ["g2"] * 15,
            "n": list(range(15)) * 2,
        })
        color_maps = {"g1": (0.2, 0.4, 0.6), "g2": (0.8, 0.2, 0.3)}
        result = plot_one_sampling(
            flux_df=df, r="R1", color_maps=color_maps,
            grps=["g1", "g2"], file_dir="./",
            group_layer="grp", plotting_style="displot", plotting_kind="kde",
        )
        assert "g" in result

    def test_catplot(self):
        df = pd.DataFrame({
            "R1": np.random.randn(30),
            "grp": ["g1"] * 15 + ["g2"] * 15,
            "n": list(range(15)) * 2,
        })
        color_maps = {"g1": (0.2, 0.4, 0.6), "g2": (0.8, 0.2, 0.3)}
        result = plot_one_sampling(
            flux_df=df, r="R1", color_maps=color_maps,
            grps=["g1", "g2"], file_dir="./",
            group_layer="grp", plotting_style="catplot", plotting_kind="box",
        )
        assert "g" in result


class TestPlotSamplingDeprecated:
    """Tests for the deprecated plot_sampling function."""

    def test_basic(self):
        df1 = pd.DataFrame({"R1": np.random.randn(10), "grp": ["g1"] * 10})
        df2 = pd.DataFrame({"R1": np.random.randn(10), "grp": ["g2"] * 10})
        plot_sampling(
            sampling_flux_df={"s1": df1, "s2": df2},
            rxn_ids=["R1"],
            group_layer="grp",
            plotting_style="displot",
            plotting_kind="kde",
        )

    def test_with_dict_rxn_ids(self):
        df1 = pd.DataFrame({"R1": np.random.randn(10), "grp": ["g1"] * 10})
        df2 = pd.DataFrame({"R1": np.random.randn(10), "grp": ["g2"] * 10})
        plot_sampling(
            sampling_flux_df={"s1": df1, "s2": df2},
            rxn_ids={"R1": "Reaction_A"},
            group_layer="grp",
            plotting_style="displot",
            plotting_kind="kde",
        )


# ===================================================================
# Additional categorical coverage
# ===================================================================

class TestPlotDataCatExtra:
    def test_horizontal(self):
        df = pd.DataFrame({
            "id": ["A", "A", "B", "B"],
            "val": [1, 2, 3, 4],
        })
        result = plot_data_cat(df, id_col="id", val_col="val", ids=["A", "B"],
                               vertical=False)
        assert "g" in result

    def test_log_scale_vertical(self):
        df = pd.DataFrame({
            "id": ["A", "A", "B", "B"],
            "val": [1, 2, 3, 4],
        })
        result = plot_data_cat(df, id_col="id", val_col="val", ids=["A", "B"],
                               log_scale=True, vertical=True)
        assert "g" in result

    def test_log_scale_horizontal(self):
        df = pd.DataFrame({
            "id": ["A", "A", "B", "B"],
            "val": [1, 2, 3, 4],
        })
        result = plot_data_cat(df, id_col="id", val_col="val", ids=["A", "B"],
                               log_scale=True, vertical=False)
        assert "g" in result


class TestPlotModelComponentsExtra:
    def test_many_groups_uses_vertical_layout(self):
        groups = [f"g{i}" for i in range(12)]
        rows = []
        for g in groups:
            for comp in ["n_rxns", "n_mets", "n_genes"]:
                rows.append({"group": g, "component": comp,
                             "number": np.random.randint(10, 100)})
        df = pd.DataFrame(rows)
        result = plot_model_components(df, order=groups)
        assert "g" in result
        assert isinstance(result["g"], plt.Figure)


class TestPlotLocalThresholdBoxplotExtra:
    def test_groups_all(self):
        genes = ["g1", "g2"]
        group_dic = {"A": ["s1", "s2"], "B": ["s3", "s4"]}
        data = pd.DataFrame(
            np.random.randn(2, 4), index=genes, columns=["s1", "s2", "s3", "s4"]
        )
        local_th = pd.DataFrame([[0.5, 0.3], [0.4, 0.2]],
                                 index=genes, columns=["A", "B"])
        global_on = pd.Series({"A": 0.8, "B": 0.7})
        global_off = pd.Series({"A": -0.5, "B": -0.4})
        result = plot_local_threshold_boxplot(
            data, genes, "all", group_dic, local_th, global_on, global_off
        )
        assert "g" in result


# ===================================================================
# Additional heatmap coverage
# ===================================================================

class TestGetQualSize:
    def test_qualitative(self):
        size = _get_qual_size("deep")
        assert isinstance(size, int) and size > 0

    def test_non_qualitative_returns_none(self):
        size = _get_qual_size("Spectral")
        assert size is None


class TestResolvePalette:
    def test_qual_within_limit(self):
        colors = _resolve_palette("deep", 3, "Spectral")
        assert len(colors) >= 3

    def test_qual_exceeds_limit(self):
        colors = _resolve_palette("deep", 200, "Spectral")
        assert len(colors) >= 200

    def test_non_qualitative(self):
        colors = _resolve_palette("viridis", 10, "Spectral")
        assert len(colors) >= 10


class TestParseColors:
    def test_row_and_col_groups(self):
        row_groups = pd.DataFrame({"grp": ["a", "b"]}, index=["r1", "r2"])
        col_groups = pd.DataFrame({"grp": ["x", "y"]}, index=["c1", "c2"])
        row_colors, col_colors = _parse_colors(
            row_groups=row_groups, col_groups=col_groups,
        )
        assert row_colors is not None
        assert col_colors is not None
        assert row_colors.shape == row_groups.shape
        assert col_colors.shape == col_groups.shape

    def test_none_groups(self):
        row_colors, col_colors = _parse_colors()
        assert row_colors is None
        assert col_colors is None


class TestModifyClustermapForSubsys:
    def test_basic(self):
        data = pd.DataFrame(np.random.randn(6, 4))
        g = sns.clustermap(data, figsize=(5, 5))
        ticks_pos = [1, 3, 5]
        subsystems = ["A", "B", "C"]
        _modify_clustermap_for_subsys(g, ticks_pos, subsystems)
        tick_labels = [t.get_text() for t in g.ax_heatmap.yaxis.get_majorticklabels()]
        assert tick_labels == subsystems


class TestPlotClustermap:
    def test_basic(self):
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(10, 4) + np.random.randn(10, 1),
            index=[f"r{i}" for i in range(10)],
            columns=["s1", "s2", "s3", "s4"],
        )
        result = plot_clustermap(data)
        assert "g" in result

    def test_with_model_groups(self):
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(10, 4) + np.random.randn(10, 1),
            index=[f"r{i}" for i in range(10)],
            columns=["s1", "s2", "s3", "s4"],
        )
        model_groups = {"s1": "A", "s2": "A", "s3": "B", "s4": "B"}
        result = plot_clustermap(data, model_groups=model_groups,
                                  group_list=["A", "B"])
        assert "g" in result

    def test_with_row_category(self):
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(6, 3) + np.random.randn(6, 1),
            index=["r1", "r2", "r3", "r4", "r5", "r6"],
            columns=["s1", "s2", "s3"],
        )
        row_category = pd.Series({
            "r1": "glyc", "r2": "glyc", "r3": "TCA",
            "r4": "TCA", "r5": "PPP", "r6": "PPP",
        })
        result = plot_clustermap(data, row_category=row_category,
                                  row_cluster=False)
        assert "g" in result

    def test_without_row_dendrogram(self):
        data = pd.DataFrame(
            np.random.randn(5, 3),
            index=[f"r{i}" for i in range(5)],
            columns=["s1", "s2", "s3"],
        )
        result = plot_clustermap(data, row_dendrogram=False)
        assert "g" in result


class TestPlotHeatmapExtra:
    def test_with_col_groups(self):
        data = pd.DataFrame(np.random.randn(4, 4),
                             index=["a", "b", "c", "d"],
                             columns=["x", "y", "z", "w"])
        col_groups = pd.DataFrame({"grp": ["g1", "g1", "g2", "g2"]},
                                   index=["x", "y", "z", "w"])
        result = plot_heatmap(data, col_groups=col_groups)
        assert "g" in result


# ===================================================================
# Additional scatter coverage
# ===================================================================

class TestPlotPCAExtra:
    def test_with_sheet_file_name(self, pca_data, groups_dict, tmp_path):
        sheet = str(tmp_path / "data.csv")
        result = plot_PCA(pca_data, groups=groups_dict, sheet_file_name=sheet)
        assert (tmp_path / "data_PCA.csv").exists()
        assert (tmp_path / "data_EXP_VAR.csv").exists()
        assert (tmp_path / "data_COMP.csv").exists()

    def test_no_scree_no_score(self, pca_data, groups_dict):
        result = plot_PCA(pca_data, groups=groups_dict,
                          plot_scree=False, plot_score=False)
        assert isinstance(result, dict)


class TestPlotEmbeddingExtra:
    def test_with_sheet_file_name(self, tmp_path):
        df = pd.DataFrame({
            "embedding 1": np.random.randn(4),
            "embedding 2": np.random.randn(4),
        }, index=["s0", "s1", "s2", "s3"])
        groups = {"g1": ["s0", "s1"], "g2": ["s2", "s3"]}
        out = str(tmp_path / "emb.csv")
        result = plot_embedding(df, groups=groups, sheet_file_name=out)
        assert (tmp_path / "emb.csv").exists()

    def test_with_extra_plotting_kwargs(self):
        df = pd.DataFrame({
            "embedding 1": np.random.randn(4),
            "embedding 2": np.random.randn(4),
        }, index=["s0", "s1", "s2", "s3"])
        groups = {"g1": ["s0", "s1"], "g2": ["s2", "s3"]}
        result = plot_embedding(df, groups=groups, file_name="test.png",
                                dpi=50, prefix="emb_")
        assert "g" in result


class TestPlotVolcano:
    def test_returns_none(self):
        result = plot_volcano(None, None, None, None)
        assert result is None


# ===================================================================
# Additional _class plotter coverage
# ===================================================================

class TestBasePlotterExtra:
    def test_plot_returns_none_when_plot_func_returns_none(self):
        class NullPlotter(BasePlotter):
            def plot_func(self, *args, **kwargs):
                return None

        plotter = NullPlotter()
        result = plotter.plot()
        assert result is None


class TestSamplingPlotter:
    def test_plot(self):
        df = pd.DataFrame({
            "R1": np.random.randn(30),
            "group": ["g1"] * 15 + ["g2"] * 15,
        })
        plotter = SamplingPlotter(dpi=50)
        result = plotter.plot(flux_df=df, rxn_id="R1", kind="box",
                              group_by="group", plotting_type="catplot")
        assert result is not None


class TestLocalThresholdPlotterClass:
    def test_plot_box(self):
        genes = ["g1", "g2"]
        groups = ["A", "B"]
        group_dic = {"A": ["s1", "s2"], "B": ["s3", "s4"]}
        data = pd.DataFrame(
            np.random.randn(2, 4), index=genes, columns=["s1", "s2", "s3", "s4"]
        )
        local_th = pd.DataFrame([[0.5, 0.3], [0.4, 0.2]],
                                 index=genes, columns=groups)
        global_on = pd.Series({"A": 0.8, "B": 0.7})
        global_off = pd.Series({"A": -0.5, "B": -0.4})
        plotter = LocalThresholdPlotter(dpi=50)
        result = plotter.plot(
            data=data, genes=genes, groups=groups, group_dic=group_dic,
            local_th=local_th, global_on_th=global_on, global_off_th=global_off,
            kind="box",
        )
        assert result is not None

    def test_plot_invalid_kind_raises(self):
        plotter = LocalThresholdPlotter(dpi=50)
        with pytest.raises(ValueError, match="not implemented"):
            plotter.plot_func(
                data=None, genes=None, groups=None, group_dic=None,
                local_th=None, global_on_th=None, global_off_th=None,
                kind="invalid",
            )


class TestComponentNumberPlotterClass:
    def test_plot(self):
        df = pd.DataFrame({
            "group": ["g1", "g1", "g2", "g2"] * 3,
            "component": ["n_rxns"] * 4 + ["n_mets"] * 4 + ["n_genes"] * 4,
            "number": np.random.randint(10, 100, 12),
        })
        plotter = ComponentNumberPlotter(dpi=50)
        result = plotter.plot(result=df, name_order=["g1", "g2"])
        assert result is not None


class TestComponentComparisonPlotterClass:
    def test_plot_basic(self):
        data = pd.DataFrame(np.random.rand(4, 4),
                             index=["a", "b", "c", "d"],
                             columns=["a", "b", "c", "d"])
        plotter = ComponentComparisonPlotter(dpi=50)
        result = plotter.plot(result=data)
        assert result is not None

    def test_plot_with_row_col_groups(self):
        data = pd.DataFrame(np.random.rand(4, 4),
                             index=["a", "b", "c", "d"],
                             columns=["a", "b", "c", "d"])
        row_groups = pd.DataFrame({"grp": ["g1", "g1", "g2", "g2"]},
                                   index=["a", "b", "c", "d"])
        col_groups = pd.DataFrame({"grp": ["g1", "g1", "g2", "g2"]},
                                   index=["a", "b", "c", "d"])
        plotter = ComponentComparisonPlotter(dpi=50)
        result = plotter.plot(
            result=data,
            row_color_by=["grp"], col_color_by=["grp"],
            row_groups=row_groups, col_groups=col_groups,
        )
        assert result is not None


class TestDimReductionPlotterClass:
    def test_pca(self, pca_data, groups_dict):
        plotter = DimReductionPlotter(dpi=50)
        result = plotter.plot(method="PCA", result_dic=pca_data,
                              groups=groups_dict)
        assert result is not None

    def test_umap(self):
        embedding_df = pd.DataFrame({
            "embedding 1": np.random.randn(4),
            "embedding 2": np.random.randn(4),
        }, index=["s0", "s1", "s2", "s3"])
        groups = {"g1": ["s0", "s1"], "g2": ["s2", "s3"]}
        plotter = DimReductionPlotter(dpi=50)
        result = plotter.plot(
            method="UMAP",
            result_dic={"embeddings": embedding_df},
            groups=groups,
        )
        assert result is not None


class TestHeatmapPlotterClass:
    def test_plot(self):
        data = pd.DataFrame(np.random.randn(5, 5))
        plotter = HeatmapPlotter(dpi=50)
        result = plotter.plot(result=data)
        assert result is not None


class TestDataCatPlotterClass:
    def test_plot(self):
        df = pd.DataFrame({
            "id": ["A", "A", "B", "B"],
            "val": [1, 2, 3, 4],
            "grp": ["x", "y", "x", "y"],
        })
        plotter = DataCatPlotter(dpi=50)
        result = plotter.plot(
            data=df, id_col="id", value_col="val",
            hue="grp", ids=["A", "B"], vertical=True,
        )
        assert result is not None
