from functools import wraps, partial
from pathlib import Path
import string

import matplotlib.pyplot as plt
import pandas as pd


def _get_subsystem_ticks(data: pd.DataFrame, rxn_subsystem):
    assert "_subsystem" not in data.columns.to_list() and "_index" not in data.columns.to_list()
    data["_subsystem"] = data.index.to_series().apply(lambda x: rxn_subsystem[x])
    data = data.sort_values("_subsystem")
    data["_index"] = pd.RangeIndex(0, len(data.index.to_list())).to_list()
    subsystems = list(data["_subsystem"].unique())
    first_ind = data.drop_duplicates(subset=["_subsystem"])["_index"].to_list()
    ticks_pos = [(first_ind[i] + first_ind[i + 1]) // 2 for i in range(len(first_ind) - 1)] + \
                [(first_ind[-1] + len(data["_index"])) // 2]
    assert len(subsystems) == len(ticks_pos)
    data = data.drop("_subsystem", axis=1)
    data = data.drop("_index", axis=1)
    return data, subsystems, ticks_pos


def _set_default_ax(ax,
                    x_label,
                    y_label,
                    face_color=(1, 1, 1, 0.4),
                    title=None,
                    with_legend=True,
                    z_label=None):
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)
    if title is not None:
        ax.set_title(title)
    ax.set_facecolor(face_color)
    if with_legend:
        ax.legend()
    ax.grid()
    plt.tight_layout()
    return ax


def _save_fig(file_name, prefix, dpi=150, g=None):
    if prefix is None:
        prefix = ""

    if isinstance(file_name, str) and "/" in file_name:
        file_path = Path("/".join(file_name.split("/")[:-1]))
        file_name = Path(prefix + file_name.split("/")[-1])
    else:
        file_name, file_path = Path(prefix + str(file_name)), Path("./")

    if g is None:
        plt.savefig(file_path / file_name, dpi=dpi, bbox_inches='tight')
    else:
        g.savefig(file_path / file_name, dpi=dpi, bbox_inches='tight')


def extract_kws(kws: dict, keys, default_kws):
    extracted = {}
    for k in keys:
        default = default_kws.get(k)
        extracted[k] = kws.pop(k, default)
    return extracted


def format_file_name(name_format, plotting_kws):
    if "name_format" in plotting_kws:
        name_format = plotting_kws.pop("name_format")

    sf = string.Formatter()
    format_kws = [x[1] for x in sf.parse(name_format) if x[1] is not None]
    if all([f in plotting_kws for f in format_kws]):
        naming_kws = {f: plotting_kws.pop(f) for f in format_kws}
        return name_format.format(**naming_kws) if all([v is not None for v in naming_kws.values()]) else None
    return None


def save_fig(func=None, *, prefix="", dpi=150, name_format="{file_name}"):
    if func is None:
        return partial(save_fig, prefix=prefix, dpi=dpi, name_format=name_format)

    @wraps(func)
    def plot_the_result(*args, **kwargs):
        plotting_kws = extract_kws(kwargs,
                                   ["file_name", "prefix", "dpi"],
                                   default_kws={"dpi": dpi, "prefix": prefix})

        info_kws = func(*args, **kwargs)
        name_format = info_kws.pop("name_format") if "name_format" in info_kws else "{file_name}"
        if info_kws is not None:
            plotting_kws.update(info_kws)
            updated_name = format_file_name(name_format, plotting_kws)
            plotting_kws["file_name"] = updated_name
            if plotting_kws["file_name"] is not None:
                print("saving ",plotting_kws["file_name"])
                _save_fig(**plotting_kws)
            plt.show()
        return info_kws
    return plot_the_result


def draw_significance(ax, x_pos_list, y_pos_list, num_stars):
    # check the line is valid
    if (y_pos_list[0] == y_pos_list[1]) == (x_pos_list[0] == x_pos_list[1]):
        raise ValueError('The line is neither a vertical nor a horizontal line.')

    ax.plot(x_pos_list, y_pos_list, c=(0.2, 0.2, 0.2), lw=3)
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    x_len, y_len = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]

    # horizontal
    if y_pos_list[0] == y_pos_list[1]:
        ax.text((x_pos_list[0] + x_pos_list[1]) / 2, y_pos_list[1] + y_len * 0.02, '*' * num_stars, size=20,
                horizontalalignment='center', verticalalignment='center')

    # vertical
    if x_pos_list[0] == x_pos_list[1]:
        ax.text(x_pos_list[1] + x_len * 0.02, (y_pos_list[0] + y_pos_list[1]) / 2, '*' * num_stars, size=20,
                horizontalalignment='center', verticalalignment='center', rotation=90)