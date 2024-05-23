import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.legend_handler import HandlerTuple
import numpy as np


def standard_error(data):
    return np.std(data, ddof=1) / np.sqrt(np.size(data))


def group_tercile_centering_bars(dataset: pd.DataFrame, axes: plt.Axes, plot_settings: dict, group_column="Group Name"):
    # same approach as before: precompute attributes before assembly
    plot_data = []
    xpos = []
    prev = 0.0

    for group in plot_settings["plot_order"]:
        group_data = dataset[dataset[group_column] == group]

        for tercile in group_data["Tercile"].unique():
            plot_data.append({
                "group": group,
                "tercile": plot_settings["label_alias"][tercile],
                "height": np.nanmean(group_data[group_data["Tercile"] == tercile]["Centering (Cents)"]),
                "error": standard_error(group_data[group_data["Tercile"] == tercile]["Centering (Cents)"]),
                "color": plot_settings["colormap"][group][tercile]
            })

            xpos.append(prev)
            prev += plot_settings["bar_spacing"]
        prev += plot_settings["group_spacing"]

    y = [bar["height"] + bar["error"] for bar in plot_data]

    axes.bar(x=xpos,
             height=[bar["height"] for bar in plot_data],
             color=[bar["color"] for bar in plot_data],
             alpha=0.9,
             edgecolor='black',
             )

    axes.set_xticks(ticks=xpos)
    axes.set_xticklabels([bar["tercile"] for bar in plot_data], rotation=0, ha="center")
    axes.tick_params(axis='x', length=0)
    axes.set_ylabel("Centering (Cents)")

    axes.errorbar(x=xpos,
                  y=[bar["height"] for bar in plot_data],
                  yerr=[bar["error"] for bar in plot_data],
                  color=plot_settings["error-color"],
                  elinewidth=plot_settings["line-width"],
                  capthick=plot_settings["line-width"],
                  capsize=plot_settings["error-cap-size"],
                  ls=plot_settings["error-line-style"])
    legend_dict = dict(zip(plot_settings["plot_order"],
                           [list(plot_settings["colormap"][group].values()) for group in plot_settings["colormap"]]
                           ))
    patches = []
    for cat, col in legend_dict.items():
        patches.append([mpatches.Patch(facecolor=c, label=cat) for c in col])
    axes.legend(handles=patches,
                labels=plot_settings["plot_order"],
                frameon=False,
                loc="upper left",
                handler_map={list: HandlerTuple(None)},
                prop={'size': 12}
                )
