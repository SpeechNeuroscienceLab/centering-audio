import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.legend_handler import HandlerTuple
from matplotlib import gridspec
import numpy as np
from scipy import signal

from matplotlib.markers import TICKDOWN


def standard_error(data):
    return np.std(data, ddof=1) / np.sqrt(np.size(data))


def mark_significance_bar(dataset: pd.DataFrame, figure: plt.Figure, pairs, plot_settings: dict, height=0.0,
                          display_string="*",
                          group_column="Group Name"):
    # default to the first axes
    axes = figure.get_axes()[0]

    plot_indices = {}
    xpos = []
    prev = 0.0

    for group in plot_settings["plot_order"]:
        group_data = dataset[dataset[group_column] == group]

        for tercile in sorted(group_data["Tercile"].unique()):
            xpos.append(prev)
            prev += plot_settings["bar_spacing"]
            plot_indices[(group, tercile)] = xpos[-1]  # position of each group/tercile combination
        prev += plot_settings["group_spacing"]

    __significance_bar(axes, plot_indices[pairs[0]], plot_indices[pairs[1]], max(axes.get_ylim()) + height,
                       display_string=display_string)


def __significance_bar(axes: plt.Axes, start, end, height, display_string,
                       line_width=1.5, marker_size=8, box_pad=0.3, fontsize=15, color='k'):
    axes.plot([start, end], [height] * 2, '-',
              color=color, lw=line_width, marker=TICKDOWN, markeredgewidth=line_width, markersize=marker_size)
    # draw the text with a bounding box covering up the line
    axes.text(0.5 * (start + end), height, display_string, ha='center', va='center',
              bbox=dict(facecolor='1.', edgecolor='none', boxstyle='Square,pad=' + str(box_pad)), size=fontsize)


def group_tercile_centering_bars(dataset: pd.DataFrame, figure: plt.Figure, plot_settings: dict,
                                 group_column="Group Name", quantity_column="Centering (Cents)"):
    # same approach as before: precompute attributes before assembly
    # default to the first axes
    axes = figure.get_axes()[0]

    plot_data = []
    plot_indices = {}
    xpos = []
    prev = 0.0

    for group in plot_settings["plot_order"]:
        group_data = dataset[dataset[group_column] == group]

        for tercile in sorted(group_data["Tercile"].unique()):
            plot_data.append({
                "group": group,
                "tercile": plot_settings["label_alias"][tercile],
                "height": np.nanmean(group_data[group_data["Tercile"] == tercile][quantity_column]),
                "error": standard_error(group_data[group_data["Tercile"] == tercile][quantity_column]),
                "color": plot_settings["colormap"][group][tercile]
            })

            xpos.append(prev)
            prev += plot_settings["bar_spacing"]

            plot_indices[(group, tercile)] = xpos[-1]  # position of each group/tercile combination

        prev += plot_settings["group_spacing"]

    axes.bar(x=xpos,
             height=[bar["height"] for bar in plot_data],
             color=[bar["color"] for bar in plot_data],
             alpha=0.9,
             edgecolor='black',
             )

    axes.set_xticks(ticks=xpos)
    axes.set_xticklabels([bar["tercile"] for bar in plot_data], rotation=0, ha="center")
    axes.tick_params(axis='x', length=0)
    axes.set_ylabel(quantity_column)

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
        patches.append([mpatches.Patch(facecolor=c, edgecolor='black', label=cat) for c in col])
    axes.legend(handles=patches,
                labels=plot_settings["plot_order"],
                frameon=False,
                loc="upper left",
                handler_map={list: HandlerTuple(None)},
                prop={'size': 12},
                )


def group_pitch_distributions(dataset: pd.DataFrame, figure: plt.Figure,
                              plot_settings: dict,
                              group_column="Group Name",
                              distribution_columns=("Starting Pitch (Cents)", "Ending Pitch (Cents)"),
                              smooth_on=False):
    legend_height_ratio = 0.2

    axes = figure.subplots(1 + len(dataset[group_column].unique()), 1, sharex=True, sharey=True,
                           gridspec_kw={
                               'height_ratios': [legend_height_ratio] + [1] * len(dataset[group_column].unique())})
    figure.tight_layout(h_pad=1.5)

    filter_length = 70
    pitch_granularity = 2
    bin_size = 20
    xrange = (-400, 400)

    bins = np.arange(xrange[0], xrange[1], bin_size)

    # set up legend
    legend = axes[0]
    legend.axis('off')
    patches = [mpatches.Patch(facecolor=plot_settings["colormap"][col], edgecolor='black') for col in
               distribution_columns]
    legend.legend(handles=patches,
                  labels=distribution_columns,
                  frameon=False,
                  loc="upper center",
                  handler_map={list: HandlerTuple(None)},
                  prop={'size': 12},
                  ncol=len(distribution_columns))

    for ax, group in zip(axes[1:], dataset[group_column].unique()):
        ax.axhline(0, color="black")  # add a 0 axis line
        for column in distribution_columns:
            hist, bin_edges = np.histogram(dataset[dataset[group_column] == group][column], bins=bins, density=True)

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            hist_upsampled = np.interp(np.arange(xrange[0], xrange[1], pitch_granularity), bin_centers, hist)

            # parzen kernel
            parzen = signal.windows.parzen(filter_length)
            normalized_parzen = parzen / np.sum(parzen)

            smooth_hist = np.convolve(normalized_parzen, hist_upsampled, mode='same')

            # savitzky_golay filter
            # smooth_hist = signal.savgol_filter(hist, filter_length, polyorder=4)

            x = np.linspace(xrange[0], xrange[1], len(smooth_hist))

            if smooth_on:
                ax.fill(x, smooth_hist, alpha=0.5, color=plot_settings["colormap"][column])
            else:
                ax.bar(bin_centers, hist,
                       alpha=0.5, color=plot_settings["colormap"][column], edgecolor=plot_settings["colormap"][column],
                       width=np.mean(bin_edges[:-1] - bin_edges[1:]))
        ax.set_title(group)
        ax.set_yticks([])


def group_centering_distribution(dataset: pd.DataFrame, figure: plt.Figure,
                                 plot_settings: dict,
                                 groups: tuple,
                                 group_column="Group Name",
                                 distribution_columns=("Starting Pitch (Cents)", "Ending Pitch (Cents)"),
                                 title=None):
    # note: axes should match the number of groups
    axes = figure.get_axes()

    for i, group in enumerate(groups):
        group_data = dataset[dataset[group_column] == group]
        axis = axes[i]

        if i != 0:
            axis.set_xticks([])

        axis.set_ylabel(group)
        axis.set_yticks([])

        ordering = np.abs(group_data[distribution_columns[0]]).sort_values().index

        initial_dev = np.abs(group_data[distribution_columns[0]])
        sign_correction = initial_dev / group_data[distribution_columns[0]]

        initial_dev = initial_dev[ordering]
        mid_dev = (sign_correction * group_data[distribution_columns[1]])[ordering]

        initial_dev.reset_index(drop=True, inplace=True)
        mid_dev.reset_index(drop=True, inplace=True)

        for i, (startx, endx) in enumerate(zip(initial_dev, mid_dev)):
            axis.plot((startx, endx), (i, i),
                      color="green" if np.abs(startx) > np.abs(endx) else ("violet" if startx > endx else "red"),
                      alpha=0.75,
                      lw=2)
        axis.vlines(0, 0, initial_dev.shape[0])

    if title is not None:
        axes[0].set_xlabel(f"Pitch (cents) {title}")

