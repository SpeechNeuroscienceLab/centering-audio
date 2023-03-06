import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.style as style


class StatisticsHelpers:
    def standard_error(data):
        return np.std(data, ddof=1) / np.sqrt(np.size(data))


def __default_colormap():
    return {"AD Patients": "darkgoldenrod", "Controls": "teal"}


def __global_styles():
    style.use('classic')


def group_tercile_centering_bars(group_dictionary: dict, dataframe: pd.DataFrame,
                                 colormap=None):
    # default colormap settings
    if colormap is None:
        colormap = __default_colormap()

    group_tercile_centering_bar_figure = plt.figure()

    # formatting options
    plt.title("Average Peripheral Centering (Cents)")

    bar_labels = []
    bar_heights = []
    bar_colors = []
    error_heights = []

    for group, _ in group_dictionary.items():
        group_data = dataframe[dataframe["Group"] == group]
        for tercile in group_data["Tercile"].drop_duplicates():
            bar_labels.append(f"{group}\n{tercile}")
            bar_colors.append(colormap[group])

            tercile_data = group_data[group_data["Tercile"] == tercile]
            bar_heights.append(np.nanmean(tercile_data["Centering"]))
            error_heights.append(StatisticsHelpers.standard_error(tercile_data["Centering"]))

    # draw the figure
    bars = plt.bar(range(len(bar_labels)), bar_heights)
    plt.errorbar(range(len(bar_labels)), bar_heights, yerr=error_heights, color="black", capsize=5, ls='')

    # set the color of the bars
    for i, bar in enumerate(bars):
        bars[i].set_color(bar_colors[i])

    # add the labels
    plt.xticks(range(len(bar_labels)), bar_labels)

    group_tercile_centering_bar_figure.name = "group_tercile_centering_bar_figure"
    return group_tercile_centering_bar_figure


def group_centering_bars(group_dictionary: dict, dataframe: pd.DataFrame,
                         colormap=None):
    # default colormap settings
    if colormap is None:
        colormap = __default_colormap()

    group_centering_bar_figure = plt.figure()

    # formatting options
    plt.title("Average Peripheral Centering (Cents)")

    bar_labels = []
    bar_heights = []
    bar_colors = []
    error_heights = []

    for group, _ in group_dictionary.items():
        group_data = dataframe[dataframe["Group"] == group]
        bar_labels.append(group)
        bar_colors.append(colormap[group])

        bar_heights.append(np.nanmean(group_data["Centering"]))
        error_heights.append(StatisticsHelpers.standard_error(group_data["Centering"]))

    # draw the figure
    bars = plt.bar(range(len(bar_labels)), bar_heights)
    plt.errorbar(range(len(bar_labels)), bar_heights, yerr=error_heights, color="black", capsize=5, ls='')

    # set the color of the bars
    for i, bar in enumerate(bars):
        bars[i].set_color(bar_colors[i])

    # add the labels
    plt.xticks(range(len(bar_labels)), bar_labels)

    group_centering_bar_figure.name = "group_centering_bar_figure"
    return group_centering_bar_figure


def group_scatter(group_dictionary, dataframe,
                  colormap=None):
    if colormap is None:
        colormap = __default_colormap()

    group_scatter_figure = plt.figure()

    # formatting options
    plt.title("Peripheral Centering (Cents)")

    for group, _ in group_dictionary.items():
        group_data = dataframe[dataframe["Group"] == group]
        plt.scatter(group_data["InitialPitch"], group_data["EndingPitch"], label=group, color=colormap[group])

    plt.legend()

    group_scatter_figure.name = "group_scatter_figure"
    return group_scatter_figure


def group_tercile_arrows(group_dictionary, dataframe,
                         colormap=None, hide_names=True):
    if colormap is None:
        colormap = {"UPPER": "black", "CENTRAL": "grey", "LOWER": "black"}

    group_tercile_arrows_figure, axes = plt.subplots(1, len(group_dictionary))

    # set figure size manually, just for this case
    group_tercile_arrows_figure.set_size_inches(9, 5)
    group_tercile_arrows_figure.tight_layout()
    group_tercile_arrows_figure.subplots_adjust(bottom=0.1, left=0.06)

    for group_idx, (group, subjects) in enumerate(group_dictionary.items()):
        axis = axes[group_idx]
        group_data = dataframe[dataframe["Group"] == group]
        for subject_idx, subject in enumerate(subjects):
            # TODO: make a cleaner way of iterating through subjects to get the subject name
            subject_data = group_data[group_data["Subject"] == f"{group.replace(' ', '')}{subject_idx}"]

            for tercile in subject_data["Tercile"].drop_duplicates():
                tercile_data = subject_data[subject_data["Tercile"] == tercile]

                base = np.mean(tercile_data["InitialPitch"])
                apex = np.mean(tercile_data["EndingPitch"])

                coordinates = np.array([[base, subject_idx - 0.25], [base, subject_idx + 0.25], [apex, subject_idx]])

                # draw a filled triangle with these coordinates
                axis.fill(coordinates[:, 0], coordinates[:, 1], color=colormap[tercile], alpha=0.75)

        axis.set_title(f"{group} (n={len(subjects)})")
        axis.set_ylabel("Subject Number")
        axis.set_ylim(-1, len(subjects))
        axis.set_xlabel("Pitch (Cents)")
        if not hide_names:
            axis.set_yticks(range(len(subjects)), subjects)

    group_tercile_arrows_figure.name = "group_tercile_arrows_figure"
    return group_tercile_arrows_figure

# global settings
__global_styles()
