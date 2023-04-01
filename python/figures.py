import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from scipy.stats import norm
from experiment import Experiment


class StatisticsHelpers:
    @staticmethod
    def standard_error(data):
        return np.std(data, ddof=1) / np.sqrt(np.size(data))


class Annotations:
    @staticmethod
    def bar_significance(axes: plt.axes, x, y, label: str, rise=0):
        axes.plot([x[0], x[0], x[1], x[1]], [y[0], max(y) + rise, max(y) + rise, y[1]],
                  color=defaults["annotation-color"])
        # add tick mark
        axes.plot([np.mean(x)] * 2,
                  [max(y) + rise, max(y) + rise + defaults["annotation-vertical-padding"]],
                  color=defaults["annotation-color"])
        axes.text(np.mean(x), max(y) + rise + 2 * defaults["annotation-vertical-padding"], label,
                  horizontalalignment="center",
                  size=defaults["annotation-text-size"], )


def __default_group_colormap():
    return {"AD Patients": "darkgoldenrod", "Controls": "teal"}


defaults = {
    "line-width": 1.5,
    # annotations
    "annotation-text-size": 16,
    "annotation-color": "black",
    "bar-significance-rise": 2,
    "annotation-vertical-padding": 0.5,
    # bar plots
    "bar-group-spacing": 0.25,
    # error bar
    "error-color": "black",
    "error-cap-size": 10,
    "error-line-style": ''
}


def __global_styles():
    style.use('classic')


class Figure:
    def __init__(self, colormap: dict = None):
        if colormap is None:
            self.colormap = {"AD Patients": "darkgoldenrod", "Controls": "teal"}
        else:
            self.colormap = colormap

        self.figure, self.axes = plt.subplots()
        self.name = None

    # All plotting functions must have a render function
    def render(self):
        pass

    def save(self, output_directory: str):
        self.figure.savefig(f"{output_directory}/{self.name}.png")


class GroupTercileCenteringBars(Figure):
    def __init__(self, experiment: Experiment, plot_order: list = None,
                 colormap: dict = None, render: bool = True):
        Figure.__init__(self, colormap=colormap)

        self.bar_x = None
        self.bar_y = None
        self.experiment = experiment

        self.axes.set_title("Average Peripheral Centering (Cents)")
        self.name = "group_tercile_centering_bar_figure"

        self.fill_map = {
            "UPPER": ' ',
            "LOWER": '/'
        }

        self.label_map = {
            "UPPER": "Lowering Pitch",
            "LOWER": "Raising Pitch"
        }

        self.plot_order = [group for group in experiment.subjects] if plot_order is None else plot_order

        if render:
            self.render()

    def render(self):
        # (label, height, error, color, fill-style)
        bars = []

        for group in self.plot_order:
            group_data = self.experiment.df[self.experiment.df["Group"] == group]
            for tercile in group_data["Tercile"].drop_duplicates():
                tercile_data = group_data[group_data["Tercile"] == tercile]
                bars.append({
                    "group": group,
                    "label": tercile,
                    "height": np.nanmean(tercile_data["Centering"]),
                    "error": StatisticsHelpers.standard_error(tercile_data["Centering"]),
                    "color": self.colormap[group],
                    "fill-style": self.fill_map[tercile]
                })

        self.bar_x = np.arange(len(bars), dtype=np.float64)
        self.bar_y = [bar["height"] + bar["error"] for bar in bars]

        # insert spaces between groups
        for i in range(len(bars)):
            if i > 0 and bars[i - 1]["group"] != bars[i]["group"]:
                self.bar_x[i:] += defaults["bar-group-spacing"]

        self.axes.bar(x=self.bar_x,
                      height=[bar["height"] for bar in bars],
                      color=[bar["color"] for bar in bars],
                      hatch=[bar["fill-style"] for bar in bars]
                      )

        self.axes.set_xticks(ticks=self.bar_x,
                             labels=[self.label_map[bar["label"]] for bar in bars])
        self.axes.tick_params(axis='x', length=0)

        self.axes.set_ylabel("Centering (Cents)")

        self.axes.errorbar(x=self.bar_x,
                           y=[bar["height"] for bar in bars],
                           yerr=[bar["error"] for bar in bars],
                           color=defaults["error-color"],
                           elinewidth=defaults["line-width"],
                           capthick=defaults["line-width"],
                           capsize=defaults["error-cap-size"],
                           ls=defaults["error-line-style"])

        self.axes.legend(handles=[patches.Patch(color=self.colormap[group], label=group)
                                  for group in self.experiment.subjects],
                         loc="upper left")

    def annotate_significance(self, x, label):
        Annotations.bar_significance(axes=self.axes,
                                     x=[self.bar_x[i] for i in x],
                                     y=[self.bar_y[i] + defaults["annotation-vertical-padding"] for i in x],
                                     rise=defaults["bar-significance-rise"],
                                     label=label)


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
            subject_data = group_data[group_data["Subject"] == subject]

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


def group_pitch_normal(group_dictionary: dict, dataframe: pd.DataFrame, colormap=None, density=10):
    if colormap is None:
        colormap = {"InitialPitch": "blue", "EndingPitch": "red"}

    time_windows = {"InitialPitch": "Initial Pitch", "EndingPitch": "Mid-trial Pitch"}

    group_pitch_normal_figure, axes = plt.subplots(len(group_dictionary), 1, sharex="all")

    # set figure size manually, just for this case
    group_pitch_normal_figure.tight_layout()
    group_pitch_normal_figure.subplots_adjust(top=0.90, bottom=0.1)

    limits = (min(dataframe["InitialPitch"].min(), dataframe["EndingPitch"].min()),
              max(dataframe["InitialPitch"].max(), dataframe["EndingPitch"].max()))

    for group_idx, (group, subjects) in enumerate(group_dictionary.items()):
        # place each group on its own subplot
        axis = axes[group_idx]
        group_data = dataframe[dataframe["Group"] == group]

        axis.set_ylabel(group)
        axis.set_yticks([])
        axis.set_xlim(limits)

        for index, label in time_windows.items():
            mean, std = (np.mean(group_data[index]), np.std(group_data[index]))
            # density ensures that rescaling the figure's x-axis won't ruin our smooth-ness of our plot
            axis.fill(np.linspace(limits[0], limits[1], num=round((limits[1] - limits[0]) / density)),
                      norm.pdf(np.linspace(limits[0], limits[1], num=round((limits[1] - limits[0]) / density)),
                               mean, std),
                      color=colormap[index], alpha=0.4, label=label)

        axis.legend(loc="upper left")

    # only label the bottom-most value
    axes[-1].set_xlabel("Pitch (Cents)")
    group_pitch_normal_figure.suptitle("Group Pitch Normal Distributions", fontweight="bold")
    group_pitch_normal_figure.name = "group_pitch_normal"
    return group_pitch_normal_figure


def group_pitch_histogram(group_dictionary: dict, dataframe: pd.DataFrame, bin_width=30, colormap=None):
    if colormap is None:
        colormap = __default_group_colormap()

    time_windows = {"InitialPitch": "Initial Pitch", "EndingPitch": "Mid-trial Pitch"}

    group_pitch_histogram_figure, axes = plt.subplots(len(group_dictionary) * len(time_windows), 1, sharex="all")
    axes_iter = iter(axes)

    # set figure size manually, just for this case
    group_pitch_histogram_figure.set_size_inches(5, 8)
    group_pitch_histogram_figure.tight_layout()
    group_pitch_histogram_figure.subplots_adjust(top=0.93, bottom=0.1)

    limits = (min(dataframe["InitialPitch"].min(), dataframe["EndingPitch"].min()),
              max(dataframe["InitialPitch"].max(), dataframe["EndingPitch"].max()))

    # create even zero aligned bins
    bins = np.arange(limits[0], limits[1], bin_width)

    for group_idx, (group, subjects) in enumerate(group_dictionary.items()):
        for index, label in time_windows.items():
            axis = next(axes_iter)
            group_data = dataframe[dataframe["Group"] == group]

            axis.hist(group_data[index], color=colormap[group], bins=bins, alpha=0.75)

            axis.set_ylabel(f"{group}\n{label}")
            axis.set_xlim(limits)
            axis.set_yticks([])

    group_pitch_histogram_figure.suptitle("Group Pitch Distributions", fontweight="bold")
    group_pitch_histogram_figure.name = "group_pitch_histogram"
    return group_pitch_histogram_figure


# global settings
__global_styles()
