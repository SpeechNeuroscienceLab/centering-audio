import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.patches as patches
import matplotlib.colors as colors
import colorsys
import pandas as pd
import numpy as np
from scipy.stats import norm
from experiment import Experiment


class ColorUtils:
    @staticmethod
    def alter_brightness(color, scale: float):
        assert -1 <= scale <= 1
        color_hls = colorsys.rgb_to_hls(*colors.to_rgb(colors.cnames[color] if color in colors.cnames else color))
        return colorsys.hls_to_rgb(color_hls[0],
                                   float(np.clip(color_hls[1] * (1 + scale), 0, 1)),
                                   color_hls[2])


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

    @staticmethod
    def plot_bar(axes: plt.axes, start, end, dashed_end, color="black", line_width=2):
        # first, draw the bar itself
        axes.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=line_width)
        axes.plot([end[0], dashed_end], [end[1], end[1]], color=color, linestyle="dashed")


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
    "bar-lighten-factor": -0.2,
    # error bar
    "error-color": "black",
    "error-cap-size": 10,
    "error-line-style": '',
    # normal distribution
    "normal-dist-density": 1,
    "hist-outline-color": "black"
}


def __global_styles():
    style.use('classic')


class Figure:
    def __init__(self, colormap: dict = None, fig_size=(6.4, 4.8), subplots=(1, 1), shared_axis=("none", "none"),
                 **kwargs):
        if colormap is None:
            self.colormap = {"AD Patients": "darkgoldenrod", "Controls": "teal"}
        else:
            self.colormap = colormap

        self.figure, self.axes = plt.subplots(nrows=subplots[0], ncols=subplots[1],
                                              sharex=shared_axis[0], sharey=shared_axis[1],
                                              figsize=fig_size,
                                              **kwargs)
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

        self.name = "group_tercile_centering_bar_figure"

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
            for tercile_index, tercile in enumerate(group_data["Tercile"].drop_duplicates()):
                tercile_data = group_data[group_data["Tercile"] == tercile]
                bars.append({
                    "group": group,
                    "label": tercile,
                    "height": np.nanmean(tercile_data["Centering"]),
                    "error": StatisticsHelpers.standard_error(tercile_data["Centering"]),
                    "color": self.colormap[group],
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
                      alpha=0.9,
                      hatch=["/", " ", "/", ""]
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


class GroupTercileArrows(Figure):
    def __init__(self, experiment: Experiment, plot_order: list = None,
                 colormap: dict = None, render: bool = True):
        Figure.__init__(self, colormap=colormap, subplots=(1, len(experiment.subjects)))

        self.experiment = experiment
        self.name = "group_tercile_arrows_figure"

        self.plot_order = [group for group in experiment.subjects] if plot_order is None else plot_order

        self.opacity_map = {
            "UPPER": 0.75,
            "CENTRAL": 0.5,
            "LOWER": 0.75
        }

        if render:
            self.render()

    def render(self):
        # set figure size manually, just for this case
        self.figure.set_size_inches(9, 5)
        self.figure.tight_layout()
        self.figure.subplots_adjust(wspace=0.3)
        self.figure.subplots_adjust(bottom=0.1, left=0.07)

        for group_idx, group in enumerate(self.plot_order):
            axis = self.axes[group_idx]
            group_data = self.experiment.df[self.experiment.df["Group"] == group]
            subjects = self.experiment.subjects[group]
            for subject_idx, subject in enumerate(subjects):
                subject_data = group_data[group_data["Subject"] == subject]

                for tercile in subject_data["Tercile"].drop_duplicates():
                    tercile_data = subject_data[subject_data["Tercile"] == tercile]

                    base = np.mean(tercile_data["InitialPitch"])
                    apex = np.mean(tercile_data["EndingPitch"])

                    coordinates = np.array([[base, subject_idx - 0.25],
                                            [base, subject_idx + 0.25],
                                            [apex, subject_idx]])

                    # draw a filled triangle with these coordinates
                    axis.fill(coordinates[:, 0], coordinates[:, 1],
                              color=self.colormap[group],
                              alpha=self.opacity_map[tercile],
                              edgecolor="black")

            axis.set_title(f"{group} (n={len(subjects)})")
            axis.set_ylabel("Subject Number")
            axis.set_ylim(-1, len(subjects))
            axis.set_xlabel("Pitch (Cents)")


class CenteringMethods(Figure):
    def __init__(self, motion_points: list, render: bool = True):
        Figure.__init__(self, subplots=(1, 2), gridspec_kw={'width_ratios': [8, 1]})

        self.name = "sample_trial"
        self.motion_points = motion_points

        if render:
            self.render()

    def render(self):
        self.figure.tight_layout()
        self.figure.subplots_adjust(bottom=0.1, left=0.1, wspace=0.05)

        axis = self.axes[0]
        motion_points = self.motion_points

        # arbitrary deg polynomial fit to data
        smooth_motion_function = np.poly1d(np.polyfit(motion_points[0], motion_points[1], 3))

        axis.set_xlim([0, 200])
        axis.set_ylabel("Pitch (cents)")
        axis.set_xlabel("Time (ms)")

        plot_domain = np.linspace(0, 200, 100)

        smooth_motion_points = smooth_motion_function(plot_domain)

        # add noise from gaussian
        noisy_motion_points = smooth_motion_points + np.random.normal(smooth_motion_points, 15)

        axis.plot(plot_domain, noisy_motion_points, color="black", linewidth=1, alpha=0.5)

        # add the "center" line
        axis.plot(plot_domain, np.zeros(plot_domain.shape), color="black")

        # compute the window averages
        mean_pitches = []
        windows = [(0, 50), (150, 200)]
        for window in windows:
            mean_pitches.append(np.mean(noisy_motion_points[(plot_domain >= window[0])
                                                            * (plot_domain <= window[1])]))
        # add annotations for the mean markers
        Annotations.plot_bar(axis,
                             start=(0.5, mean_pitches[0]), end=(50, mean_pitches[0]), dashed_end=200)
        Annotations.plot_bar(axis, start=(150, mean_pitches[1]), end=(199.5, mean_pitches[1]),
                             dashed_end=200)

        # rescale the y-axis to fit entire plot cleanly
        axis.set_ylim([min(min(noisy_motion_points) - 20, -20),
                       max(max(noisy_motion_points) + 20, 20)])

        plot_height = axis.get_ylim()[1] - axis.get_ylim()[0]

        # add rectangular overlay for windows
        axis.add_patch(patches.Rectangle((0, 0), 50, plot_height, color="gray", alpha=0.25))
        axis.add_patch(patches.Rectangle((150, 0), 50, plot_height, color="gray", alpha=0.25))

        # add text labels
        axis.text(25, 5, "Initial", horizontalalignment="center")
        axis.text(175, 5, "Mid-trial", horizontalalignment="center")

        # set up the second axis
        annotation_axis = self.axes[1]
        # match scales
        annotation_axis.set_ylim(axis.get_ylim())

        for spines in ['top', 'right', 'bottom', 'left']:
            annotation_axis.spines[spines].set_visible(False)

        annotation_axis.get_xaxis().set_ticks([])
        annotation_axis.get_yaxis().set_ticks([])
        # plot the triangle
        TRIANGLE_WIDTH = 40
        coordinates = np.array([[100 - TRIANGLE_WIDTH / 2, mean_pitches[0]],
                                [100, mean_pitches[1]],
                                [100 + TRIANGLE_WIDTH / 2, mean_pitches[0]]])

        # make the triangle take up the entire plot (plus a little margin)
        annotation_axis.set_xlim([100 - TRIANGLE_WIDTH / 2 - 5, 100 + TRIANGLE_WIDTH / 2 + 5])

        annotation_axis.fill(coordinates[:, 0], coordinates[:, 1],
                             color="black",
                             alpha=0.5,
                             edgecolor="black")


class GroupPitchNormal(Figure):
    def __init__(self, experiment: Experiment, plot_order: list = None,
                 colormap: dict = None, render: bool = True):

        # override default colormap
        if colormap is None:
            colormap = {
                "AD Patients": {
                    "InitialPitch": ColorUtils.alter_brightness("goldenrod", -0.5),
                    "EndingPitch": "goldenrod"
                },
                "Controls": {
                    "InitialPitch": ColorUtils.alter_brightness("teal", -0.7),
                    "EndingPitch": "teal"
                }
            }

        self.hatch_map = {
            "InitialPitch": " ",
            "EndingPitch": " "
        }

        Figure.__init__(self, colormap=colormap, subplots=(len(experiment.subjects), 1), shared_axis=("all", "none"))

        self.experiment = experiment
        self.name = "group_pitch_normal"

        self.plot_order = [group for group in experiment.subjects] if plot_order is None else plot_order

        if render:
            self.render()

    def render(self):
        time_window_labels = {"InitialPitch": "Initial Pitch", "EndingPitch": "Mid-trial Pitch"}

        # set figure size manually, just for this case
        self.figure.tight_layout()
        self.figure.subplots_adjust(top=0.90, bottom=0.1)

        limits = (min(self.experiment.df["InitialPitch"].min(), self.experiment.df["EndingPitch"].min()),
                  max(self.experiment.df["InitialPitch"].max(), self.experiment.df["EndingPitch"].max()))

        for group_idx, group in enumerate(self.plot_order):
            # place each group on its own subplot
            axis = self.axes[group_idx]
            group_data = self.experiment.df[self.experiment.df["Group"] == group]

            axis.set_ylabel(group)
            axis.set_yticks([])
            axis.set_xlim(limits)

            for index, label in time_window_labels.items():
                mean, std = (np.mean(group_data[index]), np.std(group_data[index]))
                # density ensures that rescaling the figure's x-axis won't ruin our smooth-ness of our plot
                axis.fill(np.linspace(limits[0], limits[1], num=round((limits[1] - limits[0]) /
                                                                      defaults["normal-dist-density"])),
                          norm.pdf(np.linspace(limits[0], limits[1],
                                               num=round((limits[1] - limits[0])
                                                         / defaults["normal-dist-density"])),
                                   mean, std),
                          color=self.colormap[group][index], alpha=0.75, label=label,
                          edgecolor=defaults["hist-outline-color"],
                          hatch=self.hatch_map[index])

            axis.legend(loc="upper left")

        # only label the bottom-most value
        self.axes[-1].set_xlabel("Pitch (Cents)")


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

    group_pitch_histogram_figure.name = "group_pitch_histogram"
    return group_pitch_histogram_figure


# global settings
__global_styles()
