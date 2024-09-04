import colorsys
from pprint import pprint

import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import matplotlib.style as style
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
        axes.text(np.mean(x), max(y) + rise + defaults["annotation-vertical-padding"], label,
                  horizontalalignment="center",
                  size=defaults["annotation-text-size"], )

    @staticmethod
    def plot_bar(axes: plt.axes, start, end, dashed_end, color="black", line_width=2):
        # first, draw the bar itself
        axes.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=line_width)
        axes.plot([end[0], dashed_end], [end[1], end[1]], color=color, linestyle="dotted")


# def __default_group_colormap():
#     return {"AD Patients": "darkgoldenrod", "Controls": "teal"}


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
    plt.rc("font", **{
        'family': 'Times',
        'size': 15
    })


class Figure:
    def __init__(self, colormap: dict = None, fig_size=(6.4, 4.8), subplots=None, shared_axis=("none", "none"),
                 **kwargs):
        if colormap is None:
            self.colormap = {"AD Patients": "indianred",
                             "Controls": "cadetblue"}
        else:
            self.colormap = colormap

        if subplots is not None:
            self.figure, self.axes = plt.subplots(nrows=subplots[0], ncols=subplots[1],
                                                  sharex=shared_axis[0], sharey=shared_axis[1],
                                                  figsize=fig_size,
                                                  **kwargs)
            for axis in self.axes:
                axis.spines[['right', 'top']].set_visible(False)
                axis.tick_params(axis='both', direction='out')
                axis.get_xaxis().tick_bottom()  # remove unneeded ticks
                axis.get_yaxis().tick_left()
        else:
            self.figure = plt.figure(figsize=fig_size, **kwargs)
            self.axes = self.figure.axes
            for axis in self.axes:
                axis.spines[['right', 'top']].set_visible(False)
                axis.tick_params(axis='both', direction='out')
                axis.get_xaxis().tick_bottom()  # remove unneeded ticks
                axis.get_yaxis().tick_left()

        self.name = None
        # increase DPI
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300

    # All plotting functions must have a render function
    def render(self):
        pass

    def save(self, output_directory: str):
        self.figure.savefig(f"{output_directory}/{self.name}.png")


class SubFigure:
    def __init__(self, colormap: dict = None):
        if colormap is None:
            self.colormap = {"AD Patients": "indianred",
                             "Controls": "cadetblue"}
        else:
            self.colormap = colormap


class Distributions(Figure):
    def __init__(self, experiments: dict, render: bool = True, **kwargs):
        Figure.__init__(self)

        self.name = "distributions_figure"
        self.sub_figures = []
        self.kwargs = kwargs

        self.experiments = experiments

        if render:
            self.render()

    def render(self):
        GroupInitialPitchesDistribution = self.figure.add_subplot(1, 1, 1)
        self.sub_figures.append(GroupCenteringTercileDots(experiment=self.experiments["peripheral"],
                                                          axes=GroupInitialPitchesDistribution, **self.kwargs))

        self.figure: plt.Figure
        self.figure.tight_layout()
        self.figure.subplots_adjust(hspace=0, wspace=0.3)

        for figure in self.sub_figures:
            figure.axes.spines[['right', 'top']].set_visible(False)
            figure.axes.tick_params(axis='both', direction='out')
            figure.axes.get_xaxis().tick_bottom()  # remove unneeded ticks
            figure.axes.get_yaxis().tick_left()


class GroupCentralPeripheralBars(SubFigure):
    def __init__(self, experiment: Experiment, axes: plt.Axes, plot_order: list = None, colormap: dict = None):
        SubFigure.__init__(self, colormap=colormap)
        self.axes = axes
        self.experiment = experiment

        self.bar_y = None
        self.bar_x = None

        self.label_alias = {
            "CENTRAL": "Central",
            "PERI": "Peripheral",
        }

        self.plot_order = [group for group in experiment.subjects] if plot_order is None else plot_order
        self.render()

    def render(self):
        # (label, height, error, color, fill-style)
        bars = []

        for group in self.plot_order:
            group_data = self.experiment.df[self.experiment.df["Group"] == group]

            central_tercile_data = group_data[group_data["Tercile"] == "CENTRAL"]
            bars.append({
                "group": group,
                "label": "CENTRAL",
                "height": np.nanmean(central_tercile_data["Centering"]),
                "error": StatisticsHelpers.standard_error(central_tercile_data["Centering"]),
                "color": self.colormap[group],
            })

        for group in self.plot_order:
            group_data = self.experiment.df[self.experiment.df["Group"] == group]

            peripheral_tercile_data = group_data[group_data["Tercile"] != "CENTRAL"]
            bars.append({
                "group": group,
                "label": "PERI",
                "height": np.nanmean(peripheral_tercile_data["Centering"]),
                "error": StatisticsHelpers.standard_error(peripheral_tercile_data["Centering"]),
                "color": self.colormap[group],
            })

        print("Group Central vs Peripheral Data")
        pprint(bars)

        self.bar_x = np.arange(len(bars), dtype=np.float64)
        self.bar_y = [bar["height"] + bar["error"] for bar in bars]

        # insert spaces between groups
        # for i in range(len(bars)):
        #     if i > 0 and bars[i - 1]["group"] != bars[i]["group"]:
        #         self.bar_x[i:] += defaults["bar-group-spacing"]

        self.axes.errorbar(x=self.bar_x,
                           y=[bar["height"] for bar in bars],
                           yerr=[bar["error"] for bar in bars],
                           color=defaults["error-color"],
                           elinewidth=defaults["line-width"],
                           capthick=defaults["line-width"],
                           capsize=defaults["error-cap-size"],
                           ls=defaults["error-line-style"])

        self.axes.bar(x=self.bar_x,
                      width=0.9,
                      height=[bar["height"] for bar in bars],
                      color=[bar["color"] for bar in bars],
                      alpha=0.9)

        self.axes.axhline(y=0, color='black')

        self.axes.set_xticks(ticks=[0.5, 2.5])
        self.axes.set_xticklabels([alias for _, alias in self.label_alias.items()], rotation=0, ha="center")
        self.axes.tick_params(axis='x', length=0)

        self.axes.set_ylabel("Centering (Cents)")

        self.axes.legend(handles=[patches.Patch(color=self.colormap[group], label=default_group_name_map()[group])
                                  for group in self.plot_order],
                         loc="upper left",
                         frameon=False,
                         prop={'size': 15})

    def annotate_significance(self, x, label):
        Annotations.bar_significance(axes=self.axes,
                                     x=[self.bar_x[i] for i in x],
                                     y=[self.bar_y[i] + defaults["annotation-vertical-padding"] for i in x],
                                     rise=defaults["bar-significance-rise"],
                                     label=label)


class GroupCenteringTercileDots(SubFigure):
    def __init__(self, experiment: Experiment, axes: plt.Axes, plot_order: list = None,
                 colormap: dict = None):
        SubFigure.__init__(self, colormap=colormap)
        self.axes = axes

        self.bar_x = None
        self.bar_y = None
        self.experiment = experiment

        self.label_map = {
            "UPPER": "Upper\nPeripheral",
            "LOWER": "Lower\nPeripheral"
        }

        self.plot_order = [group for group in experiment.subjects] if plot_order is None else plot_order
        self.render()

    def render(self):
        # (label, height, error, color, fill-style)
        dots = []

        for group in self.plot_order:
            group_data = self.experiment.df[self.experiment.df["Group"] == group]
            for tercile_index, tercile in enumerate(group_data["Tercile"].drop_duplicates()):
                tercile_data = group_data[group_data["Tercile"] == tercile]
                dots.append({
                    "group": group,
                    "label": tercile,
                    "height": np.nanmean(np.abs(tercile_data["InitialPitch"])),
                    "error": StatisticsHelpers.standard_error(np.abs(tercile_data["InitialPitch"])),
                    "color": self.colormap[group],
                })

        print("Group Tercile Initial Deviation Dots:")
        pprint(dots)

        self.bar_x = np.arange(len(dots), dtype=np.float64)
        self.bar_y = [bar["height"] + bar["error"] for bar in dots]

        # insert spaces between groups
        for i in range(len(dots)):
            if i > 0 and dots[i - 1]["group"] != dots[i]["group"]:
                self.bar_x[i:] += defaults["bar-group-spacing"]

        self.axes.errorbar(x=self.bar_x,
                           y=[bar["height"] for bar in dots],
                           yerr=[bar["error"] for bar in dots],
                           color=defaults["error-color"],
                           elinewidth=defaults["line-width"],
                           capthick=defaults["line-width"],
                           capsize=defaults["error-cap-size"],
                           ls=defaults["error-line-style"],
                           zorder=1)

        self.axes.scatter(x=self.bar_x,
                          y=[bar["height"] for bar in dots],
                          color=[bar["color"] for bar in dots],
                          edgecolors=defaults["error-color"],
                          alpha=1.0,
                          marker='h',
                          s=300,
                          zorder=2)

        self.axes.set_xticks(ticks=self.bar_x)
        self.axes.set_xticklabels([self.label_map[bar["label"]] for bar in dots], rotation=0, ha="center")
        self.axes.tick_params(axis='x', length=0)

        self.axes.set_ylabel("Initial Pitch Deviation Magnitude (Cents)")

        self.axes.legend(handles=[patches.Patch(color=self.colormap[group], label=default_group_name_map()[group])
                                  for group in self.plot_order],
                         loc="upper left",
                         frameon=False,
                         prop={'size': 15})

    def annotate_significance(self, x, label):
        Annotations.bar_significance(axes=self.axes,
                                     x=[self.bar_x[i] for i in x],
                                     y=[self.bar_y[i] + defaults["annotation-vertical-padding"] for i in x],
                                     rise=defaults["bar-significance-rise"],
                                     label=label)


class Results(Figure):
    def __init__(self, experiments: dict, render: bool = True, plot_flags=((False, False), (False, False), (), ()), **kwargs):
        Figure.__init__(self, fig_size=(9.6, 9))
        self.name = "results_figure"
        self.sub_figures = []
        self.kwargs = kwargs

        self.experiments = experiments

        self.plot_flags = plot_flags

        if render:
            self.render()

    def render(self):
        GroupTercileArrowsAxes = [self.figure.add_subplot(2, 2, 1), self.figure.add_subplot(2, 2, 2)]
        self.sub_figures.append(GroupTercileArrows(experiment=self.experiments["trimmed"],
                                                   axes=GroupTercileArrowsAxes, plot_flags=self.plot_flags[0],
                                                   **self.kwargs))
        GroupPitchNormalAxes = [self.figure.add_subplot(4, 2, 5),
                                self.figure.add_subplot(4, 2, 7)]

        # combine x-axes
        GroupPitchNormalAxes[0].get_shared_x_axes().join(GroupPitchNormalAxes[0], GroupPitchNormalAxes[1])

        self.sub_figures.append(GroupPitchNormal(experiment=self.experiments["raw"],
        # self.sub_figures.append(GroupPitchNormal(experiment=self.experiments["peripheral"],
                                                 axes=GroupPitchNormalAxes, **self.kwargs))

        GroupTercileCenteringBarsAxes = self.figure.add_subplot(4, 2, (6, 8))
        self.sub_figures.append(GroupTercileCenteringBars(experiment=self.experiments["peripheral"],
                                                          axes=GroupTercileCenteringBarsAxes, **self.kwargs))

        for figure in self.sub_figures:
            if isinstance(figure.axes, list):
                for axis in figure.axes:
                    axis.spines[['right', 'top']].set_visible(False)
                    axis.tick_params(axis='both', direction='out')
                    axis.get_xaxis().tick_bottom()  # remove unneeded ticks
                    axis.get_yaxis().tick_left()
            else:
                figure.axes.spines[['right', 'top']].set_visible(False)
                axis = figure.axes
                axis.tick_params(axis='both', direction='out')
                axis.get_xaxis().tick_bottom()  # remove unneeded ticks
                axis.get_yaxis().tick_left()

        self.figure: plt.Figure
        self.figure.tight_layout()
        self.figure.subplots_adjust(hspace=0.4, wspace=0.25)

        self.figure.text(0.01, 0.95, "A", fontsize="xx-large", fontweight="bold")
        self.figure.text(0.51, 0.95, "B", fontsize="xx-large", fontweight="bold")
        self.figure.text(0.01, 0.47, "C", fontsize="xx-large", fontweight="bold")
        self.figure.text(0.51, 0.47, "D", fontsize="xx-large", fontweight="bold")


class SubFigure:
    def __init__(self, colormap: dict = None, **kwargs):
        if colormap is None:
            self.colormap = {"AD Patients": "indianred",
                             "Controls": "cadetblue"}
        else:
            self.colormap = colormap


class GroupTercileCenteringBars(SubFigure):
    def __init__(self, experiment: Experiment, axes: plt.Axes, plot_order: list = None,
                 colormap: dict = None):
        SubFigure.__init__(self, colormap=colormap)

        # override colormap
        self.colormap = {
            "Controls": {
                "LOWER": "lightcoral",
                "UPPER": "brown"
            },
            "AD Patients": {
                "LOWER": "mediumturquoise",
                "UPPER": "darkcyan"
            }
        }

        self.axes = axes

        self.bar_x = None
        self.bar_y = None
        self.experiment = experiment

        self.label_map = {
            "UPPER": "Lowering",
            "LOWER": "Raising"
        }

        self.plot_order = [group for group in experiment.subjects] if plot_order is None else plot_order
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
                    "color": self.colormap[group][tercile],
                })

        print("Group Tercile Centering Bar Data:")
        pprint(bars)

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
                      # hatch=["/", " ", "/", ""]
                      )

        self.axes.set_xticks(ticks=self.bar_x)
        self.axes.set_xticklabels([self.label_map[bar["label"]] for bar in bars], rotation=0, ha="center")
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

        legend_dict = dict(zip(self.plot_order, [list(self.colormap[group].values()) for group in self.colormap]))
        patches = []
        for cat, col in legend_dict.items():
            patches.append([mpatches.Patch(facecolor=c, label=cat) for c in col])
        self.axes.legend(handles=patches,
                         labels=self.plot_order,
                         frameon=False,
                         loc="upper left",
                         handler_map={list: HandlerTuple(None)},
                         prop={'size': 15}
        )

    def annotate_significance(self, x, label):
        Annotations.bar_significance(axes=self.axes,
                                     x=[self.bar_x[i] for i in x],
                                     y=[self.bar_y[i] + defaults["annotation-vertical-padding"] for i in x],
                                     rise=defaults["bar-significance-rise"],
                                     label=label)


def default_group_name_map():
    return {"AD Patients": "Patients with AD", "Controls": "Controls"}


class GroupTercileArrows(SubFigure):
    def __init__(self, axes: list, experiment: Experiment, plot_order: list = None,
                 colormap: dict = None,
                 plot_flags=(False, False)):
        SubFigure.__init__(self, colormap=colormap)

        # override colormap
        self.colormap = {
            "Controls": {
                "LOWER": "lightcoral",
                "CENTRAL": "indianred",
                "UPPER": "brown"
            },
            "AD Patients": {
                "LOWER": "mediumturquoise",
                "CENTRAL": "cadetblue",
                "UPPER": "darkcyan"
            }
        }

        self.axes = axes

        self.experiment = experiment
        self.plot_order = [group for group in experiment.subjects] if plot_order is None else plot_order

        self.plot_flags = plot_flags

        self.opacity_map = {
            "UPPER": 0.75,
            "CENTRAL": 0.25,
            "LOWER": 0.75
        }

        self.error_colormap = {
            "AD Patients": {
                "InitialPitch": ColorUtils.alter_brightness("cadetblue", -0.7),
                "EndingPitch": "cadetblue"
            },
            "Controls": {
                "InitialPitch": ColorUtils.alter_brightness("lightsalmon", -0.7),
                "EndingPitch": "lightsalmon"
            }
        }

        self.render()

    def render(self):
        # set figure size manually, just for this case
        for group_idx, group in enumerate(self.plot_order):
            axis = self.axes[group_idx]
            group_data = self.experiment.df[self.experiment.df["Group"] == group]
            subjects = self.experiment.subjects[group]
            for subject_idx, subject in enumerate(subjects):
                subject_data = group_data[group_data["Subject"] == subject]

                for tercile in subject_data["Tercile"].drop_duplicates():
                    tercile_data = subject_data[subject_data["Tercile"] == tercile]

                    stderr = lambda data: np.std(data, ddof=1)/np.sqrt(np.size(data))

                    base = np.mean(tercile_data["InitialPitch"])
                    base_error = stderr(tercile_data["InitialPitch"])

                    # Plot pitch movement
                    if self.plot_flags[0]:
                        apex = np.mean(tercile_data["EndingPitch"])
                        apex_error = stderr(tercile_data["EndingPitch"])
                    # Plot centering
                    else:
                        apex = base + np.mean(tercile_data["Centering"]) * (-1 if base > 0 else 1)
                        apex_error = stderr(base + tercile_data["EndingPitch"] * (-1 if base > 0 else 1))

                    coordinates = np.array([[base, subject_idx + 1 - 0.25],
                                            [base, subject_idx + 1 + 0.25],
                                            [apex, subject_idx + 1]])

                    # draw a filled triangle with these coordinates
                    axis.fill(coordinates[:, 1], coordinates[:, 0],
                              color=self.colormap[group][tercile],
                              alpha=self.opacity_map[tercile],
                              edgecolor="black")

                    if self.plot_flags[1] and tercile in ["UPPER", "LOWER"]:
                        # plot error bar for base and apex
                        axis.plot([subject_idx + 1]*2, [base - base_error, base + base_error], color=self.error_colormap[group]["InitialPitch"], alpha=0.7, linewidth=1.5)
                        axis.plot([subject_idx + 1]*2, [apex - apex_error, apex + apex_error], color=self.error_colormap[group]["EndingPitch"], alpha=0.7, linewidth=1.5)

            group_name = default_group_name_map()[group]
            axis.set_title(f"{group_name} {' (pitch movement)' if self.plot_flags[0] else ''}")
            axis.set_xlabel("Subject Number")
            axis.set_xlim(0, len(subjects) + 1)
            axis.set_xticks([float(i) for i in range(1, len(subjects) + 1, 2)])
            axis.set_xticks([float(i) for i in range(2, len(subjects) + 1, 2)], minor=True)
            axis.set_xticklabels([str(i) for i in range(1, len(subjects) + 1, 2)])
            axis.set_ylabel("Pitch (Cents)")

        y_axes = [limit(limit(ax.get_ylim()) for ax in self.axes) for limit in [min, max]]
        for axis in self.axes:
            axis.set_ylim(y_axes)

        for axis_index in range(1, len(self.axes)):
            # disable tick marks for all except the first axis
            self.axes[axis_index].set_yticklabels([])
            self.axes[axis_index].set_ylabel("")


class CenteringMethods(Figure):
    @staticmethod
    def generate_track(motion_points):
        smooth_motion_function = np.poly1d(np.polyfit(motion_points[0], motion_points[1], 3))
        plot_domain = np.linspace(0, 200, 150)
        smooth_motion_points = smooth_motion_function(plot_domain)
        noisy_motion_points = smooth_motion_points + np.random.normal(smooth_motion_points, 1)

        return noisy_motion_points

    def __init__(self, motion_points: list, render: bool = True):
        Figure.__init__(self, fig_size=(14.08, 5.28))
        self.axes = [self.figure.add_subplot(1, 10, (1, 5)),
                     self.figure.add_subplot(1, 10, (6, 7)),
                     self.figure.add_subplot(1, 10, (8, 10))]

        # reset the axes manually
        for axis in self.axes:
            axis.spines[['right', 'top']].set_visible(False)
            axis.tick_params(axis='both', direction='out')
            axis.get_xaxis().tick_bottom()  # remove unneeded ticks
            axis.get_yaxis().tick_left()

        self.name = "sample_trial"
        self.tracks = []

        for mp in motion_points:
            self.tracks.append(CenteringMethods.generate_track(mp))

        if render:
            self.render()

        self.figure.text(0.01, 0.93, "A", fontsize="xx-large", fontweight="bold")
        self.figure.text(5/10, 0.93, "B", fontsize="xx-large", fontweight="bold")
        self.figure.text(7/10, 0.93, "C", fontsize="xx-large", fontweight="bold")

    def render(self):
        TARGET = 0
        self.figure.tight_layout()
        self.figure.subplots_adjust(left=0.06, wspace=2)

        axis = self.axes[0]
        axis.set_ylabel("Pitch (hz)")
        axis.set_xlabel("Time (ms)")

        domain = np.linspace(0, 200, len(self.tracks[0]), endpoint=True)

        axis.plot(domain, self.tracks[TARGET], color="blue", linewidth=1, alpha=0.5)

        # add the "center" line
        axis.plot(domain, np.zeros(domain.shape), color="black")

        # compute y limits so all values are shown well
        ylim = [np.min(self.tracks, axis=None) * 0.99 - 12, np.max(self.tracks, axis=None) * 1.01]
        axis.set_ylim(ylim)

        # add rectangular overlay for windows
        axis.add_patch(patches.Rectangle((0, ylim[0]), 50, ylim[1] - ylim[0], color="gray", alpha=0.1))
        axis.add_patch(patches.Rectangle((150, ylim[0]), 50, ylim[1] - ylim[0], color="gray", alpha=0.1))

        # add text labels

        axis.text(25, ylim[0] + 2, "Initial", horizontalalignment="center", fontsize="medium")
        axis.text(175, ylim[0] + 2, "Mid-trial", horizontalalignment="center", fontsize="medium")

        # compute means and medians
        windows = (domain <= 50, 150 <= domain)
        window_means = [[np.mean(track[window]) for window in windows] for track in self.tracks]

        center_hz = [np.median([track[i] for track in window_means])
                     for i, _ in enumerate(windows)]

        for i, window in enumerate(windows):
            axis.plot(domain[window], len(domain[window]) * [window_means[TARGET][i]], color="blue",
                      linewidth=2)

            # compute the median production
            axis.plot(domain[window], len(domain[window]) * [center_hz[i]],
                      color="green",
                      linewidth=2)

        sample_trial_patch = mpatches.Patch(color='blue', label='Sample Trial')
        window_median = mpatches.Patch(color='green', label='Median')

        axis.legend(handles=[sample_trial_patch, window_median],
                    loc="upper right", frameon=False,
                    prop={'size': 15})

        # cents conversion
        axis = self.axes[1]
        axis.set_xlim([0, 125])
        axis.set_ylabel("Pitch Deviation (cents)")
        # axis.set_xlabel("Window")
        axis.set_xticks([25, 100])
        axis.set_xticklabels(["Initial", "Mid-trial"])
        axis.tick_params(axis='x', length=0)

        target_cents = 1200 * (np.log(window_means[TARGET]) - np.log(center_hz)) / np.log(2)

        # rescale the windows
        windows = (domain <= 50, np.logical_and(domain >= 75, domain <= 125))

        axis.plot(domain, [0] * len(domain), color="black", zorder=1)

        for i, window in enumerate(windows):
            axis.plot(domain[window], len(domain[window]) * [target_cents[i]], color="blue",
                      linewidth=2)

            if target_cents[i] >= 0:
                axis.add_patch(patches.Rectangle((min(domain[window]), 0),
                                                 50, target_cents[i],
                                                 color="blue", alpha=0.1))
            else:
                axis.add_patch(patches.Rectangle((min(domain[window]), target_cents[i]),
                                                 50, abs(target_cents[i]),
                                                 color="blue", alpha=0.1))

            axis.plot(domain[window], len(domain[window]) * [0],
                      color="green",
                      linewidth=2)

        axis = self.axes[2]
        axis.set_xlim([0, 175])
        axis.set_ylabel("Magnitude of Pitch Deviation (cents)")
        # axis.set_xlabel("Window")

        axis.set_xticks([25, 100])
        axis.set_xticklabels(["Initial", "Mid-trial"])
        axis.tick_params(axis='x', length=0)

        target_cents = np.abs(1200 * (np.log(window_means[TARGET]) - np.log(center_hz)) / np.log(2))

        # rescale the windows
        windows = (domain <= 50, np.logical_and(domain >= 75, domain <= 125))

        for i, window in enumerate(windows):
            axis.plot(domain[window], len(domain[window]) * [target_cents[i]], color="blue",
                      linewidth=2)
            axis.plot(domain[window], len(domain[window]) * [0],
                      color="green",
                      linewidth=2)

            if target_cents[i] >= 0:
                axis.add_patch(patches.Rectangle((min(domain[window]), 0),
                                                 50, target_cents[i],
                                                 color="blue", alpha=0.1))
            else:
                axis.add_patch(patches.Rectangle((min(domain[window]), target_cents[i]),
                                                 50, abs(target_cents[i]),
                                                 color="blue", alpha=0.1), zorder=10)

        LABEL_OFFSET = 135
        arrowhead_size = 1.5

        # add centering text label

        # plot the triangle
        TRIANGLE_WIDTH = 30
        TRIANGLE_OFFSET = 150
        coordinates = np.array([[TRIANGLE_OFFSET - TRIANGLE_WIDTH / 2, target_cents[0]],
                                [TRIANGLE_OFFSET, target_cents[1]],
                                [TRIANGLE_OFFSET + TRIANGLE_WIDTH / 2, target_cents[0]]])
        axis.fill(coordinates[:, 0], coordinates[:, 1],
                  color="gray",
                  alpha=.7,
                  edgecolor="black",
                  zorder=150)

        axis.text(TRIANGLE_OFFSET,
                  (max(target_cents) - len("centering")/2 - 3),
                  "centering",
                  horizontalalignment="center",
                  verticalalignment="center",
                  size=defaults["annotation-text-size"],
                  rotation='vertical',
                  color="black",
                  zorder=200)

        axis.plot([50, TRIANGLE_OFFSET - (TRIANGLE_WIDTH / 2)], 2 * [target_cents[0]],
                  color="black", linestyle="dotted")
        axis.plot([125, TRIANGLE_OFFSET], 2 * [target_cents[1]],
                  color="black", linestyle="dotted")


class GroupPitchNormal(SubFigure):
    def __init__(self, axes: list, experiment: Experiment, plot_order: list = None):
        # do not supply a colormap
        SubFigure.__init__(self)
        self.axes = axes

        # override default colormap
        self.colormap = {
            "AD Patients": {
                "InitialPitch": ColorUtils.alter_brightness("lightblue", -0.5),
                "EndingPitch": "lightblue"
            },
            "Controls": {
                "InitialPitch": ColorUtils.alter_brightness("lightsalmon", -0.7),
                "EndingPitch": "lightsalmon"
            }
        }

        self.hatch_map = {
            "InitialPitch": " ",
            "EndingPitch": " "
        }

        self.experiment = experiment
        self.name = "group_pitch_normal"

        self.plot_order = [group for group in experiment.subjects] if plot_order is None else plot_order

        self.render()

    def render(self):
        time_window_labels = {"InitialPitch": "$F_{\\mathrm{init}}$", "EndingPitch": "$F_{\\mathrm{mid}}$"}

        limits = (min(self.experiment.df["InitialPitch"].min(), self.experiment.df["EndingPitch"].min()),
                  max(self.experiment.df["InitialPitch"].max(), self.experiment.df["EndingPitch"].max()))

        for group_idx, group in enumerate(self.plot_order):
            # place each group on its own subplot
            axis = self.axes[group_idx]
            group_data = self.experiment.df[self.experiment.df["Group"] == group]

            axis: plt.Axes
            axis.set_title(default_group_name_map()[group])
            axis.set_ylabel("Probability")
            axis.set_yticks([])
            axis.set_xlim(*limits)
            axis.tick_params(axis="x", direction="out")

            binsize = 25

            for index, label in time_window_labels.items():
                bincount = int(np.ceil((max(group_data[index]) - min(group_data[index])) / binsize))
                histogram, bins = np.histogram(group_data[index], bins=bincount)

                # normalization
                histogram = histogram / sum(histogram)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                # increase resolution of bin centers
                points = 1000
                x_points = np.linspace(min(bin_centers), max(bin_centers), points)
                histogram = np.interp(x_points, bin_centers, histogram)
                bin_centers = np.interp(x_points, bin_centers, bin_centers)

                parzen_kernel = lambda x, h: np.exp(-(x ** 2) / (2 * h ** 2)) / (np.sqrt(2 * np.pi) * h)

                # parzen kernel
                h = 0.05
                kernel_values = parzen_kernel(np.linspace(-1, 1, points), h)

                histogram = np.convolve(kernel_values / sum(kernel_values), histogram)

                axis.fill(bin_centers, histogram[int(len(bin_centers) / 2): int(len(bin_centers) * 3 / 2)],
                          color=self.colormap[group][index],
                          alpha=0.75,
                          label=label,
                          edgecolor=defaults["hist-outline-color"],
                          hatch=self.hatch_map[index]
                          )
            axis.legend(loc="upper left", frameon=False,
                        prop={'size': 15})

            limits = axis.get_ylim()
            axis.set_ylim(bottom=limits[0], top=limits[1] * 1.1)

        # only label the bottom-most value
        self.axes[-1].set_xlabel("Pitch Deviation (Cents)")

        for axis_index in range(len(self.axes) - 1):
            # disable tick marks for all except the last normal dist
            self.axes[axis_index].set_xticklabels([])


class Discussion(Figure):
    def __init__(self, render: bool = True, **kwargs):
        Figure.__init__(self)

        self.name = "discussion_figure"
        self.sub_figures = []
        self.kwargs = kwargs

        self.figure.tight_layout()
        self.figure.subplots_adjust(hspace=0, wspace=0.3)

        self.axes = [self.figure.add_subplot(1, 1, 1)]
                     # self.figure.add_subplot(2, 1, 2)]

        if render:
            self.render()
        # self.figure.text(0.03, 0.9, "A", fontsize="xx-large", fontweight="bold")
        # self.figure.text(0.03, 0.45, "B", fontsize="xx-large", fontweight="bold")

    def render(self):
        RecruitmentDiscussion(self.axes[0]).render()
        # SensitivityDiscussion(self.axes[1]).render()


# class SensitivityDiscussion(SubFigure):
#     def __init__(self, axis, **kwargs):
#         super().__init__(**kwargs)
#         self.axis = axis
#
#     def render(self):
#         def sigmoid(arr, scale=1., offset=0):
#             arr = np.asarray(arr)
#             result = 1 / (1 + np.exp(-(arr - offset) * scale))
#             return result
#
#         SchematicAxes = self.axis
#
#         SchematicAxes.set_xlim([-200, 200])
#         SchematicAxes.set_ylim([0, 1])
#         SchematicAxes.set_yticklabels([])
#         SchematicAxes.set_yticks([])
#
#         domains = [np.linspace(-200, 0, 400), np.linspace(0, 200, 400)]
#         SchematicAxes.plot(domains[0] + 10, sigmoid(-domains[0], scale=0.05, offset=100), color="black",
#                            linewidth=3,
#                            label="")
#         SchematicAxes.plot(domains[1] + 10, sigmoid(domains[1], scale=0.05, offset=100), color="indianred",
#                            linewidth=3,
#                            label="SD")
#
#         SchematicAxes.set_ylabel("Feedback Sensitivity")
#
#         x_ticks = np.arange(-200, 201, 100)
#         y_ticks = np.arange(0, 1, 0.25)
#         SchematicAxes.grid(color='grey', linestyle='--', zorder=0)
#         SchematicAxes.set_xticks(x_ticks), SchematicAxes.set_yticks(y_ticks)
#
#         SchematicAxes.set_xlabel("Pitch Deviation from Median")


class RecruitmentDiscussion(SubFigure):
    def __init__(self, axis):
        self.axis = axis

    def render(self):
        def sigmoid(arr, scale=1., offset=0):
            arr = np.asarray(arr)
            result = 1 / (1 + np.exp(-(arr - offset) * scale))
            return result

        SchematicAxes = self.axis

        SchematicAxes.set_xlim([0, 200])
        SchematicAxes.set_ylim([0, 1])
        SchematicAxes.set_yticklabels([])

        domain = np.linspace(0, 200, 400)

        SchematicAxes.plot(domain, sigmoid(domain, scale=0.05, offset=130), color="gray",
                           linewidth=3,
                           label="Raising")
        SchematicAxes.plot(domain, sigmoid(domain, scale=0.05, offset=90), color="darkgray",
                           linewidth=3,
                           label="Lowering")

        SchematicAxes.set_ylabel("Effort Required for Pitch Change")
        SchematicAxes.set_xlabel("Initial Pitch Deviation from Median")

        x_ticks = np.arange(0, 201, 50)
        y_ticks = np.arange(0, 1, 0.25)
        SchematicAxes.grid(color='grey', linestyle='--', zorder=0)
        SchematicAxes.set_xticks(x_ticks), SchematicAxes.set_yticks(y_ticks)

        SchematicAxes.legend(loc='upper left', frameon=False)


class Extra(Figure):
    def __init__(self, experiment: Experiment, plot_order: list = [], render: bool = True, **kwargs):
        Figure.__init__(self)

        self.name = "extra_figure"
        self.experiment = experiment
        self.sub_figures = []
        self.kwargs = kwargs
        self.axes = [self.figure.add_subplot(1, 1, 1)]
        self.plot_order = plot_order

        if render:
            self.render()

    def render(self):
        PitchMovement(self.experiment, self.axes[0], self.plot_order).render()


class PitchMovement(SubFigure):
    def __init__(self, experiment: Experiment, axes: plt.Axes, plot_order: list = [],
                 colormap: dict = {}):
        SubFigure.__init__(self, colormap=colormap)
        self.axes = axes

        self.bar_x = None
        self.bar_y = None
        self.experiment = experiment

        self.label_map = {
            "UPPER": "Upper\nTercile",
            "LOWER": "Lower\nTercile"
        }

        self.colormap = {
            "Controls": {
                "LOWER": "lightcoral",
                "UPPER": "brown"
            },
            "AD Patients": {
                "LOWER": "mediumturquoise",
                "UPPER": "darkcyan"
            }
        }

        print(f"Plot order {plot_order}")
        self.plot_order = [group for group in experiment.subjects] if plot_order is None else plot_order
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
                    "height": np.nanmean(tercile_data["NormPitchMovement"]),
                    "error": StatisticsHelpers.standard_error(tercile_data["NormPitchMovement"]),
                    "color": self.colormap[group][tercile],
                })

        print("Group Tercile Pitch Movement Bar Data:")
        pprint(bars)

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
                      )

        self.axes.set_xticks(ticks=self.bar_x)
        self.axes.set_xticklabels([self.label_map[bar["label"]] for bar in bars], rotation=0, ha="center")
        self.axes.tick_params(axis='x', length=0)

        self.axes.set_ylabel("Pitch Movement (Cents)")

        self.axes.errorbar(x=self.bar_x,
                           y=[bar["height"] for bar in bars],
                           yerr=[bar["error"] for bar in bars],
                           color=defaults["error-color"],
                           elinewidth=defaults["line-width"],
                           capthick=defaults["line-width"],
                           capsize=defaults["error-cap-size"],
                           ls=defaults["error-line-style"])

        legend_dict = dict(zip(self.plot_order, [list(self.colormap[group].values()) for group in self.colormap]))
        patches = []
        for cat, col in legend_dict.items():
            patches.append([mpatches.Patch(facecolor=c, label=cat) for c in col])
        self.axes.legend(handles=patches,
                         labels=self.plot_order,
                         frameon=False,
                         loc="upper left",
                         handler_map={list: HandlerTuple(None)},
                         prop={'size': 15}
                         )


# global styles
__global_styles()
