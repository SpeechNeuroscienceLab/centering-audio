import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.patches as patches
import matplotlib.colors as colors
import colorsys
import numpy as np
from scipy.stats import norm
from experiment import Experiment
from pprint import pprint


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
    plt.rc("font", **{
        'family': 'Times',
        'size': 15
    })


class Figure:
    def __init__(self, colormap: dict = None, fig_size=(6.4, 4.8), subplots=None, shared_axis=("none", "none"),
                 **kwargs):
        if colormap is None:
            self.colormap = {"AD Patients": "darkgoldenrod", "Controls": "teal"}
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
    def __init__(self, colormap: dict = None, **kwargs):
        if colormap is None:
            self.colormap = {"AD Patients": "darkgoldenrod", "Controls": "teal"}
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
                          alpha=0.9,
                          marker='h',
                          s=300,
                          zorder=2)

        self.axes.set_xticks(ticks=self.bar_x)
        self.axes.set_xticklabels([self.label_map[bar["label"]] for bar in dots], rotation=0, ha="center")
        self.axes.tick_params(axis='x', length=0)

        self.axes.set_ylabel("Magnitude of Initial Pitch (Cents)")

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
    def __init__(self, experiments: dict, render: bool = True, **kwargs):
        Figure.__init__(self, fig_size=(9.6, 9))

        self.name = "results_figure"
        self.sub_figures = []
        self.kwargs = kwargs

        self.experiments = experiments

        if render:
            self.render()

    def render(self):
        GroupTercileArrowsAxes = [self.figure.add_subplot(2, 2, 1), self.figure.add_subplot(2, 2, 2)]
        self.sub_figures.append(GroupTercileArrows(experiment=self.experiments["trimmed"],
                                                   axes=GroupTercileArrowsAxes, **self.kwargs))
        GroupPitchNormalAxes = [self.figure.add_subplot(4, 2, 5),
                                self.figure.add_subplot(4, 2, 7)]

        # combine x-axes
        GroupPitchNormalAxes[0].get_shared_x_axes().join(GroupPitchNormalAxes[0], GroupPitchNormalAxes[1])

        self.sub_figures.append(GroupPitchNormal(experiment=self.experiments["raw"],
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

        self.figure.text(0.01, 0.97, "A", fontsize="xx-large", fontweight="bold")
        self.figure.text(0.50, 0.97, "B", fontsize="xx-large", fontweight="bold")
        self.figure.text(0.01, 0.47, "C", fontsize="xx-large", fontweight="bold")
        self.figure.text(0.50, 0.47, "D", fontsize="xx-large", fontweight="bold")


class SubFigure:
    def __init__(self, colormap: dict = None, **kwargs):
        if colormap is None:
            self.colormap = {"AD Patients": "darkgoldenrod", "Controls": "teal"}
        else:
            self.colormap = colormap


class GroupTercileCenteringBars(SubFigure):
    def __init__(self, experiment: Experiment, axes: plt.Axes, plot_order: list = None,
                 colormap: dict = None):
        SubFigure.__init__(self, colormap=colormap)
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
                    "color": self.colormap[group],
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
                      hatch=["/", " ", "/", ""]
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


def default_group_name_map():
    return {"AD Patients": "Patients with AD", "Controls": "Controls"}


class GroupTercileArrows(SubFigure):
    def __init__(self, axes: list, experiment: Experiment, plot_order: list = None,
                 colormap: dict = None):
        SubFigure.__init__(self, colormap=colormap)
        self.axes = axes

        self.experiment = experiment
        self.plot_order = [group for group in experiment.subjects] if plot_order is None else plot_order

        self.opacity_map = {
            "UPPER": 0.75,
            "CENTRAL": 0.5,
            "LOWER": 0.75
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

                    base = np.mean(tercile_data["InitialPitch"])
                    apex = np.mean(tercile_data["EndingPitch"])

                    coordinates = np.array([[base, subject_idx + 1 - 0.25],
                                            [base, subject_idx + 1 + 0.25],
                                            [apex, subject_idx + 1]])

                    # draw a filled triangle with these coordinates
                    axis.fill(coordinates[:, 0], coordinates[:, 1],
                              color=self.colormap[group],
                              alpha=self.opacity_map[tercile],
                              edgecolor="black")

            group_name = default_group_name_map()[group]
            axis.set_title(f"{group_name}")
            axis.set_ylabel("Subject Number")
            axis.set_ylim(0, len(subjects) + 1)
            axis.set_yticks([float(i) for i in range(1, len(subjects) + 1, 2)])
            axis.set_yticks([float(i) for i in range(2, len(subjects) + 1, 2)], minor=True)
            axis.set_yticklabels([str(i) for i in range(1, len(subjects) + 1, 2)])
            axis.set_xlabel("Pitch (Cents)")


class CenteringMethods(Figure):
    def __init__(self, motion_points: list, render: bool = True):
        Figure.__init__(self, subplots=(1, 2), gridspec_kw={'width_ratios': [4, 1]})

        self.name = "sample_trial"
        self.motion_points = motion_points

        if render:
            self.render()

    def render(self):
        self.figure.tight_layout()
        self.figure.subplots_adjust(bottom=0.1, left=0.1, wspace=0.0)

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
        axis.set_ylim([min(min(noisy_motion_points) - 20, 0),
                       max(max(noisy_motion_points) + 20, 0)])

        plot_height = axis.get_ylim()[1] - axis.get_ylim()[0]

        # add rectangular overlay for windows
        axis.add_patch(patches.Rectangle((0, 0), 50, plot_height, color="gray", alpha=0.25))
        axis.add_patch(patches.Rectangle((150, 0), 50, plot_height, color="gray", alpha=0.25))

        # add text labels
        axis.text(25, 5, "Initial", horizontalalignment="center", fontsize="medium")
        axis.text(175, 5, "Mid-trial", horizontalalignment="center", fontsize="medium")

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
        TRIANGLE_OFFSET = 200
        coordinates = np.array([[TRIANGLE_OFFSET - TRIANGLE_WIDTH / 2, mean_pitches[0]],
                                [TRIANGLE_OFFSET, mean_pitches[1]],
                                [TRIANGLE_OFFSET + TRIANGLE_WIDTH / 2, mean_pitches[0]]])

        # make the triangle take up the entire right-half of plot (plus a little margin)
        annotation_axis.set_xlim([0, TRIANGLE_OFFSET + TRIANGLE_WIDTH / 2 + 5])

        # continue the dotted line
        annotation_axis.plot([0, TRIANGLE_OFFSET - TRIANGLE_WIDTH / 2], [mean_pitches[0]] * 2, color="black",
                             linestyle="dotted")
        annotation_axis.plot([0, TRIANGLE_OFFSET], [mean_pitches[1]] * 2, color="black", linestyle="dotted")

        # add the text labels
        TEXT_OFFSET = 15
        annotation_axis.text(TEXT_OFFSET, mean_pitches[0], "$F_{\\mathrm{init}}$",
                             backgroundcolor="white",
                             verticalalignment="center",
                             fontsize="medium")

        annotation_axis.text(TEXT_OFFSET, mean_pitches[1], "$F_{\\mathrm{mid"
        # u"\u2212"
        # "trial
                                                           "}}$",
                             backgroundcolor="white",
                             verticalalignment="center",
                             fontsize="medium")

        annotation_axis.fill(coordinates[:, 0], coordinates[:, 1],
                             color="black",
                             alpha=0.5,
                             edgecolor="black")


class GroupPitchNormal(SubFigure):
    def __init__(self, axes: list, experiment: Experiment, plot_order: list = None):
        # do not supply a colormap
        SubFigure.__init__(self)
        self.axes = axes

        # override default colormap
        self.colormap = {
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
            axis.set_ylabel(default_group_name_map()[group])
            axis.set_yticks([])
            axis.set_xlim(*limits)
            axis.tick_params(axis="x", direction="out")

            for index, label in time_window_labels.items():
                mean, std = (np.mean(group_data[index]), np.std(group_data[index]))
                print(f"Group Normal Distribution ({group}) in {index}: {mean} +- {std}")
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

            axis.legend(loc="upper left", frameon=False,
                        prop={'size': 15})

        # only label the bottom-most value
        self.axes[-1].set_xlabel("Pitch (Cents)")

        for axis_index in range(len(self.axes) - 1):
            # disable tick marks for all except the last normal dist
            self.axes[axis_index].set_xticklabels([])


# global settings
__global_styles()
