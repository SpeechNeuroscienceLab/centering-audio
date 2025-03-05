import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.legend_handler import HandlerTuple
from matplotlib import gridspec
import numpy as np
from scipy import signal

from matplotlib.markers import TICKDOWN

import analysis
from subject_analysis import Dataset


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


def group_tercile_bars(dataset: pd.DataFrame, figure: plt.Figure, plot_settings: dict,
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


# mode: zero assumes that all other values of the xrange are 0
# mode: extend assumes that outer values saturate
def __parzen_smooth(x, y, xrange=None, granularity=2, filter_length=50, mode="zeros"):
    if xrange is None:
        xrange = [min(x), max(x)]
    y_upsampled = np.interp(np.arange(xrange[0], xrange[1], granularity), x, y)

    # patch in y values to enable saturation
    if mode == "extend":
        y_upsampled = np.append(np.repeat(y_upsampled[0], filter_length), y_upsampled)
        y_upsampled = np.append(y_upsampled, np.repeat(y_upsampled[-1], filter_length))

    # parzen kernel
    parzen = signal.windows.parzen(filter_length)
    normalized_parzen = parzen / np.sum(parzen)

    smooth_y = np.convolve(normalized_parzen, y_upsampled, mode='same')
    smooth_y = smooth_y[filter_length:smooth_y.shape[0] - filter_length + 1]

    x = np.linspace(xrange[0], xrange[1], len(smooth_y))
    return x, smooth_y


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

            x, smooth_hist = __parzen_smooth(bin_centers, hist, xrange=xrange,
                                             granularity=pitch_granularity,
                                             filter_length=filter_length)

            if smooth_on:
                ax.fill(x, smooth_hist, alpha=0.5, color=plot_settings["colormap"][column])
            else:
                ax.bar(bin_centers, hist,
                       alpha=0.5, color=plot_settings["colormap"][column], edgecolor=plot_settings["colormap"][column],
                       width=np.mean(bin_edges[:-1] - bin_edges[1:]))
        ax.set_title(group)
        ax.set_yticks([])


def group_subject_overshoot_stacked_bars(dataset: pd.DataFrame, figure: plt.Figure,
                                         plot_settings: dict,
                                         group_column="Group Name",
                                         subject_column="Subject Name",
                                         centering_column="Centering (Cents)",
                                         movement_column="Normalized Pitch Movement"):
    # note: axes should match the number of groups
    axes = figure.get_axes()

    for i, group in enumerate(dataset[group_column].unique()):
        group_data = dataset[dataset[group_column] == group]
        axis = axes[i]

        plot_height = []
        plot_err = []

        for subject in group_data[subject_column].unique():
            subject_data = group_data[group_data[subject_column] == subject]
            centering_class = np.zeros_like(subject_data[centering_column])
            centering_class[subject_data[centering_column] > 0] = 1  # positive centering
            centering_class[(subject_data[centering_column] < 0) & (0 < subject_data[movement_column])] = 0  # overshoot
            centering_class[
                (subject_data[centering_column] < 0) & (subject_data[movement_column] < 0)] = -1  #anticentering

            # divide by zero fix
            trial_percentages = np.array([
                np.sum(centering_class == 1) / centering_class.shape[0],
                np.sum(centering_class == 0) / centering_class.shape[0],
                np.sum(centering_class == -1) / centering_class.shape[0],
            ])

            plot_height.append(trial_percentages)

        # plot the triples
        axis.bar(np.arange(len(plot_height)), [x[0] for x in plot_height], bottom=[0 for x in plot_height],
                 label="Centering Frequency", color="green", alpha=0.75)
        axis.bar(np.arange(len(plot_height)), [x[1] for x in plot_height], bottom=[x[0] for x in plot_height],
                 label="Overshoot Frequency", color="violet", alpha=0.75)
        axis.bar(np.arange(len(plot_height)), [x[2] for x in plot_height],
                 bottom=[x[0] + x[1] for x in plot_height],
                 label="Anticentering Frequency", color="red", alpha=0.75)

        axis.set_xticks(ticks=np.arange(len(plot_height)))
        axis.set_xticklabels(np.arange(len(plot_height)) + 1)

        axis.set_xlabel(group)

        if i == len(axes) - 1:
            axis.legend()


def group_subject_centering_overshoot_bars(dataset: pd.DataFrame, figure: plt.Figure,
                                           plot_settings: dict,
                                           group_column="Group Name",
                                           subject_column="Subject Name",
                                           centering_column="Centering (Cents)",
                                           movement_column="Normalized Pitch Movement"):
    # note: axes should match the number of groups
    axes = figure.get_axes()

    for i, group in enumerate(dataset[group_column].unique()):
        group_data = dataset[dataset[group_column] == group]
        axis = axes[i]

        plot_height = []
        # TODO: error bars?
        plot_err = [[0 for _ in range(4)] for _ in range(len(group_data[subject_column].unique()))]

        for subject in group_data[subject_column].unique():
            subject_data = group_data[group_data[subject_column] == subject]
            centering_class = np.zeros_like(subject_data[centering_column])

            centering_class[subject_data[centering_column] >= 0] = 1  # centering
            centering_class[subject_data[centering_column] < 0] = 2  #anticentering
            centering_class[np.logical_and(subject_data[centering_column] > 0,
                                           np.abs(subject_data[movement_column]) > np.abs(
                                               subject_data[centering_column]))] = 3  # centering + overshoot
            centering_class[np.logical_and(subject_data[centering_column] < 0,
                                           subject_data[movement_column] > 0)] = 4  # anticentering + overshoot

            data_by_class = [subject_data[np.logical_or(centering_class == 1, centering_class == 3)],
                             subject_data[np.logical_or(centering_class == 2, centering_class == 4)],
                             subject_data[centering_class == 3],
                             subject_data[centering_class == 4]]

            # divide by zero fix
            trial_percentages = np.array([
                class_data.shape[0] / subject_data.shape[0] for class_data in data_by_class
            ])

            plot_height.append(np.array([0 if np.isnan(x) else x for x in [
                np.abs(np.mean(class_data[centering_column])) for class_data in data_by_class
            ]]))

        # plot the triples
        spacing = 5
        relative_spacing = 1
        axis.bar(np.arange(len(plot_height)) * spacing, [x[0] for x in plot_height], yerr=[x[0] for x in plot_err],
                 label="Mean Centering", color="green", alpha=0.75)
        axis.bar(np.arange(len(plot_height)) * spacing + relative_spacing, [x[1] for x in plot_height],
                 yerr=[x[2] for x in plot_err],
                 label="Mean Anticentering", color="red", alpha=0.75)
        axis.bar(np.arange(len(plot_height)) * spacing + 2 * relative_spacing, [x[2] for x in plot_height],
                 yerr=[x[1] for x in plot_err],
                 label="Mean Overshoot Centering", color="teal", alpha=0.75)
        axis.bar(np.arange(len(plot_height)) * spacing + 3 * relative_spacing, [x[3] for x in plot_height],
                 yerr=[x[1] for x in plot_err],
                 label="Mean Overshoot Anticentering", color="purple", alpha=0.75)

        axis.set_xticks(ticks=np.arange(len(plot_height)) * spacing)
        axis.set_xticklabels(np.arange(len(plot_height)) + 1)

        axis.set_xlabel(group)

        if i == len(axes) - 1:
            axis.legend()


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
                      alpha=plot_settings["motion_alpha"],
                      lw=plot_settings["motion_lw"])
        axis.vlines(0, 0, initial_dev.shape[0])

    if title is not None:
        axes[0].set_xlabel(f"Pitch (cents) {title}")


def group_average_pitch_track(
        subjects: list, figure: plt.Figure, plot_settings: dict,
        title="Group Average Pitch Track"
):
    pitch_table = None
    taxis = None
    tercile_vector = None
    for subject in subjects:
        if taxis is None or pitch_table is None or tercile_vector is None:
            taxis = subject.taxis
            pitch_table = analysis.cents(subject.trials)
            tercile_vector = analysis.compute_trial_tercile_vector(taxis, pitch_table)
        else:
            new_taxis = np.union1d(taxis, subject.taxis)

            tercile_vector = np.hstack((
                tercile_vector,
                analysis.compute_trial_tercile_vector(subject.taxis, subject.trials)
            ))
            pitch_table = np.vstack((
                np.array([np.interp(new_taxis, taxis, pitch_vector)
                          for pitch_vector in pitch_table]),
                np.array([np.interp(new_taxis, subject.taxis, pitch_vector)
                          for pitch_vector in analysis.cents(subject.trials)])
            ))
            taxis = new_taxis

    tercile_masks = [tercile_vector == i for i in range(3)]
    tercile_labels = ["LOWER", "CENTRAL", "UPPER"]

    axis = figure.get_axes()[0]
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Pitch (cents)")

    full_mask = np.logical_and((taxis >= 0.000), (taxis <= 0.200))

    for tercile in range(3):
        tercile_trials = pitch_table[tercile_masks[tercile], :]
        mean_trial = np.mean(tercile_trials, axis=0)
        error_trial = np.std(tercile_trials, axis=0) / np.sqrt(max(mean_trial.shape))
        plt.plot(taxis[full_mask], mean_trial[full_mask], label=f"{tercile_labels[tercile]}")

        plt.fill_between(taxis[full_mask],
                         mean_trial[full_mask] - error_trial[full_mask],
                         mean_trial[full_mask] + error_trial[full_mask],
                         alpha=0.5)

    median_trial = np.median(pitch_table, axis=0)


def group_smooth_centering_distribution(dataset: pd.DataFrame, figure: plt.Figure,
                                        plot_settings: dict,
                                        groups: tuple,
                                        group_column="Group Name",
                                        distribution_columns=("Starting Pitch (Cents)", "Ending Pitch (Cents)"),
                                        title=None):
    # note: axes should match the number of groups
    axes = figure.get_axes()

    groups = filter(lambda x: x in dataset[group_column].unique(), groups)

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

        # pack the trajectories into a matrix for faster computation
        trajectories = np.zeros([initial_dev.shape[0], 4])
        # overshoot, positive_centering, negative centering
        colors = ["violet", "green", "red"]
        for i, (startx, endx) in enumerate(zip(initial_dev, mid_dev)):
            trajectories[i] = [i, startx, endx, 1 if np.abs(startx) > np.abs(endx) else (0 if startx > endx else -1)]
        # smooth out the start points
        filter_length = 75

        i, smooth_startx = __parzen_smooth(trajectories[:, 0], trajectories[:, 1],
                                           filter_length=filter_length, mode="extend")
        for tclass in [1, -1]:
            class_trajectories = trajectories[trajectories[:, 3] == tclass]

            new_x = np.linspace(trajectories[:, 0].min(), trajectories[:, 0].max(), len(trajectories))
            class_trajectories = np.column_stack(
                [new_x] + [np.interp(new_x, class_trajectories[:, 0], class_trajectories[:, i]) for i in
                           range(1, class_trajectories.shape[1])])

            i, smooth_endx = __parzen_smooth(class_trajectories[:, 0], class_trajectories[:, 2],
                                             filter_length=filter_length, mode="extend")
            axis.fill_betweenx(i, smooth_startx, smooth_endx, color=colors[tclass], alpha=0.7)

    if title is not None:
        axes[0].set_xlabel(f"Pitch (cents) {title}")


def draw_triangle(axis, x, start, end, width=0.5, color="black", alpha=0.5):
    axis.fill([x, x + width / 2, x - width / 2], [end, start, start],
              color=color, lw=2, alpha=alpha)


def group_arrow_distribution(dataset: pd.DataFrame,
                             figure: plt.Figure,
                             plot_settings: dict,
                             groups: tuple,
                             group_column="Group Name",
                             title=None):
    axes = figure.get_axes()

    for i, group in enumerate(groups):
        group_data = dataset[dataset[group_column] == group]
        axis = axes[i]

        for j, subject in enumerate(group_data["Subject Name"].unique()):
            subject_data = dataset[dataset["Subject Name"] == subject]

            # get the average start, end pitch of each tercile
            pitch_vectors = {}

            for tercile in subject_data["Tercile"].unique():
                tercile_data = subject_data[subject_data["Tercile"] == tercile]

                pitch_vectors[tercile] = {
                    "start": np.mean(tercile_data["Starting Pitch (Cents)"]),
                    "end": np.mean(tercile_data["Ending Pitch (Cents)"])
                }

                draw_triangle(axis, j + 0.5, pitch_vectors[tercile]["start"], pitch_vectors[tercile]["end"],
                              color=plot_settings["colormap"][group],
                              alpha=0.2 if tercile == "CENTRAL" else 0.5)

        axis.set_xticks(np.arange(len(group_data["Subject Name"].unique())) + 0.5)
        axis.set_xticklabels(np.arange(len(group_data["Subject Name"].unique())) + 1)
        # set x ticks labels
        axis.set_xlabel("Subject #")

        if i == 0:
            axis.set_ylabel("Pitch (cents)")

        axis.set_title(group)


def group_pitch_magnitude_comparison(dataset: pd.DataFrame,
                           figure: plt.Figure,
                           plot_settings: dict,
                           groups: tuple,
                        group_column="Group Name",
                           subgroup_column="Tercile",
                           pitch_column="Starting Pitch (Cents)",
                           ):
    labels = []
    heights = []
    errors = []
    colors = []

    for group in groups:
        group_data = dataset[dataset[group_column] == group]
        print(f"Group: {group} has {group_data.shape[0]} rows")
        for subgroup in np.sort(group_data[subgroup_column].unique()):
            print(f"Subgroup: {subgroup} has {group_data[group_data[subgroup_column] == subgroup].shape[0]} rows")
            subgroup_data = group_data[group_data[subgroup_column] == subgroup]
            heights.append(np.mean(np.abs(subgroup_data[pitch_column])))
            errors.append(standard_error(subgroup_data[pitch_column]))
            labels.append(f"{group}\n{subgroup}")
            colors.append(plot_settings["colormap"][group][subgroup])

    axis = figure.get_axes()[0]

    x_pos = np.arange(len(labels))

    axis.set_title(f"{pitch_column} Magnitude Comparison")

    axis.scatter(x=x_pos, y=heights, color=colors, zorder=2, s=250, marker='h',
                 edgecolors=plot_settings["error-color"])

    axis.errorbar(x=x_pos, y=heights, yerr=errors, color=plot_settings["error-color"], zorder=1,
                  elinewidth=plot_settings["line-width"],
                  capthick=plot_settings["line-width"],
                  capsize=plot_settings["error-cap-size"],
                  ls=plot_settings["error-line-style"])
    axis.set_xticks(x_pos)
    axis.set_xticklabels(labels)
