import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm

############ useful constants and shared resources #############
scales = dict()
# plot specific properties (default)
scales["deviations-bars"] = ((), (0, 100))
scales["pitches-histogram"] = ((-750, 750),(0, 1))
scales["pitches-histogram-bin-size"] = 20
scales["patch-errorbar-height"] = 0.3
scales["whisker-errorbar-height"] = 0.25
scales["whisker-errorbar-width"] = 5

# figure generation properties
scales["figure-figsize"] = (20, 14)
scales["default-font-size"] = 16
scales["qq-outlier-multiplier"] = 2

# colors
standard_colors = ["lightsteelblue","peru"]
special_colors = ["saddlebrown", "salmon", "tomato"]

# holds the figure object for later use
figs = []
fig_names = []

############ save figures to output folder ##########
def save_figs(output_path):
    for i in range(0, len(figs)):
        figs[i].savefig(output_path + "fig_" + fig_names[i]  + ".png")
############ subject tercile arrows ##############

# constants
draw_arrow_height = 0.75

def draw_arrow(startx, endx, starty, endy, color='black', style='flag', errors=(0, 0)):
    # start at the starting x, and draw a vertical line from y +- height/2
    # then draw two more lines
    if(style == "flag"):
        x_points = [startx, startx, endx, startx]
        y_points = [starty - draw_arrow_height/2, starty + draw_arrow_height/2, endy, starty - draw_arrow_height/2]
    if(style == "minimal"):
        x_points = [startx, endx]
        y_points = [starty, endy]
    if(style == "trapizoid"):
        # height is always 1
        x_points = [startx - errors[0], startx + errors[0], endx + errors[1], endx - errors[1], startx - errors[0]]
        y_points = [starty, starty, endy, endy, starty]

    plt.plot(x_points, y_points, color=color)

def subject_tercile_arrows(group_list, subject_list, terciles_data):
    figs.append(plt.figure(figsize=scales["figure-figsize"]))
    fig_names.append("subject_tercile_arrows")
    
    limits = [0, 0]

    plt.rc('font', size=scales["default-font-size"], weight="bold")
    for group_idx in range(0, len(group_list)):
        plt.subplot(1, len(group_list), group_idx + 1)
        plt.title(str(group_list[group_idx]), weight="bold")
        for subject_idx in range(0, len(subject_list[group_idx])):
            # for each subject
            for tercile_idx in range(0, 3):
                arrow_color = standard_colors[0] if tercile_idx % 2 == 0 else standard_colors[1]
                draw_arrow(terciles_data[group_idx][subject_idx][tercile_idx][2], terciles_data[group_idx][subject_idx][tercile_idx][3], subject_idx, subject_idx, arrow_color)
        plt.xlabel('Pitch (Cents)', weight="bold")
        plt.yticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        

        # set the limits
        min, max, _, _ = plt.axis()
        if(min < limits[0]):
            limits[0] = min
        if(max > limits[1]):
            limits[1] = max

    # set all of the limits of the subplots to match
    for group_idx in range(0, len(group_list)):
        plt.subplot(1, len(group_list), group_idx + 1)
        plt.xlim(limits)

################# subject tercile arrows with error ###################

def standard_error(data):
    return np.std(data, ddof=1) / np.sqrt(np.size(data)) # degree of freedom is 1 because sample standard deviation

def patch_errorbar(x, y, xerr=0, color=special_colors[0], width=scales["patch-errorbar-height"], size_adjustment=1):
    width = width * size_adjustment
    
    xpts = [x - xerr, x + xerr, x + xerr, x - xerr, x - xerr]
    ypts = [y - width/2, y - width/2, y + width/2, y + width/2, y - width/2]

    plt.fill(xpts, ypts, color, alpha=0.5)

def whisker_errorbar(x, y, xerr=0, color=special_colors[0], width=scales["whisker-errorbar-width"], height=scales["whisker-errorbar-height"]):
    # start by drawing a line between the points
    xpts = [x - xerr, x + xerr]
    ypts = [y, y]
    plt.plot(xpts, ypts, color=color)

    # now plot the whiskers
    xpts = [x - xerr + width, x - xerr, x - xerr + width]
    ypts = [y + height, y, y - height]
    plt.plot(xpts, ypts, color=color)

    xpts = [x + xerr - width, x + xerr, x + xerr - width]
    ypts = [y + height, y, y - height]
    plt.plot(xpts, ypts, color=color)
    

def subject_tercile_arrows_with_error(group_list, subject_list, trial_data):
    figs.append(plt.figure(figsize=scales["figure-figsize"]))
    fig_names.append("subject_tercile_arrows_with_error")
    
    limits = [0, 0]

    plt.rc('font', size=scales["default-font-size"], weight="bold")
    for group_idx in range(0, len(group_list)):
        # foreach group
        plt.subplot(1, len(group_list), group_idx + 1)
        group_data = trial_data[trial_data[:, 0] == group_idx, :]
        plt.title(str(group_list[group_idx]), weight="bold")
        for subject_idx in range(0, len(subject_list[group_idx])):
            # foreach subject
            subject_data = group_data[group_data[:, 1] == subject_idx, :]
            terciles = [-1, 0, 1]
            pitch = [0, 0, 0]
            for tercile in terciles:
                # foreach tercile
                tercile_data = subject_data[subject_data[:, 7] == tercile, :]
                pitch[tercile + 1] = \
                        ((np.mean(tercile_data[:, 2]), standard_error(tercile_data[:, 2])),
                        (np.mean(tercile_data[:, 3]), standard_error(tercile_data[:, 3])))
            # all pitches with errors have been written
            # plot the subject arrows
            
            draw_arrow(pitch[0][0][0], pitch[0][1][0], subject_idx, subject_idx, color=standard_colors[0], style="flag")
            whisker_errorbar(pitch[0][0][0], subject_idx, xerr=pitch[0][0][1], color=special_colors[1])
            whisker_errorbar(pitch[0][1][0], subject_idx, xerr=pitch[0][1][1], color=special_colors[1])
            draw_arrow(pitch[1][0][0], pitch[1][1][0], subject_idx, subject_idx, color=standard_colors[0], style="flag")
            draw_arrow(pitch[2][0][0], pitch[2][1][0], subject_idx, subject_idx, color=standard_colors[0], style="flag")
            whisker_errorbar(pitch[2][0][0], subject_idx, xerr=pitch[2][0][1], color=special_colors[1])
            whisker_errorbar(pitch[2][1][0], subject_idx, xerr=pitch[2][1][1], color=special_colors[1])

        plt.xlabel('Pitch (Cents)', weight="bold")
        plt.yticks([])
        plt.gca().axes.get_yaxis().set_ticks([])



        # set the limits
        min, max, _, _ = plt.axis()
        if(min < limits[0]):
            limits[0] = min
        if(max > limits[1]):
            limits[1] = max

    # set all of the limits of the subplots to match
    for group_idx in range(0, len(group_list)):
        plt.subplot(1, len(group_list), group_idx + 1)
        plt.xlim(limits)


########################## subject tercile trapizoid with error ####################################

def subject_tercile_trapizoids_with_error(group_list, subject_list, trial_data):
    figs.append(plt.figure(figsize=scales["figure-figsize"]))
    fig_names.append("subject_tercile_trapizoids_with_error")
    
    limits = [0, 0]

    plt.rc('font', size=scales["default-font-size"], weight="bold")
    for group_idx in range(0, len(group_list)):
        # foreach group
        plt.subplot(1, len(group_list), group_idx + 1)
        group_data = trial_data[trial_data[:, 0] == group_idx, :]
        plt.title(str(group_list[group_idx]), weight="bold")
        for subject_idx in range(0, len(subject_list[group_idx])):
            # foreach subject
            subject_data = group_data[group_data[:, 1] == subject_idx, :]
            terciles = [-1, 0, 1]
            pitch = [0, 0, 0]
            for tercile in terciles:
                # foreach tercile
                tercile_data = subject_data[subject_data[:, 7] == tercile, :]
                pitch[tercile + 1] = \
                        ((np.mean(tercile_data[:, 2]), standard_error(tercile_data[:, 2])),
                        (np.mean(tercile_data[:, 3]), standard_error(tercile_data[:, 3])))
            # all pitches with errors have been written
            # plot the subject arrows
            
            draw_arrow(pitch[0][0][0], pitch[0][1][0], subject_idx, subject_idx + draw_arrow_height, color=standard_colors[0], style="trapizoid", errors=(pitch[0][0][1], pitch[0][1][1]))
            draw_arrow(pitch[1][0][0], pitch[1][1][0], subject_idx, subject_idx + draw_arrow_height, color=standard_colors[1], style="trapizoid", errors=(pitch[1][0][1], pitch[0][1][1]))
            draw_arrow(pitch[2][0][0], pitch[2][1][0], subject_idx, subject_idx + draw_arrow_height, color=standard_colors[0], style="trapizoid", errors=(pitch[2][0][1], pitch[0][1][1]))

        plt.xlabel('Pitch (Cents)', weight="bold")
        plt.yticks([])
        plt.gca().axes.get_yaxis().set_ticks([])



        # set the limits
        min, max, _, _ = plt.axis()
        if(min < limits[0]):
            limits[0] = min
        if(max > limits[1]):
            limits[1] = max

    # set all of the limits of the subplots to match
    for group_idx in range(0, len(group_list)):
        plt.subplot(1, len(group_list), group_idx + 1)
        plt.xlim(limits)


################# group centering bars #####################
def group_centering_bars(group_list, trial_data):
    figs.append(plt.figure(figsize=scales["figure-figsize"]))
    fig_names.append("group_centering_bars")

    plt.rc('font', size=scales["default-font-size"], weight="bold")
    plt.title("Average Centering (Cents)", weight="bold")

    groups_included = np.unique(trial_data[:, 0])

    bar_labels = []
    SPACER = 0.3
    bar_locations = []

    for group_idx in groups_included:
        group_data = trial_data[trial_data[:, 0] == group_idx, :]
        peripheral_centering = (group_data[group_data[:, 7] == 1, 4], group_data[group_data[:, 7] == -1, 4])

        # each bar chart needs two datapoints. We need the mean and error
        centering_mean = (np.mean(peripheral_centering[0]), np.mean(peripheral_centering[1]))
        centering_error = (standard_error(peripheral_centering[0]), standard_error(peripheral_centering[1]))

        bar_locations.append(group_idx * 2 + group_idx * SPACER)
        bar_locations.append(group_idx * 2 + 1 + group_idx * SPACER)
        
        plt.bar(group_idx * 2 + group_idx * SPACER, centering_mean[0], color=standard_colors[int(group_idx)])
        plt.bar(group_idx * 2 + 1 + group_idx * SPACER, centering_mean[1], color=standard_colors[int(group_idx)])
        

        plt.errorbar(group_idx * 2 + group_idx * SPACER, centering_mean[0], yerr=centering_error[0], color=special_colors[0], capsize=10)
        plt.errorbar(group_idx * 2 + 1 + group_idx * SPACER, centering_mean[1], yerr=centering_error[1], color=special_colors[0], capsize=10)
        
        bar_labels.append(group_list[int(group_idx)] + " (Upper)")

        bar_labels.append(group_list[int(group_idx)] + " (Lower)")
            

    plt.xticks(bar_locations, bar_labels, weight="bold")

    plt.grid(False, axis='x')

    plt.ylabel('Centering (Cents)', weight="bold")

def group_centering_bars_o2(group_list, trial_data):
    DOTSIZE = 500

    figs.append(plt.figure(figsize=scales["figure-figsize"]))
    fig_names.append("group_centering_bars_o2")

    plt.rc('font', size=scales["default-font-size"], weight="bold")
    plt.title("Average Centering (Cents)", weight="bold")

    groups_included = np.unique(trial_data[:, 0])

    bar_labels = []

    for group_idx in groups_included:
        group_data = trial_data[trial_data[:, 0] == group_idx, :]
        peripheral_centering = (group_data[group_data[:, 7] == 1, 4], group_data[group_data[:, 7] == -1, 4])

        # each bar chart needs two datapoints. We need the mean and error
        centering_mean = (np.mean(peripheral_centering[0]), np.mean(peripheral_centering[1]))
        centering_error = (standard_error(peripheral_centering[0]), standard_error(peripheral_centering[1]))

        plt.errorbar(group_idx * 2, centering_mean[0], yerr=centering_error[0], color=special_colors[0], capsize=10)
        plt.errorbar(group_idx * 2 + 1, centering_mean[1], yerr=centering_error[1], color=special_colors[0], capsize=10)
        plt.scatter(group_idx * 2, centering_mean[0], color=standard_colors[0], s=DOTSIZE)
        plt.scatter(group_idx * 2 + 1, centering_mean[1], color=standard_colors[1], s=DOTSIZE)
        plt.scatter(0, 0, s=0)# hack to force 0 to show up in figure
        
        bar_labels.append(group_list[int(group_idx)] + " (Upper)")
        bar_labels.append(group_list[int(group_idx)] + " (Lower)")

    plt.xticks(np.arange(len(group_list)*2), bar_labels, weight="bold")
    

    plt.ylabel('Centering (Cents)', weight="bold")



################# group starting deviations bars #####################
def calculate_deviations_limits(group_list, trial_data):
    # this should fit the bar height such that all of the bars easily fit 
    max_height = 0

    for group_idx in range(0, len(group_list)):
        group_data = trial_data[trial_data[:, 0] == group_idx, :]
        peripheral_starting_deviations = (group_data[group_data[:, 7] == 1, 5], group_data[group_data[:, 7] == -1, 5])

        # each bar chart needs two datapoints. We need the mean and error
        starting_deviations_mean = np.array([np.mean(peripheral_starting_deviations[0]), np.mean(peripheral_starting_deviations[1])])
        starting_deviations_error = np.array([standard_error(peripheral_starting_deviations[0]), standard_error(peripheral_starting_deviations[1])])
        height = np.max(starting_deviations_mean + starting_deviations_error)
        if(height > max_height):
            max_height = height

    return ((), (0, max_height * 1.1))



def group_initial_deviations_bars(group_list, trial_data):
    figs.append(plt.figure(figsize=scales["figure-figsize"]))
    fig_names.append("group_initial_deviation_bars")

    plt.rc('font', size=scales["default-font-size"], weight="bold")
    plt.ylim(0, scales["deviations-bars"][1][1])

    plt.title("Average Initial Deviations (Cents)", weight="bold")

    groups_included = np.unique(trial_data[:, 0])
    
    bar_labels = []
    SPACER = 0.3
    bar_locations = []

    for tercile in [0, 1]:
        for group_idx in groups_included:
            group_data = trial_data[trial_data[:, 0] == group_idx, :]
            peripheral_starting_deviations = (group_data[group_data[:, 7] == 1, 5], group_data[group_data[:, 7] == -1, 5])

            # each bar chart needs two datapoints. We need the mean and error
            starting_deviations_mean = (np.mean(peripheral_starting_deviations[0]), np.mean(peripheral_starting_deviations[1]))
            starting_deviations_error = (standard_error(peripheral_starting_deviations[0]), standard_error(peripheral_starting_deviations[1]))

            bar_locations.append(group_idx + len(groups_included)*tercile + SPACER*tercile)
            
            plt.bar(group_idx + len(groups_included)*tercile + SPACER*tercile, starting_deviations_mean[tercile], color=standard_colors[int(group_idx)])

            plt.errorbar(group_idx + len(groups_included)*tercile + SPACER*tercile, starting_deviations_mean[tercile], yerr=starting_deviations_error[tercile], color=special_colors[0], capsize=10)
            if(tercile == 0):
                bar_labels.append(group_list[int(group_idx)] + " (Upper)")
            else:
                bar_labels.append(group_list[int(group_idx)] + " (Lower)")
        if(tercile != 1):
            bar_labels.append(" ")
            bar_locations.append(len(groups_included) + 1)
            

    plt.xticks(bar_locations, bar_labels, weight="bold")

    plt.ylabel('Initial Deviations (Cents)', weight="bold")

################# group ending deviations bars #####################
def group_midtrial_deviations_bars(group_list, trial_data):
    figs.append(plt.figure(figsize=scales["figure-figsize"]))
    fig_names.append("group_midtrial_deviations_bars")

    plt.rc('font', size=scales["default-font-size"], weight="bold")
    plt.ylim(0, scales["deviations-bars"][1][1])

    plt.title("Average Mid-trial Deviations (Cents)", weight="bold")

    groups_included = np.unique(trial_data[:, 0])

    bar_labels = []
    SPACER = 0.3
    bar_locations = []

    for tercile in [0, 1]:
        for group_idx in groups_included:
            group_data = trial_data[trial_data[:, 0] == group_idx, :]
            peripheral_ending_deviations = (group_data[group_data[:, 7] == 1, 6], group_data[group_data[:, 7] == -1, 6])

            # each bar chart needs two datapoints. We need the mean and error
            ending_deviations_mean = (np.mean(peripheral_ending_deviations[0]), np.mean(peripheral_ending_deviations[1]))
            ending_deviations_error = (standard_error(peripheral_ending_deviations[0]), standard_error(peripheral_ending_deviations[1]))

            bar_locations.append(group_idx + len(groups_included)*tercile + SPACER*tercile)
            
            plt.bar(group_idx + len(groups_included)*tercile + SPACER*tercile, ending_deviations_mean[tercile], color=standard_colors[int(group_idx)])

            plt.errorbar(group_idx + len(groups_included)*tercile + SPACER*tercile, ending_deviations_mean[tercile], yerr=ending_deviations_error[tercile], color=special_colors[0], capsize=10)
            if(tercile == 0):
                bar_labels.append(group_list[int(group_idx)] + " (Upper)")
            else:
                bar_labels.append(group_list[int(group_idx)] + " (Lower)")
        if(tercile != 1):
            bar_labels.append(" ")
            bar_locations.append(len(groups_included) + 1)
            

    plt.xticks(bar_locations, bar_labels, weight="bold")

    
    
    plt.ylabel('Mid-trial Deviations (Cents)', weight="bold")


######################## group starting pitches histogram #######################

# calc bins allows us to specify bin size
def calc_bins(data, target_width):
    bound_min = -1.0 * (np.min(data) % target_width - np.min(data))
    bound_max = np.max(data) - np.max(data) % target_width + target_width
    n = int((bound_max - bound_min) / target_width) + 1
    return np.linspace(bound_min, bound_max, n)

# plots a histogram giving equal weights to each of the x values. 
def normalized_hist(data, bins, width=40):
    hist, bin_edges = np.histogram(data, bins=bins) 
    # the area of this should always be 1
    hist = hist/np.sum(hist) * 100/width
    plt.bar(bin_edges[0:len(bin_edges)-1], hist, width=width, color=standard_colors[0])

def group_initial_pitches_histogram(group_list, trial_data):
    figs.append(plt.figure(figsize=scales["figure-figsize"]))
    fig_names.append("group_initial_pitches_histogram")
    plt.rc('font', size=scales["default-font-size"], weight="bold")


    plt.suptitle("Initial Pitch (Cents)", weight="bold")

    # vertical comparison
    for group_idx in range(0, len(group_list)):
        plt.subplot(len(group_list), 1, group_idx + 1)
        plt.title(str(group_list[group_idx]), weight="bold")
        
        # get the starting pitch
        starting_deviations = trial_data[trial_data[:, 0] == group_idx, 2]

        normalized_hist(starting_deviations, calc_bins(starting_deviations, scales["pitches-histogram-bin-size"]), scales["pitches-histogram-bin-size"])
        
        plt.xlabel("Pitch (Cents relative to subject median)", weight="bold")
        plt.ylabel("Frequency (Normalized)", weight="bold")
        limits = scales["pitches-histogram"]
        plt.xlim(limits[0][0], limits[0][1])
        plt.ylim(limits[1][0], limits[1][1])


######################## group ending pitches histogram #######################
def group_midtrial_pitches_histogram(group_list, trial_data):
    figs.append(plt.figure(figsize=scales["figure-figsize"]))
    fig_names.append("group_midtrial_pitches_histogram")
    
    plt.rc('font', size=scales["default-font-size"])
    plt.suptitle("Mid-trial Pitch (Cents)", weight="bold")

    # vertical comparison
    for group_idx in range(0, len(group_list)):
        plt.subplot(len(group_list), 1, group_idx + 1)
        plt.title(str(group_list[group_idx]), weight="bold")
        
        # get the ending pitch
        ending_deviations = trial_data[trial_data[:, 0] == group_idx, 3]

        normalized_hist(ending_deviations, calc_bins(ending_deviations, scales["pitches-histogram-bin-size"]), scales["pitches-histogram-bin-size"])
        plt.xlabel("Pitch (Cents relative to subject median)", weight="bold")
        plt.ylabel("Frequency (Normalized)", weight="bold")
        limits = scales["pitches-histogram"]
        plt.xlim(limits[0][0], limits[0][1])
        plt.ylim(limits[1][0], limits[1][1])

######################### groups pitches qq plots  #######################
def group_pitches_qq(group_list, trial_data):
    # metric
    metric = 2

    # remove outliers 
    trial_data = trial_data[abs(trial_data[:, metric] - np.mean(trial_data[:, metric])) < scales["qq-outlier-multiplier"] * np.std(trial_data[:, metric]), :]

    fig_names.append("group_pitches_qq")
    qqfigure, subplots = plt.subplots(nrows=len(group_list), ncols=2, figsize=scales["figure-figsize"])
    figs.append(qqfigure)
    plt.rc('font', size=scales["default-font-size"], weight="bold")
    
    

    for group_idx in range(0, len(group_list)):
        starting_pitch = trial_data[trial_data[:, 0] == group_idx, 2]
        ending_pitch = trial_data[trial_data[:, 0] == group_idx, 3]
        starting_fig = sm.qqplot(starting_pitch, ax=subplots[group_idx][0])
        sm.qqline(ax=subplots[group_idx][0], x=[-3, 3], y=[-200, 200], line="r") # generates a warning. This is due to a bug in dependencies.
        subplots[group_idx][0].title.set_text(group_list[group_idx] + " Initial Pitch")
        ending_fig = sm.qqplot(ending_pitch, ax=subplots[group_idx][1])
        sm.qqline(ax=subplots[group_idx][1], x=[-3, 3], y=[-200, 200], line="r")
        subplots[group_idx][1].title.set_text(group_list[group_idx] + " Mid-trial Pitch")



