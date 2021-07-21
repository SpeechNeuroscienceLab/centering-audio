import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm

############ useful constants and shared resources #############
scales = dict()
# TODO: ability to set scaling properties from command line instead of source
scales["deviations-bars"] = ((), (0, 100))
scales["pitches-histogram"] = ((-750, 750),(0, 0.75))
scales["pitches-histogram-bin-size"] = 40

# set default figure size
plt.rcParams["figure.figsize"] = (20, 14)
# set default text size
plt.rc('font', size=16)

# colors
standard_colors = ["lightsteelblue","peru"]
special_colors = ["saddlebrown"]

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

def draw_arrow(startx, endx, y, draw_color='black'):
    # start at the starting x, and draw a vertical line from y +- height/2
    # then draw two more lines
    x_points = [startx, startx, endx, startx]
    y_points = [y - draw_arrow_height/2, y + draw_arrow_height/2, y, y - draw_arrow_height/2]
    plt.plot(x_points, y_points, color=draw_color)

def subject_tercile_arrows(group_list, subject_list, terciles_data):
    figs.append(plt.figure())
    fig_names.append("subject_tercile_arrows")
    for group_idx in range(0, len(group_list)):
        plt.subplot(1, len(group_list), group_idx + 1)
        plt.title(str(group_list[group_idx]))
        for subject_idx in range(0, len(subject_list[group_idx])):
            # for each subject
            for tercile_idx in range(0, 3):
                arrow_color = standard_colors[0] if tercile_idx % 2 == 0 else standard_colors[1]
                draw_arrow(terciles_data[group_idx][subject_idx][tercile_idx][2], terciles_data[group_idx][subject_idx][tercile_idx][3], subject_idx, arrow_color)
        plt.xlabel('Pitch (Cents)')
        plt.yticks([])
        plt.gca().axes.get_yaxis().set_ticks([])


################# group centering bars #####################
def standard_error(data):
    return np.std(data, ddof=1) / np.sqrt(np.size(data)) # degree of freedom is 1 because sample standard deviation

def group_centering_bars(group_list, trial_data):
    figs.append(plt.figure())
    fig_names.append("group_centering_bars")

    plt.title("Average Peripheral Centering (Cents)")

    groups_included = np.unique(trial_data[:, 0])

    bar_labels = []

    for group_idx in groups_included:
        group_data = trial_data[trial_data[:, 0] == group_idx, :]
        peripheral_centering = (group_data[group_data[:, 7] == 1, 4], group_data[group_data[:, 7] == -1, 4])

        # each bar chart needs two datapoints. We need the mean and error
        centering_mean = (np.mean(peripheral_centering[0]), np.mean(peripheral_centering[1]))
        centering_error = (standard_error(peripheral_centering[0]), standard_error(peripheral_centering[1]))

        plt.bar(group_idx * 2, centering_mean[0], color=standard_colors[0])
        plt.bar(group_idx * 2 + 1, centering_mean[1], color=standard_colors[1])

        plt.errorbar(group_idx * 2, centering_mean[0], yerr=centering_error[0], color=special_colors[0])
        plt.errorbar(group_idx * 2 + 1, centering_mean[1], yerr=centering_error[1], color=special_colors[0])
        
        bar_labels.append(group_list[int(group_idx)] + " (Upper)")
        bar_labels.append(group_list[int(group_idx)] + " (Lower)")

    plt.xticks(np.arange(len(group_list)*2), bar_labels)

    plt.ylabel('Centering (Cents)')


################# group starting deviations bars #####################
def group_starting_deviations_bars(group_list, trial_data):
    figs.append(plt.figure())
    fig_names.append("group_starting_deviations_bars")

    plt.ylim(0, scales["deviations-bars"][1][1])

    plt.title("Average Peripheral Starting Deviations (Cents)")

    groups_included = np.unique(trial_data[:, 0])

    bar_labels = []

    for group_idx in groups_included:
        group_data = trial_data[trial_data[:, 0] == group_idx, :]
        peripheral_starting_deviations = (group_data[group_data[:, 7] == 1, 5], group_data[group_data[:, 7] == -1, 5])

        # each bar chart needs two datapoints. We need the mean and error
        starting_deviations_mean = (np.mean(peripheral_starting_deviations[0]), np.mean(peripheral_starting_deviations[1]))
        starting_deviations_error = (standard_error(peripheral_starting_deviations[0]), standard_error(peripheral_starting_deviations[1]))

        plt.bar(group_idx * 2, starting_deviations_mean[0], color=standard_colors[0])
        plt.bar(group_idx * 2 + 1, starting_deviations_mean[1], color=standard_colors[1])

        plt.errorbar(group_idx * 2, starting_deviations_mean[0], yerr=starting_deviations_error[0], color=special_colors[0])
        plt.errorbar(group_idx * 2 + 1, starting_deviations_mean[1], yerr=starting_deviations_error[1], color=special_colors[0])
                
        bar_labels.append(group_list[int(group_idx)] + " (Upper)")
        bar_labels.append(group_list[int(group_idx)] + " (Lower)")

    plt.xticks(np.arange(len(group_list)*2), bar_labels)

    plt.ylabel('Starting Deviations (Cents)')

################# group ending deviations bars #####################
def group_ending_deviations_bars(group_list, trial_data):
    figs.append(plt.figure())
    fig_names.append("group_ending_deviations_bars")

    plt.ylim(0, scales["deviations-bars"][1][1])

    plt.title("Average Peripheral Ending Deviations (Cents)")

    groups_included = np.unique(trial_data[:, 0])

    bar_labels = []

    for group_idx in groups_included:
        group_data = trial_data[trial_data[:, 0] == group_idx, :]
        peripheral_ending_deviations = (group_data[group_data[:, 7] == 1, 6], group_data[group_data[:, 7] == -1, 6])

        # each bar chart needs two datapoints. We need the mean and error
        ending_deviations_mean = (np.mean(peripheral_ending_deviations[0]), np.mean(peripheral_ending_deviations[1]))
        ending_deviations_error = (standard_error(peripheral_ending_deviations[0]), standard_error(peripheral_ending_deviations[1]))

        plt.bar(group_idx * 2, ending_deviations_mean[0], color=standard_colors[0])
        plt.bar(group_idx * 2 + 1, ending_deviations_mean[1], color=standard_colors[1])

        plt.errorbar(group_idx * 2, ending_deviations_mean[0], yerr=ending_deviations_error[0], color=special_colors[0])
        plt.errorbar(group_idx * 2 + 1, ending_deviations_mean[1], yerr=ending_deviations_error[1], color=special_colors[0])
        bar_labels.append(group_list[int(group_idx)] + " (Upper)")
        bar_labels.append(group_list[int(group_idx)] + " (Lower)")

    plt.xticks(np.arange(len(group_list)*2), bar_labels)

    
    
    plt.ylabel('Ending Deviations (Cents)')


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

def group_starting_pitches_histogram(group_list, trial_data):
    figs.append(plt.figure())
    fig_names.append("group_starting_pitches_histogram")
    plt.suptitle("Starting Pitch (Cents)")

    # vertical comparison
    for group_idx in range(0, len(group_list)):
        plt.subplot(len(group_list), 1, group_idx + 1)
        plt.title(str(group_list[group_idx]))
        
        # get the starting pitch
        starting_deviations = trial_data[trial_data[:, 0] == group_idx, 2]

        normalized_hist(starting_deviations, calc_bins(starting_deviations, scales["pitches-histogram-bin-size"]), scales["pitches-histogram-bin-size"])
        
        plt.xlabel("Pitch (Cents relative to subject median)")
        plt.ylabel("Frequency (Normalized)")
        limits = scales["pitches-histogram"]
        plt.xlim(limits[0][0], limits[0][1])
        plt.ylim(limits[1][0], limits[1][1])


######################## group ending pitches histogram #######################
def group_ending_pitches_histogram(group_list, trial_data):
    figs.append(plt.figure())
    fig_names.append("group_ending_pitches_histogram")
    plt.suptitle("Ending Pitch (Cents)")

    # vertical comparison
    for group_idx in range(0, len(group_list)):
        plt.subplot(len(group_list), 1, group_idx + 1)
        plt.title(str(group_list[group_idx]))
        
        # get the ending pitch
        ending_deviations = trial_data[trial_data[:, 0] == group_idx, 3]

        normalized_hist(ending_deviations, bins=calc_bins(ending_deviations, scales["pitches-histogram-bin-size"]))
        plt.xlabel("Pitch (Cents relative to subject median)")
        plt.ylabel("Frequency (Normalized)")
        limits = scales["pitches-histogram"]
        plt.xlim(limits[0][0], limits[0][1])
        plt.ylim(limits[1][0], limits[1][1])

######################### groups pitches qq plots  #######################
def group_pitches_qq(group_list, trial_data):
    # 2x2 subplot
    
    fig_names.append("group_pitches_qq")
    qqfigure, subplots = plt.subplots(nrows=len(group_list), ncols=2)
    figs.append(qqfigure)
    
    

    for group_idx in range(0, len(group_list)):
        starting_pitch = trial_data[trial_data[:, 0] == group_idx, 2]
        ending_pitch = trial_data[trial_data[:, 0] == group_idx, 3]
        starting_fig = sm.qqplot(starting_pitch, ax=subplots[group_idx][0])
        subplots[group_idx][0].title.set_text(group_list[group_idx] + " Starting Pitch")
        ending_fig = sm.qqplot(ending_pitch, ax=subplots[group_idx][1])
        subplots[group_idx][1].title.set_text(group_list[group_idx] + " Ending Pitch")
