import os
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from scipy.stats import t
from scipy.signal import butter, filtfilt
from matplotlib.animation import FuncAnimation
from statsmodels.stats.multitest import multipletests


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#plt.rcParams['font.size'] = 12           # Default font size for all text
#plt.rcParams['axes.titlesize'] = 14      # Font size for axes titles
plt.rcParams['axes.labelsize'] = 7      # Font size for axes labels
plt.rcParams['xtick.labelsize'] = 6     # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 6     # Font size for y-axis tick labels
plt.rcParams['legend.fontsize'] = 6     # Font size for legends (if used)


def permutation_test(observed, random, n_permutations=1000):
    
    combined = pd.concat([observed, random], axis=1)
    observed_diff = observed.mean(axis=1, skipna=True) - random.mean(axis=1, skipna=True)
    perm_diffs = []

    for i in range(n_permutations):
        # Shuffle columns and split back into two groups
        permuted_df = combined.sample(frac=1, axis=1, replace=False, random_state=i)
        perm_group1 = permuted_df.iloc[:, :observed.shape[1]]
        perm_group2 = permuted_df.iloc[:, observed.shape[1]:]

        # Calculate mean difference of permuted groups at each time point
        perm_diff = perm_group1.mean(axis=1, skipna=True) - perm_group2.mean(axis=1, skipna=True)
        perm_diffs.append(perm_diff.values)

    # Calculate p-values: proportion of permuted mean differences that are as extreme as observed
    perm_diffs = np.array(perm_diffs)
    p_values = np.array([np.mean(np.abs(perm_diffs[:, i]) >= np.abs(observed_diff[i])) for i in range(len(observed_diff))])

    return observed_diff, p_values



def get_mean_signal(signals, rand_signals=[], plot=False, plot_all=False, fps=8, time_offset=1, ax=None, limits=[], label="", colors=[], colors_fill=[]):

    if len(colors)==0:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "k"]
    if len(colors_fill)==0:
        colors_fill = ["lightblue", "moccasin", "palegreen", "lightcoral", "plum", "tan"]

    if len(signals)==0:
        return []
        
    sort_length = np.sort([len(signal) for signal in signals])
    try:
        max_length = sort_length[-10]
    except:
        max_length = sort_length[-1]
        
    time = np.linspace(-time_offset, max_length/fps-time_offset, max_length)

    confidence_level = 0.95
    alpha = 1 - confidence_level
    critical_value = t.ppf(1 - alpha / 2, df=len(signals) - 1)

    signals_df = pd.DataFrame(index=range(max_length))
    
    if plot:
        if ax==None:
            fig, ax = plt.subplots()

    
    # Accumulate sums and counts for each time step
    for i, signal in enumerate(signals):
        length = len(signal)
        length_used = min(length, max_length)
        signals_df[i] = pd.Series(signal)
        
        if plot_all:
            ax.plot(time[:length_used], signal[:length_used], color='k', alpha=0.1, linewidth=0.3)

    mean_signal = signals_df.mean(axis=1, skipna=True)
    stds = signals_df.std(axis=1, skipna=True, ddof=1)
    sems = stds / np.sqrt(signals_df.count(axis=1))

    lower_bounds = mean_signal - critical_value * sems
    upper_bounds = mean_signal + critical_value * sems


    all_selected_points = []
    all_mean_signals = []
    if len(rand_signals) > 0:
        for rand_traces in rand_signals:
            rand_signals_df = pd.DataFrame(index=range(max_length))
            for i, rand_signal in enumerate(rand_traces):
            #for i, rand_signal in enumerate(rand_signals):
                #length = len(rand_signal)
                rand_signals_df[i] = pd.Series(rand_signal)

            mean_rand_signal = rand_signals_df.mean(axis=1, skipna=True)
        
            if plot:
                lenght = len(mean_rand_signal)
                length_used = min(length, max_length)
                all_mean_signals.append(mean_rand_signal)
                ax.plot(time, mean_rand_signal, color='k', alpha=0.1, linewidth=0.3)
            
            rand_stds = rand_signals_df.std(axis=1, skipna=True, ddof=1)
            rand_sems = rand_stds / np.sqrt(rand_signals_df.count(axis=1))

            rand_lower_bounds = mean_rand_signal - critical_value * rand_sems
            rand_upper_bounds = mean_rand_signal + critical_value * rand_sems

            observed_diff, p_values = permutation_test(signals_df, rand_signals_df)
            adjusted_p_values = multipletests(p_values, alpha=0.05, method='fdr_bh')[1]
            significant_points = np.where(adjusted_p_values < 0.05)[0]
            selected_points = significant_points[significant_points>time_offset*fps]
            #print(selected_points)
            all_selected_points.extend(selected_points)

        sel_points_count = pd.Series(all_selected_points).value_counts()
        sel_points = sel_points_count[sel_points_count > len(rand_signals)*0.7].index
        #print(sel_points)
    

    if plot:
        ax.plot(time, mean_signal, color=colors[0], linewidth=0.5)
        ax.fill_between(time, lower_bounds, upper_bounds, color=colors_fill[0], alpha=0.5)

        if len(rand_signals)>0:
            
            ax.plot(time, np.mean(all_mean_signals,axis=0), color=colors[1], linewidth=0.5)
            #ax.fill_between(time, rand_lower_bounds, rand_upper_bounds, color= colors_fill[1], alpha=0.5)
            ax.scatter(time[sel_points], mean_signal[sel_points], color=colors[0], zorder=5, s=20, marker="|")
                       
            
        ax.set_ylabel(label)
        ax.set_ylim(limits)
        ax.set_xlim([-time_offset,7])
        #plt.show()
    
    return mean_signal


def plot_signals_list( time, fixed, to_compare, norm=False, limits=[[0, 16], [0, 12]], labels=['Speed (mm/s)', 'Distance (mm)'], show=True, title="", same_plot=False, save_fig=False, name_fig="", animate_line=False, axs_to_use=None, colors=None, bg_color=None, fg_color="white"):

    if colors==None:
        colors = ["k","tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    if bg_color is not None:
        colors[0] = fg_color

    if not isinstance(to_compare, list):
        to_compare = [to_compare]

    if norm:
        fixed = (fixed - min(fixed)) / (max(fixed) - min(fixed))

    if same_plot:
        if axs_to_use == None:
            fig, ax1 = plt.subplots()
            axs = [ax1]
        else:
            axs = axs_to_use
        for i in range(len(to_compare)):
            axs.append(axs[0].twinx())
    else:
        if axs_to_use == None:
            fig, axs = plt.subplots(len(to_compare) + 1, 1, sharex=True, figsize=(8, 8))
            if len(to_compare)==0:
                axs=[axs]
        else:
            axs = axs_to_use
        #axs[0].grid(True)
        axs[0].set_ylabel(labels[0], color=colors[0])
        for i, label in enumerate(labels[1:]):
            axs[i + 1].set_ylabel(label, color=colors[0])

    if bg_color is not None:
        fig.patch.set_facecolor(bg_color)
        axs[0].set_facecolor(bg_color)
        axs[0].spines["bottom"].set_color(fg_color)
        axs[0].spines["left"].set_color(fg_color)
        axs[0].tick_params(axis='x', colors=fg_color)
        axs[0].tick_params(axis='y', colors=fg_color)
        axs[0].title.set_color(fg_color)
        
    axs[0].plot(time, fixed, color=colors[0], linewidth=0.5)
    axs[0].set_title(title)
    axs[0].set_ylim(limits[0])
    axs[0].set_ylabel(labels[0], color=colors[0])
    if axs_to_use == None:
        axs[0].set_xlabel('Time (s)', color=colors[0])

    lines = []
    for i, (signal, ax_i, lim_y, color) in enumerate(zip(to_compare, axs[1:], limits[1:], colors[1:])):
        if norm:
            signal = (signal - min(signal)) / (max(signal) - min(signal))

        if same_plot:
            ax_i.spines['right'].set_position(('outward', i * 50))
            ax_i.spines['right'].set_color(color)
            ax_i.spines['top'].set_visible(False)
            ax_i.tick_params('y', colors=color)
            ax_i.set_ylabel(labels[i+1], color=color)
        else:
            ax_i.grid(True)
        
        line, = ax_i.plot(time, signal, color=color, linewidth=0.5)
        lines.append(line)
        ax_i.set_ylim(lim_y)
        if bg_color is not None:
            ax_i.set_facecolor(bg_color)
            ax_i.spines["bottom"].set_color(fg_color)
            ax_i.spines["left"].set_color(fg_color)
            ax_i.tick_params(axis='x', colors=fg_color)
            ax_i.tick_params(axis='y', colors=fg_color)

    #axs[-1].set_xlabel('Time (s)')
    #fig.tight_layout()

    frames = []
    # Initialize the vertical line if animation is enabled
    if animate_line:

        vertical_lines = [ax.axvline(x=time[0], color='r', linestyle='--') for ax in axs]

        def update(frame):
            """Update function for the animation."""
            for vertical_line in vertical_lines:
                vertical_line.set_xdata(time[frame])

            fig.canvas.draw()

            # Convert the canvas to an image and append it to the frames list
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
                
            return vertical_lines

        #ani = FuncAnimation(fig, update, frames=len(time), interval=50, blit=True)
        for i in range(len(time)):
            update(i)

    if save_fig:
        plt.savefig(name_fig, bbox_inches="tight", format="pdf")
    if show:
        plt.show()

    if animate_line:
        return frames


def find_events_from(signal, limit_values, gap_between_events, event_min_length, magnitude_factor=[2,0.8], omit_events=None, plot_signals= False, signal_name=""):

    events = []
    all_frames_above_lim = np.where((np.array(signal)>limit_values[0]) & (np.array(signal)<limit_values[1]))[0]
    if len(all_frames_above_lim) == 0:
        if plot_signals:
            print(f"Any point is between {limit_values[0]} and {limit_values[1]}")
            plt.plot(signal,label=f"{signal_name}-filtered")
            plt.legend()
            plt.show()
        return events
    distance_betw_frames = np.diff(all_frames_above_lim)
    split_points = np.where(distance_betw_frames > gap_between_events)[0]
    split_points = np.insert(split_points,0,-1)
    split_points = np.append(split_points,len(all_frames_above_lim)-1)

    if plot_signals:
        if limit_values[1] == np.inf:
            limit_value = limit_values[0]
        else:
            limit_value = limit_values[1]
        print(all_frames_above_lim[split_points])
        plt.plot(signal,label=f"{signal_name}-filtered")


    for f in range(0,len(split_points)-1):
        if split_points[f+1] - split_points[f] < 2:
            continue
        start_roi = all_frames_above_lim[split_points[f]+1]
        end_roi = all_frames_above_lim[split_points[f+1]]

        if omit_events:
            if start_roi >= omit_events[0] and start_roi < omit_events[1] and end_roi < omit_events[1]:
                continue
            elif start_roi >= omit_events[0] and start_roi < omit_events[1] and end_roi > omit_events[1]:
                start_roi = int(omit_events[1])

        duration = end_roi - start_roi

        mean_signal = np.mean(np.array(signal[start_roi:end_roi]))
        median_signal = np.median(np.array(signal[start_roi:end_roi]))
        signal_within_limits = len(np.where((np.array(signal[start_roi:end_roi])>limit_values[0]) & (np.array(signal[start_roi:end_roi])< limit_values[1]))[0])/len(np.array(signal[start_roi:end_roi]))

        if duration > event_min_length and signal_within_limits > 0.75 and np.quantile(signal[start_roi:end_roi],0.99)>=magnitude_factor[0]*limit_values[0] and np.quantile(signal[start_roi:end_roi],0.01)<=magnitude_factor[1]*limit_values[1]: 
            events.append([start_roi, end_roi, duration])
            if plot_signals:
                print(start_roi,end_roi,duration,mean_signal,median_signal,signal_within_limits)
                
                plt.plot(start_roi,limit_value,'go')
                plt.plot(end_roi,limit_value,'rx')

    if plot_signals:
        plt.plot([0,len(signal)],[limit_value, limit_value],'c-')
        plt.legend()
        plt.show()

    return events


def plot_average_dff(data,x,y,draw_connecting_lines=False,save_fig=False,name_fig=""):

    if save_fig:
        size_fig = (2.5, 2.2)
        size_markers=2
    else:
        len_data = len(data[x].unique())
        w = len_data*2
        size_fig = (w, 7)
        plt.rcParams['axes.labelsize'] = 14      # Font size for axes labels
        plt.rcParams['xtick.labelsize'] = 12     # Font size for x-axis tick labels
        plt.rcParams['ytick.labelsize'] = 12     # Font size for y-axis tick labels
        plt.rcParams['legend.fontsize'] = 12
        size_markers = 5
        
    fig, ax = plt.subplots(figsize=size_fig)

    x_names = data[x].unique()
    num_in_name = np.unique([name.split("-")[0] for name in x_names])
    sort_names = np.sort(num_in_name)
    order=["-".join([line, condition]) for line in sort_names for condition in ["iso", "gro", "iso-alone"]]
        
    hue_order = ["alone", "before", "during", "after-1h", "after-2h"]
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #fig, ax = plt.subplots()
    ax = sns.boxplot(data=data,
                     x=x,
                     y=y,
                     hue="Trial type",
                     fliersize=0,
                     notch=False,
                     order=order,
                     hue_order=hue_order,
                     ax=ax,
                     linewidth=0.5)
    
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.5))

    unique_flies = data["Fly num"].unique()
    #markers = ["o", "X", "s", "^"]
    markers = ["o", "o", "o", "o"]
    fly_to_marker = {fly: markers[i % len(markers)] for i, fly in enumerate(unique_flies)}

    for fly, marker in fly_to_marker.items():        
        ax = sns.stripplot(data=data[data["Fly num"] == fly],
                           x=x,
                           y=y,
                           hue="Trial type",
                           dodge=True,
                           jitter=True,
                           zorder=0,
                           order=order,
                           hue_order=hue_order,
                           ax=ax,
                           marker=marker,
                           size=size_markers)

    if draw_connecting_lines:
        driver_conditions = np.array(order)#data[x].unique()
        for driver in driver_conditions:
            driver_data = data[data[x] == driver]
            for exp in driver_data['Experiment'].unique():
                exp_data = driver_data[driver_data['Experiment'] == exp]
                y_vals=[]
                for t in hue_order:
                    val_dff = exp_data.loc[exp_data["Trial type"]==t, y].values
                    y_vals.append(val_dff)
                x_vals = np.where(driver_conditions == driver)[0][0]+np.linspace(-0.3,0.3,5)
                #print(x_vals)
                plt.plot(x_vals, y_vals, color='k', linestyle='-', alpha=0.5, linewidth=0.3)


    handles, labels = ax.get_legend_handles_labels()
    unique_labels = np.unique(labels)
    #len_legend = int(len(handles)/(len(unique_flies)*2))
    len_legend = len(unique_labels)
    ax.legend(handles[:len_legend], labels[:len_legend])

    if save_fig:
        plt.savefig(name_fig, bbox_inches="tight", format="pdf")
    else:
        plt.show()


def plot_dff_vs_speed(all_data, save_fig=False, path_fig=""):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "k"]
    colors_fill = ["lightblue", "moccasin", "palegreen", "lightcoral", "plum", "silver"]

    all_groups = all_data["Driver-condition"].unique()
    all_trial_types = ["alone", "before", "during", "after-1h", "after-2h"]

    num_in_name = np.unique([name.split("-")[0] for name in all_groups])
    sort_names = np.sort(num_in_name)
    order=["-".join([line, condition]) for line in sort_names for condition in ["iso", "gro", "iso-alone"]]

    #flies_to_plot={"MBON03-iso":0,
    #               "MBON03-gro":0,
    #               "MBON03-iso-alone":0,
    #               "MBON04-iso":0,
    #               "MBON04-gro":2,
    #               "MBON04-iso-alone":2,
    #               "MBON22-iso":2,
    #               "MBON22-gro":0,
    #               "MBON22-iso-alone":2}

    for num_group, group in enumerate(order):

        widths = [1.0, 1.0, 1.0]

        fig = plt.figure(figsize=(6.5, 5))
        gs = gridspec.GridSpec(5, 3, figure=fig, width_ratios=widths)

        axs = np.empty((5, 3), dtype=object)
        
        for i in range(5):
            for j in range(3):
                axs[i, j] = fig.add_subplot(gs[i, j])
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)

        for num_trial, type_trial in enumerate(all_trial_types):
            print(group, type_trial)
            subset = all_data.loc[(all_data["Driver-condition"]==group)&
                                  (all_data["Trial type"]==type_trial)]
            trial_names = subset["Trial"].unique()
            
            for num_fly, name in enumerate(trial_names):                
                time = subset.loc[subset["Trial"]==name]["Time 2p"].values[0]
                filtered_vel = subset.loc[subset["Trial"]==name]["Filtered vel"].values[0]
                filtered_dff = subset.loc[subset["Trial"]==name]["Filtered dff"].values[0]
                if "MBON03" in group:
                    plot_limits_trial = [[0,16.5],[0,250]]
                else:
                    plot_limits_trial = [[0,16.5],[0,100]]

                plot_signals_list(time, filtered_vel, [filtered_dff], limits=plot_limits_trial, labels=["Speed (mm/s)","dff (%)"], show=False, same_plot=True, axs_to_use=[axs[num_trial][num_fly]], colors=["k",colors[num_trial]])
                if num_trial == len(all_trial_types)-1:
                    axs[num_trial][num_fly].tick_params(axis='x', labelbottom=True)
                else:
                    axs[num_trial][num_fly].tick_params(axis='x', labelbottom=False)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.65, hspace=0.15)
        
        if save_fig:
            name_fig=os.path.join(path_fig,f"{group}_traces.pdf")
            plt.savefig(name_fig, bbox_inches="tight", format="pdf")

    if not save_fig:
        plt.show()


def plot_dff_responses(all_data, save_fig=False, path_fig=""):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "k"]
    colors_fill = ["lightblue", "moccasin", "palegreen", "lightcoral", "plum", "silver"]

    all_groups = all_data["Driver-condition"].unique()
    all_trial_types = ["alone", "before", "during", "after-1h", "after-2h"]

    num_in_name = np.unique([name.split("-")[0] for name in all_groups])
    sort_names = np.sort(num_in_name)
    order=["-".join([line, condition]) for line in sort_names for condition in ["iso", "gro", "iso-alone"]]

    #flies_to_plot={"MBON03-iso":0,
    #               "MBON03-gro":0,
    #               "MBON03-iso-alone":0,
    #               "MBON04-iso":0,
    #               "MBON04-gro":2,
    #               "MBON04-iso-alone":2,
    #               "MBON22-iso":2,
    #               "MBON22-gro":0,
    #               "MBON22-iso-alone":2}

    for num_group, group in enumerate(all_groups):

        widths = [1.0, 1.0, 1.0, 1.0]

        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 4, figure=fig, width_ratios=widths)

        axs = np.empty((5, 4), dtype=object)
        
        for i in range(5):
            for j in range(4):
                axs[i, j] = fig.add_subplot(gs[i, j])
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)

        for num_trial, type_trial in enumerate(all_trial_types):
            print(group, type_trial)
            subset = all_data.loc[(all_data["Driver-condition"]==group)&
                                  (all_data["Trial type"]==type_trial)]
            trial_names = subset["Trial"].unique()
            vels=[]
            dffs_from_vel=[]
            dists=[]
            dffs_from_dist=[]
            
            dffs_rand=[]
            dists_rand=[]

            rat_vels=[]
            
            for num_fly, name in enumerate(trial_names):
                #if flies_to_plot[group] == num_fly:
                #    time = subset.loc[subset["Trial"]==name]["Time 2p"].values[0]
                #    filtered_vel = subset.loc[subset["Trial"]==name]["Filtered vel"].values[0]
                #    filtered_dff = subset.loc[subset["Trial"]==name]["Filtered dff"].values[0]
                #    if "MBON03" in group:
                #        plot_limits_trial = [[0,16.5],[0,250]]
                #        plot_limits = [[0,16.5],[-20,50],[1.5,11],[-60,60]]
                #    else:
                #        plot_limits_trial = [[0,16.5],[0,100]]
                #        plot_limits = [[0,16.5],[-20,25],[1.5,11],[-30,30]]
                        
                #    plot_signals_list(time, filtered_vel, [filtered_dff], limits=plot_limits_trial, labels=["Speed (mm/s)","dff (%)"], show=True, same_plot=True, axs_to_use=[axs[num_trial][0]], colors=["k",colors[num_trial]])

                if "MBON03" in group:
                    plot_limits = [[0,16.5],[-20,50],[1.5,11],[-60,60]]
                else:
                    plot_limits = [[0,16.5],[-20,25],[1.5,11],[-30,30]]
                    
                vel = subset.loc[subset["Trial"]==name]["Traces vel"].values
                vels.extend(vel[0])
                dff_from_vel = subset.loc[subset["Trial"]==name]["Traces dff from vel"].values
                dffs_from_vel.extend(dff_from_vel[0])
                
                dff_rand = subset.loc[subset["Trial"]==name]["Traces vel rand"].values
                dffs_rand.extend(dff_rand[0])

                dist = subset.loc[subset["Trial"]==name]["Traces dist"].values
                if len(dist[0])>0:
                    dists.extend(dist[0])

                    dff_from_dist = subset.loc[subset["Trial"]==name]["Traces dff from dist"].values
                    dffs_from_dist.extend(dff_from_dist[0])
                    
                    dist_rand = subset.loc[subset["Trial"]==name]["Traces dist rand"].values
                    dists_rand.extend(dist_rand[0])

                    rat_vel = subset.loc[subset["Trial"]==name]["Ratio vels traces"].values
                    rat_vels.extend(rat_vel[0])
                    
            mean_trace_vel = get_mean_signal(vels, plot_all=True, plot=True, ax=axs[num_trial][0], limits=plot_limits[0], label="Speed (mm/s)", colors=[colors[-1]], colors_fill=[colors_fill[-1]])           
            mean_trace_dff = get_mean_signal(dffs_from_vel, rand_signals=dffs_rand, plot=True, ax=axs[num_trial][1], limits=plot_limits[1], label="dff (%)", colors=[colors[num_trial],colors[-1]], colors_fill=[colors_fill[num_trial],colors_fill[-1]])

            if len(dists)>0:
                mean_trace_dist = get_mean_signal(dists, plot_all=True, plot=True, ax=axs[num_trial][2], limits=plot_limits[2], label="Distance (mm)", colors=[colors[-1]], colors_fill=[colors_fill[-1]])
                mean_trace_dff_dist = get_mean_signal(dffs_from_dist, rand_signals=dists_rand, plot=True, ax=axs[num_trial][3], limits=plot_limits[3], label="dff (%)", colors=[colors[num_trial],colors[-1]], colors_fill=[colors_fill[num_trial],colors_fill[-1]])

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.5, hspace=0.15)
        
        
        if save_fig:
            name_fig=os.path.join(path_fig,f"{group}_responses.pdf")
            plt.savefig(name_fig, bbox_inches="tight", format="pdf")
        
    plt.show()


def plot_proximity_events_reactions(all_data, save_fig=False, name_fig=""):
    for key in ["Ratio vels", "Num events"]:
        fig, ax = plt.subplots()
        ax = sns.boxplot(data=all_data,
                         x="Fly condition",
                         y=key,
                         hue="Trial type",
                         fliersize=0,
                         notch=False,
                         hue_order=["during", "after-2h"],
                         ax=ax)

        for patch in ax.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.5))

        ax = sns.stripplot(data=all_data,
                           x="Fly condition",
                           y=key,
                           hue="Trial type",
                           dodge=True,
                           jitter=True,
                           zorder=0,
                           hue_order=["during", "after-2h"],
                           ax=ax)

        #driver_conditions = all_data['Fly condition'].unique()
        #for driver in driver_conditions:
        #    driver_data = all_data[all_data['Fly condition'] == driver]
        #    for exp in driver_data['Experiment'].unique():
        #        exp_data = driver_data[driver_data['Experiment'] == exp]
        #        x_vals = np.where(driver_conditions == driver)[0][0]+np.linspace(-0.3,0.3,3)
        #        #print(x_vals)
        #        plt.plot(x_vals, exp_data[key], color='k', linestyle='-', alpha=0.5)


        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if key=="Ratio vels":
            plt.yscale('log')
        #plt.ylim([0.3, 1.8])
        handles, labels = ax.get_legend_handles_labels()
        len_legend = int(len(handles)/2)
        ax.legend(handles[len_legend:], labels[len_legend:])
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(name_fig, bbox_inches="tight", format="pdf")

    all_data["class"]=all_data["Fly condition"]+"-"+all_data["Trial type"] 
    #all_data.to_pickle(output_file_name)

    plt.show()


def align_events(events, filtered_signal, filtered_dff, time_gap, fps, remove_baseline=True):
    traces_signal = []
    traces_dff = []
    adjusted_signal = []
    adjusted_dff = []
    adjusted_dist = []
    for e in events:
        start_event = e[0] - time_gap*fps
        stop_event = e[1] #+ 5*fps
        
        if start_event>0 and stop_event+time_gap*fps < len(filtered_dff):
            trace_signal = filtered_signal[start_event:stop_event]                       
            trace_dff = filtered_dff[start_event:stop_event]

            traces_signal.append(trace_signal)
            traces_dff.append(trace_dff)            

    if remove_baseline:
        baseline_region_signal = [trace[:time_gap*fps] for trace in traces_signal]
        baseline_signal = np.mean(baseline_region_signal)
        adjusted_signal = [trace - baseline_signal for trace in traces_signal]
    else:
        adjusted_signal = traces_signal

    baseline_region_dff = [trace[:time_gap*fps] for trace in traces_dff]
    baseline_dff = np.mean(baseline_region_dff)
    adjusted_dff = [trace - baseline_dff for trace in traces_dff]
    
    return adjusted_signal, adjusted_dff


def get_random_events(events, filtered_dff, num_events, time_gap, fps):
    all_random_dff=[]
    for i in range(num_events):
        random_stops = random.sample(range(time_gap*fps,len(filtered_dff)), len(events))
        random_events = [[random_stops[i]-duration, random_stops[i]] for i, (_, _, duration) in enumerate(events)]
        traces_dff_rand = []
        traces_dist_rand = []
        for e in random_events:
            start_event_rand = e[0] - time_gap*fps
            stop_event_rand = e[1] #+ 5*fps
            if start_event_rand>0 and stop_event_rand+fps < len(filtered_dff):
                trace_dff_rand = filtered_dff[start_event_rand:stop_event_rand]
                traces_dff_rand.append(trace_dff_rand)

        baseline_region_dff = [trace[:time_gap*fps] for trace in traces_dff_rand]
        baseline_dff_rand = np.mean(baseline_region_dff)
        adjusted_dff_rand = [trace - baseline_dff_rand for trace in traces_dff_rand]

        all_random_dff.append(adjusted_dff_rand)

    return all_random_dff


def lowpass_filter(signal):
     fps = 8
     cutoff_frequency = 0.25  # Cutoff frequency in Hz
     order = 2  # Order of the Butterworth filter
     nyquist_frequency = fps / 2
     normal_cutoff = cutoff_frequency / nyquist_frequency

     b2, a2 = butter(order, normal_cutoff, btype='low', analog=False)
     filtered_signal = filtfilt(b2, a2, signal)

     return filtered_signal


def get_mean_dff_trace(dff_masks, dff, num_clusters=1):
    
    mean_dff_per_frame = []
    all_areas = []

    for i, (mask, frame) in enumerate(zip(dff_masks, dff)):        

        dff_masked = cv2.bitwise_and(frame, frame, mask=mask)

        coordinates = np.column_stack(np.where(dff_masked > 0))

        pixel_values = dff_masked[coordinates[:, 0], coordinates[:, 1]]

        if len(pixel_values)>0: 
            mean_dff_per_frame.append(np.mean(pixel_values))
        else:
            mean_dff_per_frame.append(mean_dff_per_frame[-1])
         
        
    filtered_dff = lowpass_filter(mean_dff_per_frame)
    
    return filtered_dff

def get_mean_dff_baseline(filtered_dff, filtered_dff_wBaseline, bin_len=240, fps=8, from_baseline=True):
    baseline_th = 0.01#np.quantile(filtered_dff,0.95)*0.05
    #print(baseline_th)
    mean_per_bin=[]
    #plt.plot(filtered_dff_wBaseline)
    for init in range(0, len(filtered_dff), bin_len*fps):
        if from_baseline:
            events_dff = find_events_from(filtered_dff[init:init+bin_len*fps],[-np.inf,baseline_th],fps,fps/2,plot_signals=False)
        else:
            events_dff = find_events_from(filtered_dff[init:init+bin_len*fps],[baseline_th,np.inf],fps,fps/2,plot_signals=False)
        all_mean_baseline = []
        #print(init)
        for start, stop, _ in events_dff:
            #print(start,stop)
            if from_baseline:
                mean_baseline = np.mean(filtered_dff_wBaseline[init+start:init+stop])
            else:
                mean_baseline = np.mean(filtered_dff[init+start:init+stop])
            all_mean_baseline.append(mean_baseline)
            #plt.plot(init+start,filtered_dff_wBaseline[init+start],'go')
            #plt.plot(init+stop,filtered_dff_wBaseline[init+stop],'rx')
        mean_per_bin.append(np.mean(all_mean_baseline))
    #plt.show()
    return mean_per_bin
