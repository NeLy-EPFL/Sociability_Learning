import os
import sys
import utils2p
import numpy as np
import pandas as pd
from pathlib import Path
from pybaselines import Baseline
from scipy.signal import savgol_filter
from Sociability_Learning.utils_files import *
from Sociability_Learning.twop_setup.utils_analysis import *
from Sociability_Learning.twop_setup.utils_fictrac import filter_fictrac


output_file_name = "results/test.pkl"

adaptation_time = 60 #s

bin_len = 240 #s
mean_from_baseline=True

def main():
    all_data = pd.DataFrame()

    usr_input = sys.argv[-1]
    folders = get_folders_from_file(usr_input)

    fly_count={}
    for i, folder in enumerate(folders):
        all_experiments = get_experiments(folder, keyword="processed")
        for folder_path in all_experiments:
            trial = str(Path(folder_path).parent)
            fly_dir = Path(folder_path).parent.parent
            exp_processed = os.path.join(fly_dir, "processed")

            print(trial)

            trial_type = trial.split("_")[-1]
            fly_condition = trial.split("/")[-2][5:]

            #trial_name = "-".join([fly_condition, trial_type])
            
            beh_sync = os.path.join(folder_path,"beh_df.pkl")
            twop_sync = os.path.join(folder_path,"twop_df.pkl")
            dff_path = os.path.join(folder_path,"dff.tif")
            mask_path = os.path.join(folder_path,"dff_mask.tif")
            baseline_path = os.path.join(exp_processed,"dff_baseline.tif")
            green_path = os.path.join(folder_path,"green_com_denoised.tif")
            red_path = os.path.join(folder_path,"red_com_warped.tif")
        
            beh_df = pd.read_pickle(beh_sync)
            twop_df = pd.read_pickle(twop_sync)

            roi_df = beh_df.loc[beh_df["t"] >= adaptation_time]
            time_beh_df = roi_df["t"]
            time_beh = time_beh_df.values
            first_index_beh = time_beh_df.index.get_level_values("Frame")[0]
            last_index_beh = time_beh_df.index.get_level_values("Frame")[-1]+1
            
            time_2p_df = twop_df.loc[(twop_df["t"] >= adaptation_time)&(twop_df["t"] <= np.max(time_beh))]["t"]
            time_2p = time_2p_df.values
            first_index_2p = time_2p_df.index.get_level_values("Frame")[0]
            last_index_2p = time_2p_df.index.get_level_values("Frame")[-1]+1
            
            dff = utils2p.load_img(dff_path)[first_index_2p:last_index_2p]
            baseline_dff = utils2p.load_img(baseline_path)
            dff_masks = utils2p.load_img(mask_path)[first_index_2p:last_index_2p]
            
            
            forward_vel = filter_fictrac(beh_df["delta_rot_lab_forward"].values[first_index_beh:last_index_beh])
            side_vel = filter_fictrac(beh_df["delta_rot_lab_side"].values[first_index_beh:last_index_beh])
            heading_vel = filter_fictrac(beh_df["delta_rot_lab_turn"].values[first_index_beh:last_index_beh])

            forward_pos = filter_fictrac(beh_df["integrated_forward_movement"].values[first_index_beh:last_index_beh])
            side_pos = filter_fictrac(beh_df["integrated_side_movement"].values[first_index_beh:last_index_beh])
            heading_pos = filter_fictrac(beh_df["integrated_lab_heading"].values[first_index_beh:last_index_beh])
            
            vel_teth = beh_df["v"].values[first_index_beh:last_index_beh]
            th_teth = beh_df["th"].values[first_index_beh:last_index_beh]
            
            resamp_vel = np.interp(time_2p, time_beh, vel_teth)
            resamp_forward = np.interp(time_2p, time_beh, forward_vel)
            resamp_side = np.interp(time_2p, time_beh, side_vel)
            resamp_heading = np.interp(time_2p, time_beh, heading_vel)


            no_free_tracking = False
            try:                
                free_fly_tracking = pd.read_pickle(os.path.join(trial, "behData", "images", "free_fly_tracking.pickle"))
                free_fly_tracking_clean = [filter_fictrac(pos[first_index_beh:last_index_beh]) for pos in free_fly_tracking]

                free_fly_coords = list(zip(free_fly_tracking_clean[0], free_fly_tracking_clean[1]))
                d_free = np.linalg.norm(free_fly_coords, axis=1)               
                x_free = free_fly_tracking_clean[0]
                y_free = free_fly_tracking_clean[1]

                resamp_d_free = np.interp(time_2p, time_beh, d_free)
                resamp_x_free = np.interp(time_2p, time_beh, x_free)
                resamp_y_free = np.interp(time_2p, time_beh, y_free)
                
            except Exception as e:
                print("Exception:", e)
                no_free_tracking = True                
            
            
            filtered_dff = get_mean_dff_trace(dff_masks, dff)
            
            if not no_free_tracking:
                filtered_dist = lowpass_filter(resamp_d_free)
                
            filtered_vel = lowpass_filter(resamp_vel)
                
            baseline_fitter = Baseline(x_data=time_2p)
            baseline_vel, params_vel = baseline_fitter.asls(filtered_vel, lam=1e7, p=0.05)
            filtered_vel -= baseline_vel
            filtered_vel = np.clip(filtered_vel, 0, None)
            
            baseline_dff, params_dff = baseline_fitter.asls(filtered_dff, lam=1e6, p=0.005)
            filtered_dff_wBaseline = filtered_dff.copy()
            filtered_dff -= baseline_dff
            filtered_dff = np.clip(filtered_dff, 0, None)

            mean_baseline = get_mean_dff_baseline(filtered_dff, filtered_dff_wBaseline, bin_len=bin_len, from_baseline=mean_from_baseline)
            mean_dff=mean_baseline

            time_gap = 1 #s
            fps = 8
            events_vel = find_events_from(filtered_vel,[1,np.inf],fps*1.5,fps,plot_signals=False)
            adjusted_vel, adjusted_dff_vel = align_events(events_vel, filtered_vel, filtered_dff, time_gap, fps)

            if not no_free_tracking:
                events_dist = find_events_from(filtered_dist,[0,6],time_gap*fps,fps/2,plot_signals=False)
                adjusted_dist, adjusted_dff_dist = align_events(events_dist, filtered_dist, filtered_dff, time_gap, fps, remove_baseline=False)
            else:
                adjusted_dist = []
                adjusted_dff_dist = []

                
            num_random_events = 33
            all_random_dff_vel = get_random_events(events_vel, filtered_dff, num_random_events, time_gap, fps)
            if not no_free_tracking:
                all_random_dff_dist = get_random_events(events_dist, filtered_dff, num_random_events, time_gap, fps)
            else:
                all_random_dff_dist = []


            #### For proximity events            
            all_ratios = []
            traces_vel_ratio = []
            ratio_vels = np.nan
            num_events=0
            if not no_free_tracking and not "1h" in trial:
            #if not no_free_tracking:
                time_gap = 2.5 #s
                events = find_events_from(filtered_dist,[0,6],time_gap*fps,fps/2,plot_signals=False)

                for e in events:
                    start_event = int(e[0] - time_gap*fps)
                    stop_event = int(e[0] + time_gap*fps)
                    
                    if start_event>0 and stop_event < len(filtered_vel):
                        
                        num_events+=1
                        trace_vel = filtered_vel[start_event:stop_event]
                        trace_dist = filtered_dist[start_event:stop_event]
                        trace_acc = savgol_filter(trace_vel, 5, 3, deriv=1, delta=1/fps)
                        
                        before = abs(trace_acc[:int(time_gap*fps)])
                        after = abs(trace_acc[int(time_gap*fps):])

                        mean_before = np.mean(before)
                        mean_after = np.mean(after)
                        std_before = np.std(before)
                        std_after = np.std(after)

                        if mean_before-std_before <= mean_after:
                           
                            traces_vel_ratio.append(abs(trace_acc))
                            if mean_before == 0 and mean_after == 0:
                                ratio_means = 1
                            elif mean_before < 0.1 and mean_after != 0:
                                ratio_means = mean_after/0.1 
                            else:
                                ratio_means = mean_after/mean_before
                            
                            all_ratios.append(ratio_means)

                if len(all_ratios)>2:
                    ratio_vels = np.median(all_ratios)
            
            genotype = trial.split("/")[-3].split("_")[1]
            gal4 = genotype.split("-")[0]
            name_trial = trial.split("/")[-1].split("_")[1]
            driver_condition = "-".join([gal4,fly_condition])
            if driver_condition in fly_count.keys():
                fly_count[driver_condition] += 1
            else:
                fly_count[driver_condition] = 1
            #plt.plot(time_2p, average_dff, label=name_trial)
            #print(fly_condition, name_trial, mean_dff)
            for mean_val in mean_dff:
                trial_df = pd.DataFrame({"Experiment": [fly_dir],
                                         "Trial": [trial],
                                         "Genotype": [genotype],
                                         "Driver": [gal4],
                                         "Fly condition": [fly_condition],
                                         "Driver-condition": [driver_condition],
                                         "Fly num": [fly_count[driver_condition]],
                                         "Trial type": [name_trial],
                                         "Mean dff": [mean_val],
                                         "Time 2p": [time_2p],
                                         "Filtered dff": [filtered_dff_wBaseline],
                                         "Filtered vel": [filtered_vel],
                                         "Traces vel": [adjusted_vel],
                                         "Traces dff from vel": [adjusted_dff_vel],
                                         "Traces dist": [adjusted_dist],
                                         "Traces dff from dist": [adjusted_dff_dist],
                                         "Traces vel rand": [all_random_dff_vel],
                                         "Traces dist rand": [all_random_dff_dist],
                                         "Num events": [num_events],
                                         "Ratio vels traces":[traces_vel_ratio],
                                         "Ratio vels": [ratio_vels],
                                         "All ratios": [all_ratios],
                                         })
                all_data = all_data.append(trial_df, ignore_index=True)
            
    all_data.to_pickle(output_file_name)

                

if __name__ == "__main__":
    main()
