import os
import h5py
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import savgol_filter, butter, sosfilt

import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def get_distance_between(pos0, pos1, pixel_size=32/832, in_pixels=False):
    if len(pos0.shape) > 1:
        dist_px = np.linalg.norm(pos0 - pos1, axis=1)
    else:
        dist_px = np.linalg.norm(pos0 - pos1)

    if in_pixels:
        return dist_px
    
    dist_mm = dist_px * pixel_size

    return dist_mm

def get_orientation(start_point, end_point):
    theta = np.arctan2(end_point[:,1] - start_point[:,1], end_point[:,0] - start_point[:,0])
    
    return -theta # Minus added because "y" increases from top to bottom in the images

def get_kinematics(locations,
                   fps,
                   orientation=[],
                   nodes=[],
                   linear_velocity = True,
                   angular_velocity = True,
                   linear_acceleration = True,
                   angular_acceleration = True,
                   movement_direction = True,
                   absolute_values = False,
                   win=25,
                   poly=3,
                   pixel_size=32/832):
    """
        
    """
    kinematics = {}

    if len(nodes)==0 and isinstance(locations, dict):
        nodes = list(locations.keys())

    if len(nodes)==0 and isinstance(locations, list):
        nodes = ["data"]
        angular_velocity = False
        linear_acceleration = False
        angular_acceleration = False
        movement_direction = False
        shape_list = np.array(locations).shape
        if len(shape_list) == 1:
            node_loc = np.array(locations).reshape(len(locations), 1)
        else:
            node_loc = np.array(locations)

        
    for node in nodes:
        kinematics[node] = {}
        if node is not "data":
            node_loc = locations[node]["locations"] 
            
        if not absolute_values:
            if angular_velocity:
                kinematics[node]["ang_vel"] = savgol_filter(orientation, win, poly, deriv=1, delta=1/fps)

            if angular_acceleration:
                kinematics[node]["ang_acc"] = savgol_filter(orientation, win, poly, deriv=2, delta=1/fps)

            if linear_velocity or movement_direction or linear_acceleration:

                node_loc_vel = np.zeros_like(node_loc)
                for c in range(node_loc.shape[-1]):
                    node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1, delta=1/fps)

                if linear_acceleration:
                    node_loc_acc = np.zeros_like(node_loc)
                    for c in range(node_loc.shape[-1]):
                        node_loc_acc[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=2, delta=1/fps)
                if node is "data":
                    linear_vel = np.linalg.norm(node_loc_vel,axis=1) * pixel_size
                    return linear_vel

                movement_dir = -np.arctan2(node_loc_vel[:, 1],node_loc_vel[:, 0])

                #ratio_theta = np.cos(orientation - theta_pos)        
                #dir_vel = ratio_theta/np.abs(ratio_theta)

                dir_mov = [1 if np.abs(th_pos - th_vel) <= np.pi/2 else -1 for th_pos, th_vel in zip(orientation, movement_dir)]
                linear_vel = np.linalg.norm(node_loc_vel,axis=1) * pixel_size * dir_mov
                movement_dir[np.abs(linear_vel) < 2] = orientation[np.abs(linear_vel) < 2]

                linear_acc = np.linalg.norm(node_loc_acc,axis=1) * pixel_size * dir_mov

                if linear_velocity:
                    kinematics[node]["lin_vel"] = linear_vel
                if movement_direction:
                    kinematics[node]["mov_dir"] = movement_dir
                if linear_acceleration:
                    kinematics[node]["lin_acc"] = linear_acc        

        

    #plt.figure()
    #plt.plot(kinematics["ang_vel"])
    #plt.figure()
    #plt.plot(kinematics["ang_acc"])
    #plt.figure()
    #plt.plot(kinematics["lin_vel"])
    #plt.figure()
    #plt.plot(kinematics["lin_acc"])
    #plt.figure()
    #plt.plot(np.abs(orientation-kinematics["mov_dir"])*180/np.pi)
    #plt.show()
        
    return kinematics 


def flies_closer_than(threshold, locations, start_frame, end_frame, get_nodes_info=False, beetle=False):

    if beetle:
        landmarks = ["thorax"]
        #threshold *= 2
    else:
        landmarks = ["thorax","head","abdomen"]#"Lwing","Rwing"]#list(locations[flies[0]].keys())#
    flies = list(locations.keys())
    duration = end_frame - start_frame
    
    nodes_info = [[[],[],[]] for _ in range(duration)]

    for node1 in landmarks:
        locations_node_fly0 = locations[flies[0]][node1]["locations"][start_frame:end_frame] 
        scores_node_fly0 = locations[flies[0]][node1]["scores"][start_frame:end_frame]
        bad_tracking_fly0 = np.where(scores_node_fly0<0.1)[0]
        for node2 in landmarks:
            locations_node_fly1 = locations[flies[1]][node2]["locations"][start_frame:end_frame]
            scores_node_fly1 = locations[flies[1]][node2]["scores"][start_frame:end_frame]
            bad_tracking_fly1 = np.where(scores_node_fly1<0.1)[0]

            dist_betw=get_distance_between(locations_node_fly0,
                                           locations_node_fly1)
            frames_included = np.where(dist_betw<threshold)[0]
            
            frames_included_clean=[]
            for frame in frames_included:
                if frame not in bad_tracking_fly0 and frame not in bad_tracking_fly1:
                    frames_included_clean.append(frame)
            
            if len(frames_included_clean)>0:   
                for frame in frames_included_clean:
                    nodes_info[frame][0].append(node1)
                    nodes_info[frame][1].append(node2)
                    nodes_info[frame][2].append(dist_betw[frame])

    below_threshold = [any(node) for node in nodes_info]
    if get_nodes_info:
        return below_threshold, nodes_info
    else:
        return below_threshold


def find_events_from(signal, limit_values, gap_between_events, event_min_length, omit_events=None, plot_signals= False, signal_name=""):

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

        if duration > event_min_length and signal_within_limits > 0.75: 
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

def find_proximity_events(nodes_in_proximity, dist_limit, gap_between_events, event_min_length, event_max_length=800, omit_events=None, nodes_info=None):
    prox_events = []
    all_events = find_events_from(nodes_in_proximity, [0,np.inf], gap_between_events, event_min_length, omit_events)
    
    if nodes_info and len(all_events) > 0:
        dist_betw = np.array(np.array(nodes_info,dtype=object).T[2])
        mean_dist = [np.mean(d) if any(d) else dist_limit for d in dist_betw]
    else:
        return prox_events
        

    for [start_roi, end_roi, duration] in all_events:        
        mean_dist_roi = mean_dist[start_roi:end_roi]
        min_dist = min(mean_dist_roi)
        ind_min_dist = np.nanargmin(mean_dist_roi)


        #if start_roi == 11960 and end_roi == 12071:
        #    print(dist_betw[start_roi:end_roi])
        #    print(mean_dist_roi)
        #    print(min_dist)
        #    print(ind_min_dist)
        #    input("")
        
        
        if min_dist < 0.8*dist_limit:
            if duration > event_max_length:
                if ind_min_dist/duration > 0.5:
                    dist_to_end =  duration - (ind_min_dist + event_max_length/2)
                    end_roi -= int(dist_to_end) if dist_to_end > 0 else 0
                    start_roi = end_roi - event_max_length
                    ind_min_dist = int(event_max_length/2) if dist_to_end > 0 else event_max_length - (duration-ind_min_dist)
                else:
                    start_roi += int(ind_min_dist - event_max_length/2) if event_max_length/2 < ind_min_dist else 0
                    end_roi = start_roi + event_max_length
                    ind_min_dist = int(event_max_length/2) if event_max_length/2 < ind_min_dist else ind_min_dist 
                duration = end_roi - start_roi
                mean_dist_roi = mean_dist[start_roi:end_roi]
            
            prox_events.append([start_roi, end_roi, duration, min_dist, ind_min_dist, mean_dist_roi])

    return prox_events

def get_proximity_events(experiment, locations, fps, dist_limit=5, gap_between_events=20, event_min_length=40, event_max_length=800, omit_events=None, save_pkl=False, from_file=False, beetle=False):

    if from_file:
        try:
            file_name = os.path.join(experiment.folder, "proximity_events.pkl") 
            proximity_events = pd.read_pickle(file_name)
            return proximity_events
        except:
            print(f"{file_name} not found, computing proximity events")
            pass
        
        
    proximity_events = pd.DataFrame()
    
    num_frames = len(locations["fly0"]["thorax"]["locations"])
    
    nodes_in_proximity, nodes_info = flies_closer_than(dist_limit,
                                                       locations,
                                                       0,
                                                       num_frames,
                                                       get_nodes_info=True,
                                                       beetle=beetle)
    
    all_events = find_proximity_events(nodes_in_proximity,
                                       dist_limit,
                                       gap_between_events,
                                       event_min_length,
                                       omit_events=omit_events,
                                       event_max_length=event_max_length,
                                       nodes_info=nodes_info)

    if len(all_events)>0:
        start_frames = np.array(all_events,dtype=object).T[0]
        stop_frames = np.array(all_events,dtype=object).T[1]
        duration = np.array(all_events,dtype=object).T[2]/fps
        min_dist = np.array(all_events,dtype=object).T[3]
        ind_min_dist = np.array(all_events,dtype=object).T[4]
        dist_between = np.array(all_events,dtype=object).T[5]
    else:
        return proximity_events
        #start_frames = [0]
        #stop_frames = [0]
        #duration = [0]

    if int(experiment.time) < 120000:
        light_box = "Morning"
    elif int(experiment.time) > 170000:
        light_box = "Evening"
    else:
        light_box = "Afternoon"    
    

    for fly_num, fly in enumerate(locations.keys()):
        if "fly" in fly:
            fly_events = pd.DataFrame({"Folder": experiment.folder,
                                       "Video name": experiment.vidName,
                                       "Gal4": experiment.gen1,
                                       "UAS": experiment.gen2,
                                       "Date": experiment.date,
                                       "Time": experiment.time,
                                       "LightBox": light_box,
                                       "Arena": experiment.arena,
                                       "Start": start_frames,
                                       "Stop": stop_frames,
                                       "Duration": duration,
                                       "Dist between": dist_between,
                                       "Min distance": min_dist,
                                       "Ind min dist":ind_min_dist,
                                       "Fly": fly_num,
                                       "Distancing direction": 0.0,
                                       "AUC":0.0,
                                       "pre":-1,
                                       "post":-1})
            
        proximity_events = proximity_events.append(fly_events,ignore_index=True)

    if save_pkl:
        file_name = os.path.join(experiment.folder, "proximity_events.pkl") 
        proximity_events.to_pickle(file_name)  

    return proximity_events


def get_reaction_classes(events,
                         kinematics,
                         node="thorax"):
    #save_fig=False
    #cont=0
    for i, event in events.iterrows():
        fly = f"fly{event['Fly']}"
        start_frame = event["Start"]
        stop_frame = event["Stop"]
        closest_frame = event["Ind min dist"]

        abs_velocity = np.abs(kinematics[fly][node]["lin_vel"][start_frame:stop_frame])

        abs_acc = np.abs(kinematics[fly][node]["lin_acc"][start_frame:stop_frame])

        time_aligned = np.linspace(-closest_frame, len(abs_acc)-closest_frame, len(abs_acc))

        #print(i, start_frame, stop_frame, closest_frame, np.mean(abs_velocity[closest_frame:]))

        pre_class = 0 if np.mean(abs_velocity[:closest_frame]) < 2 else 1
        post_class = 0 if np.mean(abs_velocity[closest_frame:]) < 2 else 1

        events.at[i,"pre"] = pre_class 
        events.at[i,"post"] = post_class
        

def get_metrics(proximity_events, kinematics, locations, control_thresholds, gen_name, fly, fps, total_exp_time, exp_folder, bins=1, adaptation_time=120, print_info=True, return_events="moving", save_events=False):

    prop_df = pd.DataFrame()
    sel_events = pd.DataFrame()
    distancing_limits = control_thresholds[0]
    standstill_th = control_thresholds[1]
    moving_th = control_thresholds[2]
    bin_size = int(np.ceil(total_exp_time*fps/bins))
    bins_ideal = [bin_size*i for i in range(1,bins)]
    bins_start = [int(adaptation_time*fps)]

    for b in bins_ideal:
        if b > bins_start[-1]:
            bins_start.append(b)
    bins_start.append(int(np.ceil(total_exp_time*fps)))
    
    for i, b in enumerate(bins_start[:-1]):
        bin_name = f"{int(np.ceil(b/(fps*60)))}min-{int(np.ceil(bins_start[i+1]/(fps*60)))}min"
        print(bin_name)
        standstill_events = proximity_events.loc[(proximity_events["Fly"] == fly)&
                                                 (proximity_events["post"] == 0)&
                                                 (proximity_events["Start"] >= b)&
                                                 (proximity_events["Stop"] < bins_start[i+1])]

        moving_events = proximity_events.loc[(proximity_events["Fly"] == fly)&
                                             (proximity_events["post"] == 1)&
                                             (proximity_events["Start"] >= b)&
                                             (proximity_events["Stop"] < bins_start[i+1])]

        total_events = len(standstill_events) + len(moving_events)

        if total_events < 3:
            continue

        if print_info:
            print(f"Fly {fly} is standstill in {len(standstill_events)}/{int(total_events)} and moving in {len(moving_events)}/{int(total_events)} events.")


        get_immobile_freezing(standstill_events, kinematics, locations, fps, distancing_limits)
        standstill_prop = len(np.where(standstill_events["AUC"]<standstill_th)[0])/(total_events)
        standstill_events = standstill_events.copy()
        standstill_events["Control threshold"] = standstill_th

       
        
        get_distancing_efficiency(moving_events, kinematics, locations, fps, distancing_limits)
        moving_prop = len(np.where(moving_events["AUC"]<moving_th)[0])/(total_events)
        moving_events = moving_events.copy()
        moving_events["Control threshold"] = moving_th
        
        
        sociability_prop = standstill_prop + moving_prop

        if print_info:
            print(f"MEDIAN = {sociability_prop}")

        fly_values = pd.DataFrame({"Experiment": exp_folder,
                                   "Genotype": gen_name,
                                   "Fly": fly,
                                   "Bin": bin_name,
                                   "Median velocity":np.median(kinematics[f"fly{fly}"]["thorax"]["lin_vel"][b:bins_start[i+1]]),
                                   "Standstill": standstill_prop,
                                   "Moving": moving_prop,
                                   "Prop":sociability_prop}, index=[0])
        
        prop_df = prop_df.append(fly_values, ignore_index=True)

        if return_events=="moving":
            sel_events = sel_events.append(moving_events.copy(), ignore_index=True)
        if return_events=="stand":
            sel_events = sel_events.append(standstill_events.copy(), ignore_index=True)
        if return_events=="both":
            sel_events = sel_events.append(moving_events.copy(), ignore_index=True)
            sel_events = sel_events.append(standstill_events.copy(), ignore_index=True)

        if save_events:
            file_name = os.path.join(exp_folder, f"{return_events}_events.pkl") 
            sel_events.to_pickle(file_name)

            #file_name = os.path.join(exp_folder, "standstill_events.pkl") 
            #standstill_events.to_pickle(file_name)
            
    return prop_df, sel_events

def get_distancing_efficiency(events, kinematics, locations, fps, distancing_limits, node="thorax", metric=""):
    for i, event in events.iterrows():
        fly = f"fly{event['Fly']}"
        if fly == "fly0":
            other_fly = "fly1"
        else:
            other_fly = "fly0"
        start_frame = event["Start"]
        stop_frame = event["Stop"]
        closest_frame = event["Ind min dist"]

        abs_acc1 = np.abs(kinematics[fly][node]["lin_acc"][start_frame:stop_frame])
        abs_vel1 = np.abs(kinematics[fly][node]["lin_vel"][start_frame:stop_frame])
        #acc1 = kinematics[fly][node]["lin_acc"][start_frame:stop_frame]
        #vel1 = kinematics[fly][node]["lin_vel"][start_frame:stop_frame]
        #abs_acc2 = np.abs(kinematics[other_fly][node]["lin_acc"][start_frame:stop_frame])
        #abs_vel2 = np.abs(kinematics[other_fly][node]["lin_vel"][start_frame:stop_frame])
        vel_betw = get_kinematics(event["Dist between"], fps)
       
        profile =  abs_acc1 * vel_betw #* (abs_vel1/(abs_vel1+abs_vel2)) #* vel_ang
        
        roi_start = closest_frame-int(fps/4) if closest_frame >= int(fps/4) else 0
        roi_stop = closest_frame+int(fps/4) if len(profile)-closest_frame >= int(fps/4) else len(profile)
        roi = profile[roi_start:roi_stop]
        auc = simpson(roi, dx=1/fps)

        perc_15 = distancing_limits[0]
        perc_85 = distancing_limits[1]
        diff_val = perc_85 - perc_15
        
        
        auc_norm = auc/((roi_stop-roi_start)/fps) * (event["Min distance"]-perc_15)/diff_val * (np.mean(abs_vel1[closest_frame:roi_stop])/np.mean(abs_vel1[roi_start:closest_frame]))

        events.at[i,"AUC"] = auc_norm
        

def get_immobile_freezing(events, kinematics, locations, fps, distancing_limits, node="thorax", metric=""):
    for i, event in events.iterrows():
        fly = f"fly{event['Fly']}"
        if fly == "fly0":
            other_fly = "fly1"
        else:
            other_fly = "fly0"
        start_frame = event["Start"]
        stop_frame = event["Stop"]
        closest_frame = event["Ind min dist"]

        abs_acc1 = np.abs(kinematics[fly][node]["lin_acc"][start_frame:stop_frame])
        abs_vel1 = np.abs(kinematics[fly][node]["lin_vel"][start_frame:stop_frame])
        
        vel_betw = get_kinematics(event["Dist between"], fps)
        high_freq = get_high_freq_mov(locations, fps, start_frame, stop_frame, window_size=fps/8)
       
        profile =  abs_acc1 * vel_betw      

        roi_start = closest_frame-int(fps/4) if closest_frame >= int(fps/4) else 0
        roi_stop = closest_frame+int(fps/4) if len(profile)-closest_frame >= int(fps/4) else len(profile)
        
        roi_freezing = profile[closest_frame:roi_stop]
        auc_freezing = simpson(roi_freezing, dx=1/fps)

        roi_hf = high_freq[closest_frame:roi_stop]
        auc_hf = simpson(roi_hf, dx=1/fps)

        auc_freezing_norm = ((auc_freezing - auc_hf)/((roi_stop-closest_frame)/fps)) * (np.mean(abs_vel1[closest_frame:roi_stop])/np.mean(abs_vel1[roi_start:closest_frame]))

        events.at[i,"AUC"] = auc_freezing_norm
        

def get_high_freq_mov(locations, fps, start, stop, legs=["Fleg","Hleg"], window_size=None):

    hf_signals = np.zeros(stop-start)
    cutoff = 7 #Hz
    order_highpass = 5
    
    order_lowpass = 2

    if window_size:
        window_size = int(window_size)
    else:
        window_size = int(fps/4)
    if window_size%2 == 0:
        window_size+=1
    
    for leg in legs:
        r_leg = "R"+leg
        l_leg = "L"+leg
        r_pos = locations[r_leg]["locations"][start:stop,:]
        l_pos = locations[l_leg]["locations"][start:stop,:]        
        mean_pos = np.mean([r_pos,l_pos],axis=0)
        #plt.plot(r_pos.T[0],r_pos.T[1],label=r_leg)
        #plt.plot(l_pos.T[0],l_pos.T[1],label=l_leg)
        #plt.plot(mean_pos.T[0],mean_pos.T[1],label="mean_"+leg)
        #plt.xlim(0,400)
        #plt.ylim(0,400)
        #plt.legend()
        #plt.show()
        
        norm_leg = np.linalg.norm(mean_pos,axis=1)

        norm_leg -= norm_leg[0]

        #plt.plot(norm_leg,label=leg)
        #plt.legend()
        #plt.show()
        
        sos = butter(order_highpass, cutoff, btype='high', fs= fps, output='sos')
        highpass_norm = abs(sosfilt(sos, norm_leg))
        #plt.plot(highpass_norm[20:],label="highpass")
        
        filter_groom = savgol_filter(highpass_norm, window_size, order_lowpass)
        hf_signals += filter_groom
        #plt.plot(filter_groom,label="filtered-5th order-"+leg)
    #hf_signals[hf_signals<1]=0
    #plt.plot(hf_signals,label="sum")
    #plt.ylim(0,10)
    #plt.legend()
    #plt.show()
    return hf_signals

def get_thresholds_from_controls(control_path, th_standstill=0.85, th_moving=0.85):
    from Sociability_Learning.utils_files import get_experiments, Experiment
    from Sociability_Learning.utils_sleap import load_sleap_tracking
    
    min_dist=[]
    duration=[]
    all_prox_events=[]
    all_kinematics=[]
    all_auc_moving=[]
    all_auc_standstill=[]
    adaptation_time= 120#s

    all_experiments = get_experiments(control_path)
    for j, path_exp in enumerate(all_experiments):

        print(f"Getting thresholds from {path_exp}")
        try:
            exp = Experiment(path_exp)
            data_path = os.path.join(exp.folder, exp.vidName)
            locations, num_flies, fps = load_sleap_tracking(data_path)
            total_exp_time = len(locations["fly0"]["thorax"]["locations"])/fps

        except Exception as e:
            print('Experiment exception: ' + str(e))
            continue

        if num_flies == 2:
            kinematics={}
            orientation=[]
            for fly in locations.keys():
                
                orientation = get_orientation(locations[fly]['thorax']["locations"],
                                              locations[fly]['head']["locations"])    


                kinematics[fly] = get_kinematics(locations[fly],
                                                 fps,
                                                 orientation,
                                                 nodes=["thorax","head"])

                kinematics[fly]["heading"] = orientation

            proximity_events = get_proximity_events(exp,
                                                    locations,
                                                    fps,
                                                    dist_limit=5,#mm
                                                    gap_between_events=fps/4,#Frames
                                                    event_min_length=fps/2,
                                                    event_max_length=10*fps,
                                                    save_pkl=False,
                                                    from_file=False,
                                                    omit_events=[0,adaptation_time*fps])#Frames

            min_num_events = (total_exp_time-adaptation_time)/(60*2)

            if len(proximity_events)/num_flies < min_num_events: #len(proximity_events) == 0:
                print("Discarding experiment because there aren't enough proximity events")
                continue


            get_reaction_classes(proximity_events, kinematics)

            all_prox_events.append(proximity_events)
            all_kinematics.append(kinematics)
            min_dist.extend(proximity_events["Min distance"][:int(len(proximity_events)/2)])
            duration.extend(proximity_events["Duration"][:int(len(proximity_events)/2)])

        else:
            print(f"Num flies is: {num_flies}")
                
    min_dist_df = pd.Series(min_dist)
    low_th_dist = min_dist_df.quantile(0.15)
    high_th_dist = min_dist_df.quantile(0.85)
    distancing_limits = [low_th_dist, high_th_dist]

    for proximity_events, kinematics in zip(all_prox_events,all_kinematics):
        data_path = os.path.join(proximity_events["Folder"].values[0], proximity_events["Video name"].values[0])
        locations, num_flies, fps = load_sleap_tracking(data_path)
        for fly in range(2):
            standstill_events = proximity_events.loc[(proximity_events["Fly"] == fly)&
                                                     (proximity_events["post"] == 0)]

            moving_events = proximity_events.loc[(proximity_events["Fly"] == fly)&
                                                 (proximity_events["post"] == 1)]

            get_immobile_freezing(standstill_events, kinematics, locations[f"fly{fly}"], fps, distancing_limits)
            all_auc_standstill.extend(standstill_events["AUC"])

            get_distancing_efficiency(moving_events, kinematics, locations[f"fly{fly}"], fps, distancing_limits)
            all_auc_moving.extend(moving_events["AUC"])
    
    all_auc_standstill_df = pd.Series(all_auc_standstill)
    standstill_th = all_auc_standstill_df.quantile(th_standstill)

    all_auc_moving_df = pd.Series(all_auc_moving)
    moving_th = all_auc_moving_df.quantile(th_moving)

    return [[low_th_dist, high_th_dist], standstill_th, moving_th]

