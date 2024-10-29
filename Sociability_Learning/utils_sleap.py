import os
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from Sociability_Learning.utils_videos import get_raw_images
from Sociability_Learning.metrics import get_distance_between


def fill_missing(Y, filter_window, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        y_filtered = median_filter(y, size=filter_window, mode='nearest')
    
        # Save slice
        Y[:, i] = y_filtered

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)    

    return Y

def add_second_tracking(sleap_data, vid_path):
    data_path = vid_path.replace(".mp4","_tracked_simple.pkl")
    if not os.path.isfile(data_path):
        return sleap_data
    df_second = pd.read_pickle(data_path)
    
    second_0 = df_second[0].values
    second_1 = df_second[1].values

    fly_loc = sleap_data["fly0"]["thorax"]["locations"]
    dist_to_0 = get_distance_between(fly_loc, second_0)
    dist_to_1 = get_distance_between(fly_loc, second_1)
    diff_pos = dist_to_1 - dist_to_0

    new_tracking = np.array([one_pos if dist > 0 else zero_pos for dist, zero_pos, one_pos in zip(diff_pos, second_0, second_1)])
    
    sleap_data["weevil"] = {}
    sleap_data["weevil"]["thorax"] = {}
    sleap_data["weevil"]["thorax"]["locations"] = new_tracking
    sleap_data["weevil"]["thorax"]["scores"] = np.ones(len(new_tracking))

    
    return sleap_data
    

def clean_tracking(sleap_data, num_flies=2, second_tracking=False, path=""):

    for fly in range(num_flies-1):
        loc0 = sleap_data[f"fly{fly}"]["thorax"]["locations"][2:]
        loc1 = sleap_data[f"fly{fly+1}"]["thorax"]["locations"][2:]

        loc0_prev = sleap_data[f"fly{fly}"]["thorax"]["locations"][:-2]
        loc1_prev = sleap_data[f"fly{fly+1}"]["thorax"]["locations"][:-2]

        dist_intra0 = get_distance_between(loc0, loc0_prev)
        dist_inter0 = get_distance_between(loc0, loc1_prev)
        diff_dist0 = dist_inter0 - dist_intra0
        potential_swap0 = np.where(diff_dist0<0)[0]

        dist_intra1 = get_distance_between(loc1, loc1_prev)
        dist_inter1 = get_distance_between(loc1, loc0_prev)
        diff_dist1 = dist_inter1 - dist_intra1
        potential_swap1 = np.where(diff_dist1<0)[0]

        swap_points = np.intersect1d(potential_swap0, potential_swap1)

        diff_swap = np.diff(swap_points)

        if len(diff_swap)>0:
            duplicated_points = np.where(diff_swap<5)[0]
            clean_swap_points = [swap for i, swap in enumerate(swap_points) if i not in duplicated_points]
        else:
            clean_swap_points = list(swap_points)        

        clean_swap_points.insert(0, 0)
        clean_swap_points.append(len(sleap_data[f"fly{fly}"]["thorax"]["locations"]))


        for landmark, data in sleap_data[f"fly{fly}"].items():
            for dataset in data.keys():
                new_values0 = []
                new_values1 = []
                for ind, swap in enumerate(clean_swap_points[1:]):
                    prev_point = clean_swap_points[ind]
                    if ind%2 == 0:
                        new_values0.extend(data[dataset][prev_point:swap])
                        new_values1.extend(sleap_data[f"fly{fly+1}"][landmark][dataset][prev_point:swap])
                    else:
                        new_values0.extend(sleap_data[f"fly{fly+1}"][landmark][dataset][prev_point:swap])
                        new_values1.extend(data[dataset][prev_point:swap])

        
                data[dataset] = np.array(new_values0)
                sleap_data[f"fly{fly+1}"][landmark][dataset] = np.array(new_values1)

    if second_tracking:
        sleap_data = add_second_tracking(sleap_data, path)

    return sleap_data
    

def load_sleap_tracking(vid_path, nodes=[], min_frames= 47960, max_frames=np.inf, flies=2, get_node_names=False, filter_window=20, second_tracking=False):
    sleap_data={}
    num_flies = 0
    fps= 0
    vid_info, raw_imgs = get_raw_images(vid_path, num_imgs=1)
    fps = int(vid_info.get(5))
    file_name = vid_path.split('/')[-1]

    data_path = vid_path.replace(".mp4","-analysis.h5")
        
    if os.path.isfile(data_path):
        print(f"Adding data from: {data_path}")
        with h5py.File(data_path, "r") as f:
            dset_names = list(f.keys())
            locations = f["tracks"][:].T
            if "point_scores" in dset_names:
                scores = f["point_scores"][:].T
            else:
                scores = np.ones((locations.shape[0],locations.shape[1],locations.shape[3]))
                
            node_names = [n.decode() for n in f["node_names"][:]]
    
        nodes_index = {}
        for ind, name in enumerate(node_names):
            nodes_index[name] = ind

        num_frames, num_nodes, _, num_flies = locations.shape
        #print(locations.shape)
        locations = fill_missing(locations, filter_window)

        if num_frames < min_frames:
            raise Exception(f"Not processing because num of frames= {num_frames}")

        if num_flies > flies and not second_tracking:
            raise Exception(f"Not processing because num of flies= {num_flies}")

        if num_frames > max_frames:
            locations = locations[:max_frames,:,:,:]
            scores = scores[:max_frames,:,:]
        if not nodes:
            nodes = node_names

        sample_x_fly0 = locations[:fps,nodes_index["thorax"],0,0]
        if num_flies > 1:
            sample_x_fly1 = locations[:fps,nodes_index["thorax"],0,1]

            if np.mean(sample_x_fly0) < np.mean(sample_x_fly1):
                order_flies = [0, 1]
            else:
                order_flies = [1, 0]
        else:
            order_flies = [0]
            
        for fly in range(num_flies): 
            sleap_data[f'fly{order_flies[fly]}'] = {}
            for name in nodes:
                sleap_data[f'fly{order_flies[fly]}'][name] = {}
                sleap_data[f'fly{order_flies[fly]}'][name]["locations"] = locations[:,nodes_index[name],:,fly]
                sleap_data[f'fly{order_flies[fly]}'][name]["scores"] = scores[:,nodes_index[name],fly]
          
    else:
        raise Exception(f"File {data_path} not found.")

    sleap_data_clean = clean_tracking(sleap_data, num_flies=num_flies, second_tracking=second_tracking, path=vid_path)
    
    if get_node_names:
        return sleap_data_clean, num_flies, fps, nodes
    else:
        return sleap_data_clean, num_flies, fps

