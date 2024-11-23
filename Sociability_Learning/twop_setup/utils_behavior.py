import os
import random
import pickle
import pkgutil
import cv2 as cv
import numpy as np
import pandas as pd
import utils2p
import utils2p.synchronization
from pathlib import Path
from Sociability_Learning.utils_videos import get_raw_images
from Sociability_Learning.utils_sleap import load_sleap_tracking
from Sociability_Learning.twop_setup.utils_fictrac import get_fictrac_df


def run_sleap_model(video_name, model_folder):
    num_flies=1
    batch_size = 32
    
    sleap_file = video_name.replace(".mp4", "-inference.slp")
    tracking_file = video_name.replace(".mp4", "-tracking.slp")
    analysis_file = video_name.replace(".mp4", "-analysis.h5")
    
    flags_performance = f"--verbosity rich --batch_size {batch_size}"
    flags_tracking = f"--tracking.tracker simple --tracking.target_instance_count {num_flies} --tracking.similarity instance --tracking.match hungarian"
    flags_corrections = f"--tracking.post_connect_single_breaks {num_flies} --tracking.clean_instance_count {num_flies}"
    inference_cmd = f"sleap-track {video_name} -m {model_folder} {flags_performance} --output {sleap_file}"
    os.system(inference_cmd)
    
    tracking_cmd = f"sleap-track {sleap_file} {flags_tracking} {flags_corrections} --output {tracking_file}"
    os.system(tracking_cmd)
    
    convert_to_h5_cmd = f"sleap-convert {tracking_file} --format analysis --output {analysis_file}"
    os.system(convert_to_h5_cmd)

   
def get_fly_coords(img):
    h, w = img.shape
    
    output = cv.connectedComponentsWithStats(img, 4, cv.CV_32S)
    stats = np.transpose(output[2])
    #sizes = stats[4]
    centroids = np.transpose(output[3])
    left = stats[0]
    closest_to_center = [l-w/2 for l in left]
    label_fly = np.argsort(np.abs(closest_to_center))[0]

    x = centroids[0][label_fly]
    y = centroids[1][label_fly]
        
    return x, y


def get_coords(imgs, cam):
    all_x = []
    all_y = []
    
    for img_original in imgs:
        gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
        img = cv.pyrUp(cv.pyrDown(gray))
        _, img_bw = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        #_, img_bw = cv.threshold(img, 90, 255, cv.THRESH_BINARY)
        if cam == "front":
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (50, 4))
        else:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 70))
        img_clean = cv.morphologyEx(img_bw, cv.MORPH_OPEN, kernel)
        x, y = get_fly_coords(img_clean)
        all_x.append(x)
        all_y.append(y)
        
        #cv.imshow('flies', img_clean)
        #cv.waitKey()
        
    x_fly = int(np.mean(all_x))
    y_fly = int(np.mean(all_y))

    return [x_fly, y_fly]

def get_tethered_coords(all_trials):
    
    alone_trial = [trial for trial in all_trials if "alone" in trial][0]

    alone_front = os.path.join(alone_trial, "images", "camera_4.mp4")
    alone_side = os.path.join(alone_trial, "images", "camera_6.mp4")

    _, imgs_front = get_raw_images(alone_front, num_imgs=200, start=0)
    random_frames_front = random.sample(imgs_front, 50)
    front_coords = get_coords(random_frames_front, "front")
    #print(front_coords)

    _, imgs_side = get_raw_images(alone_side, num_imgs=200, start=0)
    random_frames_side = random.sample(imgs_side, 50)
    side_coords = get_coords(random_frames_side, "side")
    #print(side_coords)
    
    return front_coords[0], side_coords[0]



def get_free_fly_tracking(all_trials, px_size_front=18.5/960, px_size_side=14/1264):
   
    x_0, y_0 = get_tethered_coords(all_trials)
    #print(x_0, y_0)

    for trial in all_trials:
        if "alone" in trial or "before" in trial:
            continue
        beh_folder = os.path.join(trial,  "images")
        front_cam = os.path.join(beh_folder, "camera_4.mp4")
        side_cam = os.path.join(beh_folder, "camera_6.mp4")

        front_analysis = front_cam.replace(".mp4", "-analysis.h5")
        side_analysis = side_cam.replace(".mp4", "-analysis.h5")

        package = pkgutil.get_loader("Sociability_Learning")
        path_package = Path(package.get_filename())

        if not os.path.isfile(front_analysis):
            model_folder = os.path.join(path_package.parents[1],
                                    'data',
                                    'sleap_models',
                                    'front_2p',
                                    'models',
                                    '240417_170822.multi_instance.n=150')
            run_sleap_model(front_cam, model_folder)

        if not os.path.isfile(side_analysis):
            model_folder = os.path.join(path_package.parents[1],
                                    'data',
                                    'sleap_models',
                                    'side_2p',
                                    'models',
                                    '240415_200739.multi_instance.n=200')
            run_sleap_model(side_cam, model_folder)


        try:
            locations_front, num_flies, fps, nodes = load_sleap_tracking(front_cam, min_frames=29000, get_node_names=True)

            locations_side, _, _, _ = load_sleap_tracking(side_cam, min_frames=29000, get_node_names=True)
        except:
            continue
        x_pos = (x_0 - locations_front["fly0"]["thorax"]["locations"][:,0]) * px_size_front
        y_pos = (locations_side["fly0"]["thorax"]["locations"][:,0] - y_0) * px_size_side

        x_pos_head = (x_0 - locations_front["fly0"]["head"]["locations"][:,0]) * px_size_front
        y_pos_head = (locations_side["fly0"]["head"]["locations"][:,0] - y_0) * px_size_side

        x_pos_abdomen = (x_0 - locations_front["fly0"]["abdomen"]["locations"][:,0]) * px_size_front
        y_pos_abdomen = (locations_side["fly0"]["abdomen"]["locations"][:,0] - y_0) * px_size_side

        theta_head = np.arctan2(y_pos_head-y_pos, x_pos_head-x_pos)
        theta_abdomen = np.arctan2(y_pos_abdomen-y_pos, x_pos_abdomen-x_pos)

        tracking_results = [x_pos, y_pos, theta_head, theta_abdomen]

        results_path = os.path.join(beh_folder, "free_fly_tracking.pickle")

        with open(results_path, "wb") as f:
            pickle.dump(tracking_results, f)

        #plt.plot(x_pos, y_pos)
        #plt.title(trial)
        #plt.show()


def get_trial_info(trial_dir, i_trial):
    #print("trial info")
    _, trial_name = os.path.split(trial_dir)
    fly_dir = Path(trial_dir).parent
    tmp, fly = os.path.split(fly_dir)
    fly_num = int(fly.split("-")[0][3:])
    _, date_gen = os.path.split(tmp)
    date = int(date_gen[:6])
    genotype = date_gen[7:]
 
    trial_info = {"Date": date,
                  "Genotype": genotype,
                  "Fly": fly_num,
                  "TrialName": trial_name,
                  "i_trial": i_trial
                  }
    return trial_info

def get_processed_lines(sync_file,
                        sync_metadata_file,
                        metadata_2p_file,
                        seven_camera_metadata_file=None,
                        read_cams=True,
                        additional_lines=["LightSheetGalvoPosition",
                                          "LightSheetLaserOn",
                                          "LightSheetLaserPower",
                                          "odor", "pid"]):
    """
    This function extracts all the standard lines and processes them.
    It works for both microscopes.
    Parameters
    ----------
    sync_file : str
        Path to the synchronization file.
    sync_metadata_file : str
        Path to the synchronization metadata file.
    metadata_2p_file : str
        Path to the ThorImage metadata file.
    seven_camera_metadata_file : str
        Path to the metadata file of the 7 camera system.
    additional_lines : list
        Line names of additional synchronisation lines to load.
    Returns
    -------
    processed_lines : dictionary
        Dictionary with all processed lines.
    Examples
    --------
    >>> import utils2p
    >>> import utils2p.synchronization
    >>> experiment_dir = "data/mouse_kidney_raw/"
    >>> sync_file = utils2p.find_sync_file(experiment_dir)
    >>> metadata_file = utils2p.find_metadata_file(experiment_dir)
    >>> sync_metadata_file = utils2p.find_sync_metadata_file(experiment_dir)
    >>> seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(experiment_dir)
    >>> processed_lines = utils2p.synchronization.get_processed_lines(sync_file, sync_metadata_file, metadata_file, seven_camera_metadata_file)
    """
    processed_lines = {}
    processed_lines["Capture On"], processed_lines[
        "Frame Counter"] = utils2p.synchronization.get_lines_from_sync_file(
            sync_file, ["CaptureOn", "FrameCounter"])

    try:
        # For microscope 1
        if read_cams:
            processed_lines["CO2"], processed_lines["Cameras"], processed_lines[
                "Optical flow"] = utils2p.synchronization.get_lines_from_sync_file(sync_file, [
                    "CO2_Stim",
                    "Basler",
                    "OpFlow",
                ])
        else:
            processed_lines["CO2"], processed_lines[
                "Optical flow"] = utils2p.synchronization.get_lines_from_sync_file(sync_file, [
                    "CO2_Stim",
                    "OpFlow",
                ])
    except KeyError:
        # For microscope 2
        if read_cams:
            processed_lines["CO2"], processed_lines[
                "Cameras"] = utils2p.synchronization.get_lines_from_h5_file(sync_file, [
                    "CO2_Stim",
                    "Basler",
                ])
        else:
            processed_lines["CO2"] = utils2p.synchronization.get_lines_from_h5_file(sync_file, ["CO2"])

    if read_cams:
        processed_lines["Cameras"] = utils2p.synchronization.process_cam_line(processed_lines["Cameras"],  # TODO
                                                    seven_camera_metadata_file)
    if metadata_2p_file is not None:
        metadata_2p = utils2p.Metadata(metadata_2p_file)
        processed_lines["Frame Counter"] = utils2p.synchronization.process_frame_counter(
            processed_lines["Frame Counter"], metadata_2p)
    if len(np.unique(processed_lines["CO2"])) > 1:  # i.e. the CO2 line was actually used
        processed_lines["CO2"] = utils2p.synchronization.process_stimulus_line(processed_lines["CO2"])
    if "Optical flow" in processed_lines.keys() and np.sum(processed_lines["Optical flow"])>0:
        processed_lines["Optical flow"] = utils2p.synchronization.process_optical_flow_line(
            processed_lines["Optical flow"])

    for add_key in additional_lines:
        try:
            processed_lines[add_key] = utils2p.synchronization.get_lines_from_h5_file(sync_file,
                [add_key])[0]
        except:
            print(f"Could not load line {add_key} from sync file {sync_file}")

    if metadata_2p_file is not None:
        mask = np.logical_and(processed_lines["Capture On"],
                            processed_lines["Frame Counter"] >= 0)

        # Make sure the clipping start just before the
        # acquisition of the first frame
        indices = np.where(mask)[0]
        mask[max(0, indices[0] - 1)] = True

        for line_name, _ in processed_lines.items():
            processed_lines[line_name] = utils2p.synchronization.crop_lines(mask, [
                processed_lines[line_name],
            ])[0]

    # Get times of ThorSync ticks
    metadata = utils2p.synchronization.SyncMetadata(sync_metadata_file)
    freq = metadata.get_freq()
    times = utils2p.synchronization.get_times(len(processed_lines["Frame Counter"]), freq)
    processed_lines["Times"] = times

    return processed_lines

def get_frame_times_indices(trial_dir, beh_trial_dir=None, sync_trial_dir=None):
    """get the times of different data acquisition modalities from the sync files.
    Also computes the frame indices to lateron sync between behavioural and two photon data.

    Attention: the absolute time is only precise to the second because it is based on the 
    unix time stamp from the metadatafile of the recording.

    Parameters
    ----------
    trial_dir : str
        directory containing the 2p data and ThorImage output

    beh_trial_dir : [type], optional
        directory containing the 7 camera data. If not specified, will be set equal
        to trial_dir,, by default None

    sync_trial_dir : [type], optional
        directory containing the output of ThorSync. If not specified, will be set equal
        to trial_dir, by default None

    Returns
    -------
    unix_t_start: int
        start time of the experiment as a unix time stamp

    frame_times_2p: numpy array
        absolute times when the two photon frames were acquired

    frame_times_beh: numpy array
        absolute times when the 7 camera images were acquired

    beh_frame_idx: numpy array
        same length as frame_times_beh. contains the index of the two photon frame
        that was acquired during each behaviour frame

    (frame_times_opflow): numpy array
        only returned if opflow == True
        absolute times when the optic flow sensor values were acquired

    (opflow_frame_idx): numpy array
        only returned if opflow == True
        same length as frame_times_opflow. contains the index of the two photon frame
        that was acquired during each optical flow sample
    """
    beh_trial_dir = trial_dir if beh_trial_dir is None else beh_trial_dir
    sync_trial_dir = trial_dir if sync_trial_dir is None else sync_trial_dir

    sync_file = utils2p.find_sync_file(sync_trial_dir)
    try:
        metadata_file = utils2p.find_metadata_file(trial_dir)
        twop_present = True
    except:
        metadata_file = None
        twop_present = False
        print(f"Could not finde 2p data for trial {trial_dir}. will proceed with behavioural data only.")
    sync_metadata_file = utils2p.find_sync_metadata_file(sync_trial_dir)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(beh_trial_dir)

    unix_t_start = int(utils2p.Metadata(metadata_file).get_metadata_value("Date", "uTime")) if twop_present else 0
    # don't use utils2p.synchronization, but temporarily use the one below to keep flexibility if twop was not recorded
    processed_lines = get_processed_lines(sync_file, sync_metadata_file,
                                          metadata_file, seven_camera_metadata_file,
                                          additional_lines=[])
    frame_times_beh = utils2p.synchronization.get_start_times(processed_lines["Cameras"],
                                                              processed_lines["Times"])
    if twop_present:
        frame_times_2p = utils2p.synchronization.get_start_times(processed_lines["Frame Counter"],
                                                                processed_lines["Times"])
        beh_frame_idx = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(len(frame_times_beh)),
                                                                processed_lines["Cameras"],
                                                                processed_lines["Frame Counter"])
        
        
        beh_frame_idx[beh_frame_idx<0] = -9223372036854775808  # smallest possible uint64 number
        beh_frame_idx[beh_frame_idx>=len(frame_times_2p)] = -9223372036854775808
    else:
        frame_times_2p = frame_times_beh
        beh_frame_idx = np.ones_like(frame_times_beh) * -9223372036854775808
    
    return unix_t_start, frame_times_2p, frame_times_beh, beh_frame_idx

def get_multi_index_trial_df(trial_info, N_samples, t=None, twop_index=None, abs_t_start=None):
    """create an empty dataframe for a trial that can later be enriched with data

    Parameters
    ----------
    trial_info : dict
        dictionary with info about the trial containing:
        Date (int), Genotype (str), Fly (int), TrialName (str), i_trial (int)

    N_samples : int
        number of samples in the trial

    t : numpy array, optional
        experiment time vector to be added as "t" column in the data frame, by default None

    twop_index : numpy array, optional
        synchronisation indices. to be added as "twop_index" column, by default None

    abs_t_start : int or float, optional
        absolute time of first sample. t + abs_t_start will be added as "abs_t" column
        in the data frame, by default None

    Returns
    -------
    pandas DataFrame
        multiindex dataframe with the trial_info and potentially a "t", "abs_t",
        and a "twop_index" column.
    """
    frames = np.arange(N_samples)
    indices = pd.MultiIndex.from_arrays(([trial_info["Date"], ] * N_samples,  # e.g 210301
                                            [trial_info["Genotype"], ] * N_samples,  # e.g. 'J1xCI9'
                                            [trial_info["Fly"], ] * N_samples,  # e.g. 1
                                            [trial_info["TrialName"], ] * N_samples,  # e.g. 1
                                            [trial_info["i_trial"], ] * N_samples,  # e.g. 1
                                            frames
                                        ),
                                        names=[u'Date', u'Genotype', u'Fly',
                                               u'TrialName', u'Trial', u'Frame'])
    df = pd.DataFrame(index=indices)
    if t is not None:
        assert len(t) == N_samples
        df["t"] = t
        if abs_t_start is not None:
            df["abs_t"] = abs_t_start + t
    if twop_index is not None:
        assert len(twop_index) == N_samples
        df["twop_index"] = twop_index
    return df

def get_multi_index_fly_df(trial_dfs=None):
    """append multiple trial index dataframes to one fly dataframe

    Parameters
    ----------
    trial_dfs : list of pandas DataFrames, optional
        list of dataframes as generated by get_multi_index_trial_df(), by default None

    Returns
    -------
    pandas DataFrame
       one dataframe for a fly that unites all trial dataframes
    """
    multi_index = pd.MultiIndex(levels=[[]] * 5, codes=[[]]  * 5,
                                names=[u'Date', u'Genotype', u'Fly',
                                       u'TrialName', u'Trial', u'Frame'])
    df = pd.DataFrame(index=multi_index)
    if trial_dfs is not None and isinstance(trial_dfs, list):
        for trial_df in trial_dfs:
            df.append(trial_df)
    return df

def get_synchronized_dataframes(beh_dirs):
    for i_trial, beh_dir in enumerate(beh_dirs):
        trial_dir = Path(beh_dir).parent
        sync_dir = os.path.join(trial_dir,"2p")
        print("Creating data frames: " + str(trial_dir))
        trial_info = get_trial_info(trial_dir, i_trial)
               
        unix_t_start, frame_times_2p, frame_times_beh, beh_frame_idx = \
            get_frame_times_indices(trial_dir, beh_trial_dir=beh_dir, sync_trial_dir=sync_dir)

        twop_df = get_multi_index_trial_df(trial_info, len(frame_times_2p), t=frame_times_2p,
                                       abs_t_start=unix_t_start)

        beh_out_dir = os.path.join(trial_dir, "processed", "beh_df.pkl")
        twop_out_dir = os.path.join(trial_dir, "processed", "twop_df.pkl")

        twop_df.to_pickle(twop_out_dir)
        beh_df = get_multi_index_trial_df(trial_info, len(frame_times_beh), t=frame_times_beh, twop_index=beh_frame_idx, abs_t_start=unix_t_start)
        beh_df.to_pickle(beh_out_dir)
        
        _ = get_fictrac_df(str(trial_dir),
                           index_df=beh_out_dir,
                           df_out_dir=beh_out_dir)
