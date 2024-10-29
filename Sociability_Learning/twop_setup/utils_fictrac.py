"""
sub-module to run and analyse fictrac.
Includes functions to prepare the required config file and run ficrac in a new process.
Includes functionality to read results from fictrac & combine them with an existing Pandas dataframe
partially copied and modified from Florian Aymann's
https://github.com/NeLy-EPFL/ABO_data_processing/blob/master/add_fictrac_config.py
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import multiprocessing
import subprocess
import glob
from scipy.ndimage import gaussian_filter1d, median_filter



IGNORE_ROI = [[280, 100, 280, 200, 740, 200, 740, 100], [200, 430, 200, 480, 800, 480, 800, 430]]

# see https://github.com/rjdmoore/fictrac/blob/master/doc/data_header.txt for fictrac output description
col_names = ["Frame_counter",
             "delta_rot_cam_right", "delta_rot_cam_down", "delta_rot_cam_forward",
             "delta_rot_error",
             "delta_rot_lab_side", "delta_rot_lab_forward", "delta_rot_lab_turn",
             "abs_rot_cam_right", "abs_rot_cam_down", "abs_rot_cam_forward",
             "abs_rot_lab_side", "abs_rot_lab_forward", "abs_rot_lab_turn",
             "integrated_lab_x", "integrated_lab_y",
             "integrated_lab_heading",
             "animal_movement_direction_lab",
             "animal_movement_speed",
             "integrated_forward_movement", "integrated_side_movement",
             "timestamp",
             "seq_counter",
             "delta_time",
             "alt_time"
            ]

f_s = 100
r_ball = 4.95


def run_shell_command(command, allow_ctrl_c=True, suppress_output=False):
    """use the subprocess module to run a shell command

    Parameters
    ----------
    command : str
        shell command to execute

    allow_ctrl_c : bool, optional
        whether a CTRL+C event will allow to continue or not, by default True

    suppress_output : bool, optional
        whether to not show outputs, by default False
    """
    if allow_ctrl_c:
        try:
            if suppress_output:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
            else:
                process = subprocess.Popen(command, shell=True)
            process.communicate()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
    else:
        if suppress_output:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
        else:
            process = subprocess.Popen(command, shell=True)
        process.communicate()



def apply_clahe(image):
    # Create a CLAHE object (you can adjust the clip limit and tile grid size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    # Apply CLAHE on the grayscale image
    clahe_image = clahe.apply(image)
    return clahe_image


def equalize_histogram(input_video_path, overwrite=False):    
    # Open the input video file
    output_video_path = input_video_path.replace(".mp4","_clahe.mp4")

    if os.path.isfile(output_video_path):# and not overwrite:
        return output_video_path

    print("Equalizing histogram...")
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format

    # Define the VideoWriter object to save the processed video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE on the grayscale frame
        clahe_frame = apply_clahe(frame)

        # Stack original and CLAHE-corrected frames horizontally for display (optional)
        #combined_frame = cv2.hconcat([frame, clahe_frame])

        # Display the combined frame
        #cv2.imshow('Original (Left) vs CLAHE (Right)', combined_frame)

        # Write the CLAHE-processed frame to the output video
        out.write(clahe_frame)

        # Exit when 'q' is pressed
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Release the video capture and writer objects, and close all OpenCV windows
    cap.release()
    out.release()
    #cv2.destroyAllWindows()

    return output_video_path


def get_mean_image(video_file, camera_num, skip_existing=True, max_count = 6000):
    """compute the mean image of a video and save it as a file.
    partially copied and modified from Florian Aymann's
    https://github.com/NeLy-EPFL/ABO_data_processing/blob/master/add_fictrac_config.py

    Parameters
    ----------
    video_file : string
        absolute path of the video to be averaged

    skip_existing : bool, optional
        if already computed, read the image and return it, by default True

    output_name : string, optional
        file name of the resulting mean image, by default "_camera_3_mean_image.jpg"

    Returns
    -------
    numpy array
        mean image
    """
    output_name = f"camera_{camera_num}_mean_image.jpg"
    directory = os.path.dirname(video_file)
    mean_frame_file = os.path.join(directory, output_name)
    if skip_existing and os.path.isfile(mean_frame_file):
        print(f"{mean_frame_file} exists loading image from file without recomputing.")
        mean_frame = cv2.imread(mean_frame_file,0)
    else:
        all_frames = []
        f = cv2.VideoCapture(video_file)
        rval, frame = f.read()
        # Convert rgb to grey scale
        mean_frame = np.zeros_like(frame[:, :, 0], dtype=np.int64)
        count = 0
        while rval and count<max_count:
            all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            mean_frame =  mean_frame + frame[:, :, 0]
            rval, frame = f.read()
            count += 1
        f.release()
        
        mean_frame = mean_frame / count
        mean_frame = mean_frame.astype(np.uint8)

        std_img = np.std(np.array(all_frames),axis=0).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))

        if np.mean(std_img) > 10:
            equ_img = clahe.apply(std_img)
        else:            
            equ_img = clahe.apply(mean_frame)       
        
        #equ_img = cv2.equalizeHist(std_img)

        #cv2.imshow("mean", mean_frame)
        #cv2.imshow("std", std_img)
        #cv2.imshow("equ", equ_img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        
        cv2.imwrite(mean_frame_file, equ_img)
    return equ_img

def get_ball_parameters(img, camera_num, output_dir=None):
    """Using an image that includes the ball, for example the mean image,
    compute the location and the radius.
    Uses cv2.HoughCircles to find circles in the image and then selects the most likely one.
    partially copied and modified from Florian Aymann's
    https://github.com/NeLy-EPFL/ABO_data_processing/blob/master/add_fictrac_config.py
    Parameters
    ----------
    img : np.array
        image to be analysed

    output_dir : string, optional
        if specified, make image that includes the analysis results and save to file,
        by default None

    Returns
    -------
    float
        x position in pixels

    float
        y position in pixels

    float
        radius in pixels
    """
    img = cv2.medianBlur(img, 5)
    canny_params = dict(threshold1 = 120, threshold2 = 60)  # Florian's original params: 40 & 50
    edges = cv2.Canny(img, **canny_params)
    inside = np.inf
    x_min, y_min, r_min = np.nan, np.nan, np.nan

    #cv2.imshow("img", img)
    #cv2.waitKey()

    for i in range(3):
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 200, param1=120, param2=10, minRadius=200, maxRadius=260)

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for x, y, r in circles:
                #cv2.circle(img, (x, y), r, (255, 255, 255), 1)
                #cv2.imshow("img", img)
                #cv2.waitKey()
                if x + r > img.shape[1] or x - r < 0:  # check that ball completely in the image in x
                    #print(f"in image:{x+r}>{img.shape[1]} or {x - r}<0")
                    continue
                elif x < img.shape[1] * 3 / 8 or x > img.shape[1] * 5 / 8:  # check that ball center in central quarter of x axis
                    #print(f"center:{x} < {img.shape[1] * 3 / 8} or {x} > {img.shape[1] * 5 / 8}")
                    continue
                elif y - r <= img.shape[0] / 10:  # check that top of the ball is below 1/10 of the image
                    #print(f"top: {y - r} <= {img.shape[0] / 10}")
                    continue
                #print(f"used: {x}, {y}, {r}")
                xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
                xx = xx - x
                yy = yy - y
                rr = np.sqrt(xx ** 2 + yy ** 2)
                mask = (rr < r)
                current_inside = np.mean(edges[mask])  # np.diff(np.quantile(edges[mask], [0.05, 0.95]))
                if  current_inside < inside:
                    x_min, y_min, r_min = x, y, r
                    inside = current_inside
        else:
            print("No circles found.")
    if output_dir is not None:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x, y, r in circles:
            #print(x, y, r)
            cv2.circle(img, (x, y), r, (255, 255, 255), 1)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (255, 128, 255), -1)
        #print(x_min, y_min, r_min)
        cv2.circle(img, (x_min, y_min), r_min, (255, 0, 0), 1)
        cv2.rectangle(img, (x_min - 5, y_min - 5), (x_min + 5, y_min + 5), (255, 128, 255), -1)
        cv2.imwrite(os.path.join(output_dir, f"camera_{camera_num}_circ_fit.jpg"), img)

        
    return x_min, y_min, r_min

def get_circ_points_for_config(x, y, r, img_shape, n=12):
    """convert circle parameters into individual points on the surfac of the ball
    as if they were generated from the fictrac config gui

    Parameters
    ----------
    x : float
        x position of ball in pixels

    y : float
        y position of ball in pixels

    r : float
        radius of ball in pixels

    img_shape : tuple/list
        shape of the image as (y, x)

    n : int, optional
        number of points, by default 12

    Returns
    -------
    list
        points on the ball surface, to be handed over to write_config_file()
    """
    # Compute angular limit given by image size
    theta1 = np.arcsin((img_shape[0] - y) / r)
    theta2 = 1.5 * np.pi - (theta1 - 1.5 * np.pi)

    points = []
    for theta in np.linspace(theta1, theta2, n):
        point_x = x - np.cos(theta) * r
        point_y = y - np.sin(theta) * r
        points.append(int(point_x))
        points.append(int(point_y))

    return points

def _format_list(l):
    """format a list as a string in a format that is suitable for the fictrac config file

    Parameters
    ----------
    l : list

    Returns
    -------
    string
    """
    s = repr(l)
    s = s.replace("[", "{ ")
    s = s.replace("]", " }")
    return s

def write_config_file(video_file, roi_circ, vfov=3.5017, q_factor=10, c2a_src="c2a_cnrs_yz", do_display="n",
                      c2a_t=[0.681828, -3.965286, 178.532700], c2a_r=[0.642342, -0.648050, -1.474485 ],
                      c2a_cnrs_yz=[532, 49, 488, 49, 488, 82, 532, 82], thr_ratio=1.1, thr_win_pc=0.2, overwrite=False,
                      ignore_roi=IGNORE_ROI, use_ball_template=False):
    """Create a config file for fictrac.
    See: https://github.com/rjdmoore/fictrac/blob/master/doc/params.md for interpretation of parameters

    Parameters
    ----------
    video_file : string
        absolute path of video file to run fictrac on

    roi_circ : list
        points on the circumference of the ball defining the ball.
        can be generated using get_circ_points_for_config()

    vfov : float, optional
        [description], by default 3.05

    q_factor : int, optional
        quality factor of fictrac, by default 40

    c2a_src : str, optional
        [description], by default "c2a_cnrs_xz"

    do_display : str, optional
        [description], by default "n"

    c2a_t : list, optional
        [description], by default [-5.800291, -23.501165, 1762.927645]

    c2a_r : list, optional
        [description], by default [1.200951, -1.196946, -1.213069]

    c2a_cnrs_xz : list, optional
        [description], by default [422, 0, 422, 0, 422, 10, 422, 10]

    overwrite : bool, optional
        whether to overwrite an existing config file, by default False

    ignore_roi : list, optional
        list of points defining the ROI to be ignored by Fictrac, by default IGNORE_ROI

    Returns
    -------
    string
        location of config file
    """

    template_path = "/mnt/upramdya_files/LOBATO_RIOS_Victor/Experimental_data/2p/camera_4_clahe-template.png"
    global_opt = "y"
    max_bad_frames = 5
    opt_max_err = 25000
    #opt_bound = 0.5
    
    directory = os.path.dirname(video_file)
    config_file = os.path.join(directory, "config.txt")
    if not overwrite and os.path.isfile(config_file):
        print(f"Not writing to {config_file} because it exists.")
        return config_file

    content = f"vfov             : {vfov:.2f}"
    content += f"\nsrc_fn           : {video_file}"
    content += f"\nq_factor         : {int(q_factor)}"
    content += f"\nc2a_src          : {c2a_src}"
    content += f"\ndo_display       : {do_display}"
    content += f"\nroi_ignr         : {_format_list(ignore_roi)}"
    content += f"\nc2a_t            : {_format_list(c2a_t)}"
    content += f"\nc2a_r            : {_format_list(c2a_r)}"
    content += f"\nc2a_cnrs_yz      : {_format_list(c2a_cnrs_yz)}"
    content += f"\nroi_circ         : {_format_list(roi_circ)}"
    content += f"\nthr_ratio        : {thr_ratio}"
    content += f"\nthr_win_pc       : {thr_win_pc}"
    content += f"\nopt_do_global    : {global_opt}"
    content += f"\nmax_bad_frames   : {max_bad_frames}"
    content += f"\nopt_max_err      : {opt_max_err}"
    if use_ball_template:
        content += f"\nsphere_map_fn    : {template_path}"
    #content += f"\nopt_bound        : {opt_bound}"
    
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(content)

    return config_file

def run_fictrac_config_gui(config_file, fictrac_config_gui="~/fictrac/bin/configGui"):
    """runs the fictrac config gui in a subprocess and sequentially sends "y\n" responses to continue.
    This is required because the config gui computes some parameters based on the inputs given.

    Parameters
    ----------
    config_file : str
        absolut path of config file

    fictrac_config_gui : str, optional
        location of fictrac config gui command, by default "~/fictrac/bin/configGui"
    """
    directory = os.path.dirname(config_file)
    command = f'/bin/bash -c "cd {directory} && yes | xvfb-run -a {fictrac_config_gui} {config_file}"'
    run_shell_command(command, allow_ctrl_c=False, suppress_output=True)

def run_fictrac(config_file, fictrac="~/fictrac/bin/fictrac"):
    """Runs fictrac in the current console using the subprocess module.
    The console will not be blocked, but the outputs will be printed regularily

    Parameters
    ----------
    config_file : str
        path to config file generate by the config gui or automatically

    fictrac : str, optional
        location of fictrac on computer, by default "~/fictrac/bin/fictrac"
    """
    command = f"{fictrac} {config_file}"
    run_shell_command(command, allow_ctrl_c=True, suppress_output=False)
    return

def get_treadmill_tracking(trial_dirs, camera_num=4, overwrite=False, check_results=True):
    """Automatically create config file for fictrac and then run it using the newly generated config.

    Parameters
    ----------
    fly_dir : string
        absolute directory pointing to a folder that contains the trial directories.
        Could be anything that is accepted by print() if trial_dirs is not None

    trial_dirs : list, optional
        if trial directories are not specified, automatically choose all subfolders of fly_dir
        that start with "0", by default None
    """
    
    N_trials = len(trial_dirs)

    config_files = []
    
    for trial_dir in tqdm(trial_dirs):
        video_file = utils.find_file(trial_dir, f"camera_{camera_num}.mp4")
        image_dir = os.path.dirname(video_file)
        if not os.path.isfile(video_file):
            print("Could not find video file: ", video_file, "Will continue.")
            continue
        video_file = equalize_histogram(video_file, overwrite=overwrite)
        mean_image = get_mean_image(video_file, camera_num, skip_existing= not overwrite)
        x_min, y_min, r_min = get_ball_parameters(mean_image, camera_num, output_dir=image_dir)
        points = get_circ_points_for_config(x_min, y_min, r_min, img_shape=mean_image.shape[:2])
        config_file = write_config_file(video_file, points, overwrite=overwrite)
        run_fictrac_config_gui(config_file)
        config_files.append(config_file)

    
    N_proc = np.minimum(10, len(config_files))
    multiprocessing.set_start_method('spawn', True)
    pool = multiprocessing.Pool(N_proc)
    pool.map(run_fictrac, config_files)
    
    if check_results:
        trials_to_fix = check_fictrac_results(trial_dirs)
        if len(trials_to_fix)>0:
            config_files = []
            for trial_dir in trials_to_fix:
                video_file = utils.find_file(trial_dir, f"camera_{camera_num}_clahe.mp4")
                config_file = write_config_file(video_file, points, overwrite=overwrite, use_ball_template=True)
                run_fictrac_config_gui(config_file)
                config_files.append(config_file)
            N_proc = np.minimum(10, len(config_files))
            multiprocessing.set_start_method('spawn', True)
            pool = multiprocessing.Pool(N_proc)
            pool.map(run_fictrac, config_files)
            

def get_v_th_from_fictrac(trial_dir, f_s=f_s, r_ball=r_ball):
    """extract the forward velocity and the orientation of the fly from the fictrac output
    see https://github.com/rjdmoore/fictrac/blob/master/doc/data_header.txt 
    for fictrac output description

    Parameters
    ----------
    trial_dir : string
        trial directory that contains the behData/images subfolder,
        which in turn holds the fictrac output

    f_s : float, optional
        sampling frequency, by default f_s

    r_ball : float, optional
        ball radius, by default r_ball

    Returns
    -------
    numpy array
        vector of velocity across time

    numpy array
        vector of orientation across time
    """
    # 
    trial_image_dir = os.path.join(trial_dir, "behData", "images")
    fictrac_data_file = glob.glob(os.path.join(trial_image_dir, "camera*.dat"))[0]

    # col_names = np.arange(25) + 1
    df = pd.read_csv(fictrac_data_file, header=None, names=col_names)

    v_raw = df["animal_movement_speed"] * f_s  # df[19] * f_s  # convert from rad/frame to rad/s
    th_raw = df["animal_movement_direction_lab"]  # df[18]

    v = gaussian_filter1d(median_filter(v_raw, size=5), sigma=10) * r_ball  # rad/s == mm/s on ball with 1mm radius
    th = (gaussian_filter1d(median_filter(th_raw, size=5), sigma=10) - np.pi) / np.pi * 180
    return v, th

def filter_fictrac(x, med_filt_size=5, sigma_gauss_size=10):
    """apply Median filter and Gaussian filter to fictrac quantities

    Parameters
    ----------
    x : numpy array
        time series to filter

    med_filt_size : int, optional
        size of median filter, by default 5

    sigma_gauss_size : int, optional
        width of Gaussian kernel, by default 10

    Returns
    -------
    numpy array
        filtered time series
    """
    return gaussian_filter1d(median_filter(x, size=med_filt_size), sigma=sigma_gauss_size)


def check_fictrac_results(trial_dirs, med_filt_size=5, sigma_gauss_size=10):

    to_fix = []
    for trial_dir in trial_dirs:
        trial_image_dir = os.path.join(trial_dir, "behData", "images")

        possible_fictrac_dats = glob.glob(os.path.join(trial_image_dir, "camera*.dat"))

        if len(possible_fictrac_dats) == 0:
            raise IOError(f"No file camera*.dat in {trial_image_dir}.")

        change_times = [os.stat(path).st_mtime for path in possible_fictrac_dats]
        most_recent_fictrac_dat = possible_fictrac_dats[np.argmax(change_times)]

        fictrac_df = pd.read_csv(most_recent_fictrac_dat, header=None, names=col_names)
        v_raw = fictrac_df["animal_movement_speed"] * f_s * r_ball
        v_filtered = filter_fictrac(v_raw, med_filt_size, sigma_gauss_size)
        if np.quantile(v_filtered,0.99) > 20:
            to_fix.append(trial_dir)
    return to_fix
        


def get_fictrac_df(trial_dir, index_df=None, df_out_dir=None, med_filt_size=5, sigma_gauss_size=10):
    """Read the output of fictrac, convert it into physical units and save it in dataframe.
    If index_df is supplied, fictrac results will be added to this dataframe.

    Parameters
    ----------
    trial_dir : str
        trial directory

    index_df : pandas Dataframe or str, optional
        pandas dataframe or path of pickle containing dataframe to which the fictrac result is added.
        This could, for example, be a dataframe that contains indices for synchronisation with 2p data,
        by default None

    df_out_dir : str, optional
        if specified, will save the dataframe as .pkl, by default None

    med_filt_size : int, optional
        size of median filter applied to velocity and orientation, by default 5

    sigma_gauss_size : int, optional
        width of Gaussian kernel applied to velocity and orientation, by default 10

    Returns
    -------
    pandas DataFrame
        dataframe containing the output of fictrac

    Raises
    ------
    IOError
        If fictract output file cannot be located

    ValueError
        If the length of the specified index_df and the fictrac output do not match
    """
    # partially adapted from Florian: https://github.com/NeLy-EPFL/ABO_data_processing/blob/master/fictrac_sync_odor.py
    if isinstance(index_df, str) and os.path.isfile(index_df):
        index_df = pd.read_pickle(index_df)
    if index_df is not None:
        assert isinstance (index_df, pd.DataFrame)

    trial_image_dir = os.path.join(trial_dir, "behData", "images")

    possible_fictrac_dats = glob.glob(os.path.join(trial_image_dir, "camera*.dat"))
    
    if len(possible_fictrac_dats) == 0:
        raise IOError(f"No file camera*.dat in {trial_image_dir}.")
    
    change_times = [os.stat(path).st_mtime for path in possible_fictrac_dats]
    most_recent_fictrac_dat = possible_fictrac_dats[np.argmax(change_times)]

    print(f"Getting data from: {most_recent_fictrac_dat}")

    # col_names = np.arange(25) + 1
    fictrac_df = pd.read_csv(most_recent_fictrac_dat, header=None, names=col_names)

    fictrac_df["v_raw"] = fictrac_df["animal_movement_speed"] * f_s * r_ball # convert from rad/frame to rad/s and mm/s
    fictrac_df["th_raw"] = (fictrac_df["animal_movement_direction_lab"] - np.pi) / np.pi * 180
    fictrac_df["x"] = fictrac_df["integrated_lab_x"] * r_ball
    fictrac_df["y"] = fictrac_df["integrated_lab_y"] * r_ball
    fictrac_df["integrated_forward_movement"] *=  r_ball
    fictrac_df["integrated_side_movement"] *=  r_ball
    fictrac_df["delta_rot_lab_side"] *= r_ball * f_s
    fictrac_df["delta_rot_lab_forward"] *= r_ball * f_s
    fictrac_df["delta_rot_lab_turn"] *= r_ball * f_s / np.pi * 180

    fictrac_df["v"] = filter_fictrac(fictrac_df["v_raw"], med_filt_size, sigma_gauss_size)
    fictrac_df["th"] = filter_fictrac(fictrac_df["th_raw"], med_filt_size, sigma_gauss_size)


    fictrac_df = fictrac_df[["v_raw", "th_raw", "x", "y", "integrated_forward_movement",
                             "integrated_side_movement", "delta_rot_lab_side",
                             "delta_rot_lab_forward", "delta_rot_lab_turn", "v", "th","integrated_lab_heading"]]

    if index_df is not None:
        if len(index_df) != len(fictrac_df):
            if np.abs(len(index_df) - len(fictrac_df)) <=10:
                Warning("Number of Thorsync ticks and length of fictrac file do not match. \n"+\
                        "Thorsync has {} ticks, fictrac file has {} lines. \n".format(len(index_df), len(fictrac_df))+\
                        "Trial: "+ trial_dir)
                print("Difference: {}".format(len(index_df) - len(fictrac_df)))
                length = np.minimum(len(index_df), len(fictrac_df))
                index_df = index_df.iloc[:length, :]
                fictrac_df = fictrac_df.iloc[:length, :]
            else:
                #raise ValueError("Number of Thorsync ticks and length of fictrac file do not match. \n"+\
                #        "Thorsync has {} ticks, fictrac file has {} lines. \n".format(len(index_df), len(fictrac_df))+\
                #        "Trial: "+ trial_dir)
                print("Number of Thorsync ticks and length of fictrac file do not match. \n"+\
                        "Thorsync has {} ticks, fictrac file has {} lines. \n".format(len(index_df), len(fictrac_df))+\
                        "Trial: "+ trial_dir)
                return None
        df = index_df
        for key in list(fictrac_df.keys()):
            df[key] = fictrac_df[key].values
    else:
        df = fictrac_df

    if df_out_dir is not None:
        df.to_pickle(df_out_dir)
    return df

def get_dfs(trial_dirs, trial_processed_dirs, trial_names):
        
    
    for i_trial, (trial_dir, processed_dir, trial_name) \
        in enumerate(zip(trial_dirs, trial_processed_dirs, trial_names)):
        
        print(time.ctime(time.time()), " creating data frames: " + trial_dir)

        trial_info = {"Date": self.date,
                    "Genotype": self.genotype,
                    "Fly": self.fly,
                    "TrialName": trial_name,
                    "i_trial": self.selected_trials[i_trial]
                    }
        if not os.path.isdir(processed_dir):
            os.makedirs(processed_dir)
        
        twop_out_dir = os.path.join(processed_dir, self.params.twop_df_out_dir)
        
        _1, _2, _3 = get_synchronised_trial_dataframes(
            trial_dir,
            crop_2p_start_end=0,#self.params.denoise_params.pre_post_frame,
            beh_trial_dir=self.beh_trial_dirs[i_trial],
            sync_trial_dir=self.sync_trial_dirs[i_trial],
            trial_info=trial_info,
            opflow=False,
            df3d=False,
            opflow_out_dir=opflow_out_dir,
            df3d_out_dir=df3d_out_dir,
            twop_out_dir=twop_out_dir
        )
        
        _ = get_fictrac_df(self.beh_trial_dirs[i_trial],
                           index_df=df3d_out_dir,
                           df_out_dir=df3d_out_dir)

            
# missing in automatically generated config file:
"""
max_bad_frames   : -1
opt_bound        : 0.350000
opt_do_global    : n
opt_max_err      : -1.000000
opt_max_evals    : 50
opt_tol          : 0.001000
roi_c            : { -0.000151, 0.016618, 0.999862 } --> automatically generated by calling gui
roi_r            : 0.029995 --> automatically generated by calling gui
save_debug       : n
save_raw         : n
src_fps          : -1.000000
thr_ratio        : 1.250000
thr_win_pc       : 0.250000
"""
