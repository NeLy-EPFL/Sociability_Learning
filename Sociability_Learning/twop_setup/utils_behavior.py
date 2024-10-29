import os
import random
import pickle
import pkgutil
import cv2 as cv
import numpy as np
from pathlib import Path
from Sociability_Learning.utils_videos import get_raw_images
from Sociability_Learning.utils_sleap import load_sleap_tracking


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
