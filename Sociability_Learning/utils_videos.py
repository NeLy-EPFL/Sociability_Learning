import os
import scipy
import shutil
import random
import cv2 as cv
import numpy as np


def get_raw_images(video_path, num_imgs=np.inf, start=0, gray=False):
    raw_imgs=[]
    cap = cv.VideoCapture(video_path)
    
    if cap.isOpened() == False:
        raise Exception('Video file cannot be read! Please check in_path to ensure it is correctly pointing to the video file')
    cap.set(cv.CAP_PROP_POS_FRAMES, start)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True and len(raw_imgs)<num_imgs:
            if gray:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            raw_imgs.append(frame)
        else:
            break
    
    return cap, raw_imgs

def get_video_writer(out_path, width, height, fps):
    # Video writer class to output video with contour and centroid of tracked object(s)
    # Make sure the frame size matches size of array 'final'
    #fourcc = int(cap.get(6))
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    output_framesize = (width, height)
    out = cv.VideoWriter(filename = out_path,
                         fourcc = fourcc,
                         fps=fps,
                         frameSize = output_framesize,
                         isColor = True)
    
    return out

def split_arenas(exp_paths, only_analysis, batch_size=24000, frame_gap=30):
    new_paths = []
    with tqdm(total=len(exp_paths),desc="Splitting arenas") as pbar:
        for path in exp_paths:
            all_frames=False
            exists = False
            for f in os.listdir(path):
                if 'arena' in f:
                    new_paths.append(os.path.join(path,f))
                    exists = True
            if not exists and not only_analysis:
                vid_files = [vid_file for vid_file in os.listdir(path) if 'fps.' in vid_file]
                if vid_files:
                   vidName = vid_files[0]
                else:
                    #raise Exception("Experiment's raw video not found")
                    print("Experiment's raw video not found")
                    continue

                vid_path = os.path.join(path, vidName)
                cont = 0
                to_skip = []
                
                while not all_frames:
                    init_frame = cont*batch_size
                    cap, raw_imgs = get_raw_images(vid_path, num_imgs=batch_size, start=init_frame)
                    fps_video = int(cap.get(5))
                    if raw_imgs:
                        if cont == 0:
                            coords = get_coords(raw_imgs)
                            grabbers, crops = get_grabbers_from_coords(coords,
                                                                       path,
                                                                       vidName,
                                                                       raw_imgs[0],
                                                                       fps_video,
                                                                       frame_gap)
                            to_skip = get_arenas_to_skip(raw_imgs,
                                                         crops,
                                                         fps_video)
                            
                        grabbers = write_frames(raw_imgs,
                                                grabbers,
                                                crops,
                                                to_skip)
                        
                        cont += 1
                        if len(raw_imgs) < batch_size:
                            all_frames = True
                    else:
                        all_frames = True
                    cap.release()
                
                for num_arena, video in enumerate(grabbers):
                    video.release()
                    folder_path = os.path.join(path, f'arena{num_arena+1}')
                    if num_arena in to_skip:                        
                        shutil.rmtree(folder_path)
                    else:
                        new_paths.append(folder_path)
                del cap
            pbar.update(1)
                
    return new_paths

def get_coords(raw_imgs):
    sum_img = raw_imgs[-1]

    for i in range(-50,-1):
        sum_img = cv.addWeighted(sum_img, 0.5, raw_imgs[i], 0.5, 0)

    walls, arenas = find_arenas(sum_img)

    output = cv.connectedComponentsWithStats(arenas, 4, cv.CV_32S)
    stats = output[2]

    return stats[1:]

def find_arenas(img):
    height, width, _ = img.shape
    dim1 = int(np.ceil(0.015*width))
    dim2 = int(np.ceil(0.05*width))
    gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    
    hist = cv.calcHist([gray],[0],None,[256],[0,256])
    diff = np.diff(hist.flatten())
    sort = np.argsort(diff)
    th = np.mean(sort[-2:])
    ret, thresh = cv.threshold(gray, int(th), 255, cv.THRESH_BINARY_INV)

    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (dim1, dim1))
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (dim2, dim2))
    thresh_clean = cv.dilate(thresh, kernel1 ,iterations = 1)
    walls = cv.morphologyEx(thresh_clean, cv.MORPH_CLOSE, kernel2)
    arenas = cv.bitwise_not(walls)

    #img = cv.addWeighted(gray, 0.5, arenas, 0.5, 0)
    #cv.imshow('img', img)
    #cv.imshow('walls', walls)
    #cv.imshow('clean', arenas)
    #cv.waitKey()

    return walls, arenas


def get_grabbers_from_coords(coords, path, vidName, raw_frame, fps, frame_gap):
    grabbers = []
    crops = []

    for i, coord in enumerate(coords):
        folder_name = f'arena{i+1}'
        folder_path = os.path.join(path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        video_name = f'{folder_name}_{vidName}'
        video_path = os.path.join(folder_path, video_name)
        left = coord[0] - frame_gap
        right = coord[0] + coord[2] + frame_gap
        top = coord[1] - frame_gap
        bottom = coord[1] + coord[3] + frame_gap
        height_raw, width_raw, _ = raw_frame.shape
        x1 = left if left > 0 else 0
        x2 = right if right < width_raw-1 else width_raw-1
        y1 = top if top > 0 else 0
        y2 = bottom if bottom < height_raw-1 else height_raw-1
        if (x2-x1)%2 > 0:
            x2 -= 1
        if (y2-y1)%2 > 0:
            y2 -= 1

        width = int(x2-x1)
        height = int(y2-y1)
        
        out = get_video_writer(video_path, width, height, fps)

        grabbers.append(out)
        crops.append([x1, x2, y1, y2])

    return grabbers, crops

def get_arenas_to_skip(raw_imgs, crops,fps):
    to_skip = []
    random_list = random.sample(range(int(len(raw_imgs)/3), len(raw_imgs)), fps*1)
    for i, coords in enumerate(crops):
        x1 = coords[0]
        x2 = coords[1]
        y1 = coords[2]
        y2 = coords[3]
        sum_img = np.ones((y2-y1,x2-x1), dtype=np.uint8)*255
        for j in random_list:
            img = raw_imgs[j][y1:y2, x1:x2].copy()
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(img,(5,5),0)
            _, th = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            sum_img = cv.bitwise_and(sum_img,th)
        if np.mean(sum_img) > 125:
            to_skip.append(i)
        #cv.imshow("th",sum_img)
        #cv.waitKey()
        #cv.destroyAllWindows()
    return to_skip

def write_frames(raw_imgs, grabbers, crops, to_skip):
    for i, coords in enumerate(crops):
        if i not in to_skip:
            x1 = coords[0]
            x2 = coords[1]
            y1 = coords[2]
            y2 = coords[3]
            for img in raw_imgs:
                crop = img[y1:y2, x1:x2].copy()
                grabbers[i].write(crop)
  
    return grabbers
