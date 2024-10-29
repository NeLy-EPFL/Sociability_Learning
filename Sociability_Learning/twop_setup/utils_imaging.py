import os
import cv2
import utils2p
import numpy as np
from scipy.ndimage.filters import median_filter

green_com_warped = "green_com_warped.tif"
red_com_warped = "green_com_warped.tif"
green_denoised = "green_denoised.tif"
dff_mask = "dff_mask.tif"
dff_baseline_name = "dff_baseline.tif"
dff_name = "dff.tif"

def denoise_images(processed_dir):       
    
    green_warped = utils2p.load_img(os.path.join(processed_dir, green_com_warped))
   
    denoised_stack = np.zeros_like(green_warped)
    for i, img in enumerate(green_warped):
        denoised_stack[i,:,:] = cv2.medianBlur(img, 5)

    #cv2.imshow("sum",sum_stack)
    #cv2.waitKey()
    utils2p.save_img(os.path.join(processed_dir, green_denoised), denoised_stack)


def get_mask_from_red(processed_dir):
    red_name =os.path.join(processed_dir, red_com_warped)
    stack = utils2p.load_img(red_name)
    all_masks = []
    for i, img in enumerate(stack):
        denoised_img = cv2.medianBlur(img, 5)

        img_8uint = (denoised_img // 256).astype(np.uint8)
        equ = cv2.equalizeHist(img_8uint)
        _, img_th = cv2.threshold(equ, 235, 255, cv2.THRESH_BINARY)

        output = cv2.connectedComponentsWithStats(img_th, 4, cv2.CV_32S)
        stats = np.transpose(output[2])
        sizes = stats[4]

        labels = np.where(sizes > 25)[0]

        h, w = img_th.shape
        mask = np.zeros((h,w),np.uint8)
        for l in labels[1:]:
            mask[np.where(output[1] == l)] = 255
        all_masks.append(mask)

    mask_dir = os.path.join(processed_dir, dff_mask)
    utils2p.save_img(mask_dir, np.array(all_masks))


def quantile_filt_baseline(stack, quantile, masks=[]): 
    stack_filt = []
    if len(masks)>0:
        for (frame, mask) in zip(stack, masks):
            frame_masked = frame.copy()
            frame_masked[mask==0] = 32768
            stack_filt.append(frame_masked)
            #cv2.imshow("masked",frame_masked)
            #cv2.waitKey()
    else:
        stack_filt = stack    
    return np.quantile(stack_filt, q=quantile, axis=0)


def get_fly_dir(trial_dirs):
    path1 = trial_dirs[0].split("/")
    path2 = trial_dirs[-1].split("/")

    common_path = []
    
    for p1, p2 in zip(path1, path2):
        if p1 == p2:
            common_path.append(p1)
        else:
            break
    
    return "/".join(common_path)


def compute_dff_baseline(stacks,
                         masks,
                         baseline_quantile=0.05,
                         min_baseline=0,
                         baseline_dir=None):
    
    stacks = [np.clip(stack, 0, None) for stack in stacks]
    stacks_cat = np.concatenate(stacks, axis=0)

    masks = [utils2p.load_img(mask) for mask in masks]
    masks_cat = np.concatenate(masks, axis=0)
        
    dff_baseline = quantile_filt_baseline(stacks_cat, baseline_quantile, masks=masks_cat)   
    
    if min_baseline is not None:
        dff_baseline[dff_baseline <= min_baseline] = 0

    if baseline_dir is not None:
        utils2p.save_img(baseline_dir, dff_baseline)

    return dff_baseline


        
def compute_dff_trial(stack, baseline, dff_out_dir):

    dff_img = (
        np.divide(
            (stack - baseline),
            baseline,
            out=np.zeros_like(stack, dtype=np.double),
            where=(baseline != 0),
        )
        * 100
    )
    dff = median_filter(dff_img, (3, 3, 3))

    dff = np.clip(dff, 0, None)
    
    utils2p.save_img(dff_out_dir, dff)

        

def get_dff(trial_processed_dirs):
    
    stacks = [utils2p.load_img(os.path.join(processed_dir, green_denoised)) for processed_dir in trial_processed_dirs]
    
    masks = [os.path.join(processed_dir, dff_mask) for processed_dir in trial_processed_dirs] 
    
    print("Computing dff baseline")

    fly_processed_dir = get_fly_dir(trial_processed_dirs)
    baseline_dir = os.path.join(fly_processed_dir, "processed", dff_baseline_name)
    
    baseline = compute_dff_baseline(stacks=stacks,
                                    masks=masks,            
                                    baseline_dir=baseline_dir)


    for trial, processed_dir in enumerate(trial_processed_dirs):
        print("Computing dff for trial: " + processed_dir)
        dff_out_dir = os.path.join(processed_dir, dff_name)
        compute_dff_trial(stacks[trial], baseline, dff_out_dir)
