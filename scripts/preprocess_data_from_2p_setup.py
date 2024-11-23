import os
import sys
from Sociability_Learning.utils_files import *
from Sociability_Learning.twop_setup import *

def main():
    usr_input = sys.argv[-1]
    folders = get_folders_from_file(usr_input)
    
    for i, folder in enumerate(folders):
        all_experiments = get_experiments(folder, keyword="processed")
        for processed_dir in all_experiments:
            print(f"Processing {processed_dir}")
            denoise_images(processed_dir)
            get_mask_from_red(processed_dir)

        get_dff(all_experiments)
    
    for i, folder in enumerate(folders):
        all_experiments = get_experiments(folder, keyword="behData")
        get_free_fly_tracking(all_experiments)
        
        get_treadmill_tracking(all_experiments, overwrite=True)
        get_synchronized_dataframes(all_experiments)

if __name__ == "__main__":
    main()
