import sys
import numpy as np
import pandas as pd
from Sociability_Learning.metrics import *
from Sociability_Learning.utils_files import *
from Sociability_Learning.utils_plots import plot_single_metric
from Sociability_Learning.utils_sleap import load_sleap_tracking

#control_path = "/mnt/upramdya_files/LOBATO_RIOS_Victor/Experimental_data/Optogenetics/Optobot/screening-TNT-learning/Empty-split"
control_path = "/mnt/upramdya_files/LOBATO_RIOS_Victor/Experimental_data/Optogenetics/Optobot/grouped-control/gro-gro"

results_file = "results/paper/241018_screens.pkl"
save_results = False
save_events = False
adaptation_time = 120#s
num_bins = 1
beetle = False

def main():
    
    usr_input = sys.argv[-1]
    folders = get_folders_from_file(usr_input)
    
    sociability_df = pd.DataFrame()
    discarded_df = pd.DataFrame()
    
    control_thresholds = get_thresholds_from_controls(control_path)
    #print(control_thresholds)
    
    for i, folder in enumerate(folders):
        all_experiments = get_experiments(folder)
        for j, path_exp in enumerate(all_experiments):
            print(f"({i+1} / {len(folders)}) {j+1} / {len(all_experiments)}")

            exp = Experiment(path_exp)
            data_path = os.path.join(exp.folder, exp.vidName)

            try:
                locations, num_flies, fps = load_sleap_tracking(data_path, second_tracking=beetle)#, max_frames=48000)
                total_exp_time = len(locations["fly0"]["thorax"]["locations"])/fps
            except Exception as e:
                print('Experiment exception: ' + str(e))
                discarded_data = pd.DataFrame({"Genotype": exp.gen_name,
                                               "Num events": np.nan,
                                               "Num flies": np.nan,
                                               "Exception": str(e),
                                               "Folder": exp.folder}, index=[0])
                discarded_df = discarded_df.append(discarded_data, ignore_index=True)
                continue

            if num_flies == 2 or beetle:
                kinematics={}
                orientation=[]
                discard_beetle = False
                for fly in locations.keys():
                    if "fly" in fly:
                        orientation = get_orientation(locations[fly]['thorax']["locations"],
                                                      locations[fly]['head']["locations"])   
                        
                        kinematics[fly] = get_kinematics(locations[fly],
                                                         fps,
                                                         orientation,
                                                         nodes=["thorax","head"])
                                        
                        kinematics[fly]["heading"] = orientation
                                        
                    else:
                        beetle_vel = get_kinematics(list(locations[fly]["thorax"]["locations"]),
                                                    fps,
                                                    orientation,
                                                    nodes=[])
                        low_speed = np.where(beetle_vel<5)[0]                       
                        
                        if len(low_speed)/len(beetle_vel)>0.9:
                            print("Discarding experiment because the beetle is not moving")
                            print(len(low_speed)/len(beetle_vel))
                            discard_beetle = True
                if discard_beetle:
                    proximity_events = []
                else:
                    proximity_events = get_proximity_events(exp,
                                                            locations,
                                                            fps,
                                                            dist_limit=5,#mm
                                                            gap_between_events=fps/4,#Frames
                                                            event_min_length=fps/2,
                                                            event_max_length=10*fps,
                                                            beetle=beetle,
                                                            save_pkl=save_events,
                                                            from_file=False,
                                                            omit_events=[0,adaptation_time*fps])#Frames

                print(f"Proximity events: {len(proximity_events)}")

                min_num_events = (total_exp_time-adaptation_time)/(60*2)
                
                if len(proximity_events)/num_flies < min_num_events: #len(proximity_events) == 0:
                    print("Discarding experiment because there aren't enough proximity events")
                    discarded_data = pd.DataFrame({"Genotype":exp.gen_name,
                                                   "Num events": len(proximity_events),
                                                   "Num flies": num_flies,
                                                   "Exception": "No",
                                                   "Folder":exp.folder}, index=[0])
                    discarded_df = discarded_df.append(discarded_data, ignore_index=True)
                    continue
                

                get_reaction_classes(proximity_events, kinematics)

                
                for fly in range(num_flies):
                    fly_values, sel_events = get_metrics(proximity_events,
                                                         kinematics,
                                                         locations[f"fly{fly}"],
                                                         control_thresholds,
                                                         exp.gen_name,
                                                         fly,
                                                         fps,
                                                         total_exp_time,
                                                         exp.folder,
                                                         bins=num_bins,
                                                         return_events="moving",
                                                         #return_events="stand",
                                                         adaptation_time=adaptation_time,
                                                         save_events=save_events)

                    sociability_df = sociability_df.append(fly_values, ignore_index=True)

            else:
                print(f"Num flies is: {num_flies}")
                discarded_data = pd.DataFrame({"Genotype":exp.gen_name,
                                               "Num events": np.nan,
                                               "Num flies": num_flies,
                                               "Exception": "No",
                                               "Folder":exp.folder}, index=[0])
                discarded_df = discarded_df.append(discarded_data, ignore_index=True)

    if save_results:
        sociability_df.to_pickle(results_file)
        discarded_df.to_pickle(results_file.replace(".pkl","_discarded.pkl"))
    else:
        all_median = sociability_df.groupby('Genotype')['Prop'].median()
        all_sorted_by_value = all_median.sort_values(ascending=True).index.tolist()
        plot_single_metric(sociability_df.copy(), "Genotype", "Prop", order=all_sorted_by_value)

if __name__ == "__main__":
    main()
