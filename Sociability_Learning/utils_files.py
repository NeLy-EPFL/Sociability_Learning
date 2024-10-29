import os
import yaml
import pandas as pd
import pkgutil
from pathlib import Path

def get_folders_from_file(file_to_process):
    if os.path.isfile(file_to_process) and file_to_process.endswith('.yaml'):
        par = yaml.load( open( file_to_process, 'rb' ), Loader=yaml.FullLoader)
        if not par['lines']:
            dirs=[]
            for gen1 in par['parentfolder']:
                lines = next(os.walk(gen1))[1]
                for gen2 in lines:
                    if os.path.isdir(os.path.join(gen1, gen2)):
                        dirs.append(os.path.join(gen1, gen2))
        else:        
            dirs = [os.path.join(gen1, gen2) for gen1 in par['parentfolder'] for gen2 in par['lines'] if os.path.isdir(os.path.join(gen1, gen2))]
    else:
        raise Exception('No yaml file found.')

    if dirs==[]:
        raise Exception('No folders found.')

    return dirs

def get_experiments(folder, keyword="arena"):
    dates = next(os.walk(folder))[1] #folders

    experiments = []
    for date in dates: 
        exp_dirs = next(os.walk(os.path.join(folder, date)))[1] #folders
        for exp in exp_dirs:
            exp_full_path = os.path.join(folder, date, exp)
            for f in os.listdir(exp_full_path):
                if keyword in f:
                    experiments.append(os.path.join(exp_full_path,f))

    return experiments

class Experiment:
    def __init__(self, exp_dir):
        #Data got from the path
        self.folder = exp_dir
        ind = -2
        dir_path = exp_dir.split('/')
        self.gen1 = dir_path[ind-3]
        self.gen2 = dir_path[ind-2]
        self.date = dir_path[ind-1]
        self.get_genotype_name()
        arenaInfo = dir_path[ind].split('_')
        self.time = arenaInfo[0]
        self.arena = int(arenaInfo[1][-1])
        
        vid_files = [vid_file for vid_file in os.listdir(exp_dir) if vid_file[-7:-3] == 'fps.']
        if vid_files:
           self.vidName = vid_files[0]
        else:
            raise Exception("Experiment's raw video not found")

    def get_genotype_name(self):
        package = pkgutil.get_loader("Sociability_Learning")
        path_package = Path(package.get_filename())
        names_all_lines = os.path.join(path_package.parents[1],
                                       'data',
                                       'names_all_lines.csv')
        df_lines = pd.read_csv(names_all_lines, header=0)
        match = None
        new_gen2 = None
        for index, line in df_lines.iterrows():
            line_name = line["driver_line"]
            line_name_parts = line_name.split(" ")
            nickname1 = line_name_parts[0]
            nickname2 = None
            if len(line_name_parts)>1:
                nickname2 = line_name_parts[1][1:-1]
            if nickname1 == self.gen2 or nickname2 == self.gen2:
                if len(line_name_parts)>1:
                    new_gen2 = nickname2
                else:
                    new_gen2 = nickname1
                break

        if new_gen2 == None:
            new_gen2 = self.gen2

        self.gen_name = "_".join([self.gen1, new_gen2])
