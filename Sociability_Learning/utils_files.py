import os
import yaml
import numpy as np
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

def get_cached_data_dir():
    d = Path(__file__).parent.parent / "outputs" / "data"
    d.mkdir(exist_ok=True, parents=True)
    return d

def read_data(h5_path, n_frames_init=80, interpolate=True, flip=False, **kwargs):
    """Load tracking data from h5 file to dataframe.

    Parameters
    ----------
    h5_path
        Path to the h5 file.
    n_frames_init : int
        Number of frames (from the start of the video).
    interpolate : bool
        Whether to fill missing values by interpolation.
    as_mm : bool
        Whether to convert units from pixel to mm.
    kwargs : dict
        Keyword arguments passed to DataFrame.

    Returns
    -------
    DataFrame
        Dataframe containing the data.
    """
    import h5py

    with h5py.File(h5_path) as file:
        data = file["tracks"][:].transpose(3, 0, 2, 1)
        if flip:
            data[:, 0] = -data[:, 0]
        nodes = np.vectorize(lambda x: x.decode())(file["node_names"][:])

    # sort flies by their initial x coordinates
    x_thorax = data[:, :, 0, 0].T
    x_thorax = np.array([i[np.isfinite(i)][:n_frames_init].mean() for i in x_thorax])
    data = data[:, x_thorax.argsort()]
    columns = pd.MultiIndex.from_product(
        [["l", "r"], nodes, ["x", "y"]], names=["fly", "node", "coord"]
    )

    df = pd.DataFrame(data.reshape((len(data), -1)), columns=columns, **kwargs)
    df.sort_index(axis=1, inplace=True)
    if interpolate:
        df.interpolate(inplace=True)

    return df


def get_10min_control_dataset(base_data_dir=None, cached_data_path=None):
    from datetime import datetime
    from Sociability_Learning.utils_embedding import xy2c

    if base_data_dir is None:
        base_data_dir = Path(
            "/mnt/upramdya_files/LOBATO_RIOS_Victor/"
            "Experimental_data/Optogenetics/Optobot/"
        )
    else:
        base_data_dir = Path(base_data_dir)

    if cached_data_path is None:
        cached_data_path = get_cached_data_dir() / "10min_control.h5"

    cached_data_path = Path(cached_data_path)

    if not cached_data_path.exists():

        def parse_arena_dir(data_path: Path):
            mapping = {
                "grouped": "g",
                "isolated": "i",
                "gro-gro": "g",
                "iso-iso": "i",
            }
            parts = data_path.relative_to(base_data_dir).parts

            return {
                "datetime": datetime.strptime(parts[2] + parts[3][:6], "%y%m%d%H%M%S"),
                "arena": int(parts[-1][-1]),
                "condition": mapping[parts[1]],
                "path": data_path.as_posix(),
            }

        def get_df(row):
            df = pd.read_pickle(Path(row.path) / "proximity_events.pkl")
            df = df[["Start", "Stop", "Ind min dist"]]
            df.drop_duplicates(inplace=True)
            df.columns = ["start", "stop", "rel_ind_min_dist"]
            df["start"] = df["start"].astype(int)
            df["stop"] = df["stop"].astype(int)
            df["condition"] = row.condition
            df["datetime"] = row.datetime
            df["arena"] = row.arena
            df["ind_min_dist"] = np.array(df["start"] + df["rel_ind_min_dist"]).astype(
                int
            )
            return df[
                ["condition", "datetime", "arena", "start", "stop", "ind_min_dist"]
            ]

        cached_data_path.parent.mkdir(exist_ok=True, parents=True)

        arena_dirs = [
            i.parent
            for i in sorted(
                base_data_dir.glob("control/*-*/*/*/*/proximity_events.pkl")
            )
        ]

        df = {}
        df["arenas"] = pd.DataFrame([parse_arena_dir(i) for i in arena_dirs])
        df["arenas"].sort_values(["datetime", "arena"], inplace=True)
        df["clips"] = pd.concat(
            [get_df(i) for i in df["arenas"].itertuples()], ignore_index=True
        )
        df["data"] = {}

        for row in df["arenas"].itertuples():
            h5_path = next(Path(row.path).glob("*-analysis.h5"))
            df["data"][row.datetime, row.arena] = xy2c(read_data(h5_path))

        df["data"] = pd.concat(df["data"], names=["datetime", "arena"])

        df["arenas"].set_index(["datetime", "arena"], inplace=True)
        df["clips"].set_index(["datetime", "arena"], inplace=True)
        df["clips"]["type"] = ""
        df["clips"]["auc"] = np.nan
        df["clips"]["below_threshold"] = False
        df["clips"]["i"] = np.arange(len(df["clips"]))

        thresholds = dict()

        for arena in df["arenas"].itertuples():
            arena_dir = Path(arena.path)
            paths = dict(
                distancing=arena_dir / "distancig_events.pkl",
                standstill=arena_dir / "standstill_events.pkl",
            )
            df_clips = df["clips"].loc[arena.Index].set_index("ind_min_dist")

            for k, p in paths.items():
                if not p.exists():
                    continue

                df_ = pd.read_pickle(p)

                if not len(df_):
                    continue

                ind_min_dist = df_["Start"].values + df_["Ind min dist"].values
                index = df_clips.loc[ind_min_dist, "i"].values
                thr = df_["Control threshold"].unique().item()
                if k not in thresholds:
                    thresholds[k] = thr
                else:
                    assert thresholds[k] == thr

                j = (df["clips"].columns == "type").argmax()
                df["clips"].iloc[index, j] = k
                j = (df["clips"].columns == "auc").argmax()
                df["clips"].iloc[index, j] = df_["AUC"].values
                j = (df["clips"].columns == "below_threshold").argmax()
                df["clips"].iloc[index, j] = df_["AUC"].values < thr

        df["clips"].drop(columns=["i"], inplace=True)
        df["threshold"] = pd.Series(thresholds)

        with pd.HDFStore(cached_data_path, "w") as store:
            for k, v in df.items():
                store[k] = v

    with pd.HDFStore(cached_data_path, "r") as store:
        df = {k.lstrip("/"): store[k] for k in store.keys()}

    return df


def get_learning_dark_dataset(base_data_dir=None, cached_data_path=None):
    from datetime import datetime

    if base_data_dir is None:
        base_data_dir = Path(
            "/mnt/upramdya_files/LOBATO_RIOS_Victor/"
            "Experimental_data/Optogenetics/Optobot/"
        )
    else:
        base_data_dir = Path(base_data_dir)
    
    if cached_data_path is None:
        cached_data_path = get_cached_data_dir() / "learning_dark.h5"

    if not cached_data_path.exists():

        def parse_arena_dir(data_path: Path):
            mapping = {
                "gro-gro": "g",
                "iso-iso": "i",
            }
            parts = data_path.relative_to(base_data_dir).parts

            return {
                "datetime": datetime.strptime(parts[2] + parts[3][:6], "%y%m%d%H%M%S"),
                "arena": int(parts[-1][-1]),
                "condition": mapping[parts[1]],
                "path": data_path.as_posix(),
            }

        def get_df(row):
            df = pd.read_pickle(Path(row.path) / "proximity_events.pkl")
            df = df[["Start", "Stop", "Ind min dist"]]
            df.drop_duplicates(inplace=True)
            df.columns = ["start", "stop", "rel_ind_min_dist"]
            df["start"] = df["start"].astype(int)
            df["stop"] = df["stop"].astype(int)
            df["condition"] = row.condition
            df["datetime"] = row.datetime
            df["arena"] = row.arena
            df["ind_min_dist"] = np.array(df["start"] + df["rel_ind_min_dist"]).astype(
                int
            )
            return df[
                ["condition", "datetime", "arena", "start", "stop", "ind_min_dist"]
            ]

        arena_dirs = [
            i.parent
            for i in sorted(
                base_data_dir.glob("learning-dark/**/arena*/proximity_events.pkl")
            )
        ]

        df = {}
        df["arenas"] = pd.DataFrame([parse_arena_dir(i) for i in arena_dirs])
        df["arenas"].sort_values(["datetime", "arena"], inplace=True)
        df["clips"] = pd.concat(
            [get_df(i) for i in df["arenas"].itertuples()], ignore_index=True
        )

        df["arenas"].set_index(["datetime", "arena"], inplace=True)
        df["clips"].set_index(["datetime", "arena"], inplace=True)
        df["clips"]["type"] = ""
        df["clips"]["auc"] = np.nan
        df["clips"]["below_threshold"] = False
        df["clips"]["i"] = np.arange(len(df["clips"]))

        thresholds = dict()

        for arena in df["arenas"].itertuples():
            arena_dir = Path(arena.path)
            paths = dict(
                distancing=arena_dir / "distancig_events.pkl",
                standstill=arena_dir / "standstill_events.pkl",
            )
            df_clips = df["clips"].loc[arena.Index].set_index("ind_min_dist")

            for k, p in paths.items():
                if not p.exists():
                    continue

                df_ = pd.read_pickle(p)

                if not len(df_):
                    continue

                ind_min_dist = df_["Start"].values + df_["Ind min dist"].values
                index = df_clips.loc[ind_min_dist, "i"].values
                thr = df_["Control threshold"].unique().item()
                if k not in thresholds:
                    thresholds[k] = thr
                else:
                    assert thresholds[k] == thr

                j = (df["clips"].columns == "type").argmax()
                df["clips"].iloc[index, j] = k
                j = (df["clips"].columns == "auc").argmax()
                df["clips"].iloc[index, j] = df_["AUC"].values
                j = (df["clips"].columns == "below_threshold").argmax()
                df["clips"].iloc[index, j] = df_["AUC"].values < thr

        df["clips"].drop(columns=["i"], inplace=True)
        df["threshold"] = pd.Series(thresholds)

        with pd.HDFStore(cached_data_path, "w") as store:
            for k, v in df.items():
                store[k] = v

    with pd.HDFStore(cached_data_path, "r") as store:
        df = {k.lstrip("/"): store[k] for k in store.keys()}

    return df


def get_learning_mesh_dataset(base_data_dir=None, cached_data_path=None):
    from datetime import datetime

    if base_data_dir is None:
        base_data_dir = Path(
            "/mnt/upramdya_files/LOBATO_RIOS_Victor/"
            "Experimental_data/Optogenetics/Optobot/"
        )
    else:
        base_data_dir = Path(base_data_dir)

    if cached_data_path is None:
        cached_data_path = get_cached_data_dir() / "learning_mesh.h5"

    if not cached_data_path.exists():

        def parse_arena_dir(data_path: Path):
            mapping = {
                "gro-gro": "g",
                "iso-iso": "i",
            }
            parts = data_path.relative_to(base_data_dir).parts

            return {
                "datetime": datetime.strptime(parts[2] + parts[3][:6], "%y%m%d%H%M%S"),
                "arena": int(parts[-1][-1]),
                "condition": mapping[parts[1]],
                "path": data_path.as_posix(),
            }

        def get_df(row):
            df = pd.read_pickle(Path(row.path) / "proximity_events.pkl")
            df = df[["Start", "Stop", "Ind min dist"]]
            df.drop_duplicates(inplace=True)
            df.columns = ["start", "stop", "rel_ind_min_dist"]
            df["start"] = df["start"].astype(int)
            df["stop"] = df["stop"].astype(int)
            df["condition"] = row.condition
            df["datetime"] = row.datetime
            df["arena"] = row.arena
            df["ind_min_dist"] = np.array(df["start"] + df["rel_ind_min_dist"]).astype(
                int
            )
            return df[
                ["condition", "datetime", "arena", "start", "stop", "ind_min_dist"]
            ]

        arena_dirs = [
            i.parent
            for i in sorted(
                base_data_dir.glob("learning-mesh/**/arena*/proximity_events.pkl")
            )
        ]

        df = {}
        df["arenas"] = pd.DataFrame([parse_arena_dir(i) for i in arena_dirs])
        df["arenas"].sort_values(["datetime", "arena"], inplace=True)
        df["clips"] = pd.concat(
            [get_df(i) for i in df["arenas"].itertuples()], ignore_index=True
        )

        df["arenas"].set_index(["datetime", "arena"], inplace=True)
        df["clips"].set_index(["datetime", "arena"], inplace=True)
        df["clips"]["type"] = ""
        df["clips"]["auc"] = np.nan
        df["clips"]["below_threshold"] = False
        df["clips"]["i"] = np.arange(len(df["clips"]))

        thresholds = dict()

        for arena in df["arenas"].itertuples():
            arena_dir = Path(arena.path)
            paths = dict(
                distancing=arena_dir / "distancig_events.pkl",
                standstill=arena_dir / "standstill_events.pkl",
            )
            df_clips = df["clips"].loc[arena.Index].set_index("ind_min_dist")

            for k, p in paths.items():
                if not p.exists():
                    continue

                df_ = pd.read_pickle(p)

                if not len(df_):
                    continue

                ind_min_dist = df_["Start"].values + df_["Ind min dist"].values
                index = df_clips.loc[ind_min_dist, "i"].values
                thr = df_["Control threshold"].unique().item()
                if k not in thresholds:
                    thresholds[k] = thr
                else:
                    assert thresholds[k] == thr

                j = (df["clips"].columns == "type").argmax()
                df["clips"].iloc[index, j] = k
                j = (df["clips"].columns == "auc").argmax()
                df["clips"].iloc[index, j] = df_["AUC"].values
                j = (df["clips"].columns == "below_threshold").argmax()
                df["clips"].iloc[index, j] = df_["AUC"].values < thr

        df["clips"].drop(columns=["i"], inplace=True)
        df["threshold"] = pd.Series(thresholds)

        with pd.HDFStore(cached_data_path, "w") as store:
            for k, v in df.items():
                store[k] = v

    with pd.HDFStore(cached_data_path, "r") as store:
        df = {k.lstrip("/"): store[k] for k in store.keys()}

    return df
