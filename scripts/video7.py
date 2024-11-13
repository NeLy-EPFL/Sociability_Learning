from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from decord import VideoReader
from imageio.v2 import get_writer, imread
from Sociability_Learning.utils_files import get_learning_mesh_dataset
from Sociability_Learning.utils_videos import draw_text
from tqdm import tqdm


def get_t_for_slowed_video(
    fps_in,
    fps_out,
    intervals_slow,
    intervals_fast,
    speed_slow=1,
    speed_fast=10,
):
    intervals_slow = np.reshape(intervals_slow, (-1, 2)) / fps_in
    intervals_fast = np.reshape(intervals_fast, (-1, 2)) / fps_in

    tlist = []
    is_slow = []

    for fast_interval in intervals_fast:
        t = fast_interval[0]

        while t < fast_interval[1]:
            if any((a <= t < b) for a, b in intervals_slow):
                t += 1 / fps_out * speed_slow
                is_slow.append(True)
            else:
                t += 1 / fps_out * speed_fast
                is_slow.append(False)

            tlist.append(t)

    return tlist, is_slow


def get_nearest_indices(t_ref, t_que):
    """Get the index of the closest element in t_ref for each element in t_que.

    Parameters
    ----------
    t_ref : array-like
        Reference time points.
    t_que : array-like
        Query time points.

    Returns
    -------
    idx : array-like
        Index of the closest element in t_ref for each element in t_que.
    """

    import numpy as np
    from sklearn.neighbors import KDTree

    t_ref = np.asarray(t_ref).reshape(-1, 1)
    t_que = np.asarray(t_que).reshape(-1, 1)

    tree = KDTree(t_ref)
    idx = tree.query(t_que, return_distance=False).ravel()
    return idx


def format_time(n_secs):
    if n_secs < 0:
        return format_time(-n_secs).replace("+", "−")

    dt = datetime.fromtimestamp(n_secs)
    t_str = dt.strftime("+%M:%S.%f")[:-4]
    return t_str


save_dir = Path("../outputs/videos/")
save_name = "Video7-MeshLearning"
fps_in = 80
fps_out = 80
n_seconds_before = 7.5
n_seconds_after = 30
n_frames = 864000
speed_slow = 1
speed_fast = 8
t_in = np.arange(n_frames) / fps_in

opening_gates = {
    "g": 574723,
    "i": 579670,
}
df = get_learning_mesh_dataset()
arenas = {cond: df_.index[-1] for cond, df_ in df["arenas"].groupby("condition")}
video_paths = {}

for k, v in arenas.items():
    paths = [
        i
        for i in Path(df["arenas"].loc[v, "path"]).parent.glob("*.mp4")
        if "interactions" not in i.stem
    ]
    assert len(paths) == 1
    video_paths[k] = paths[0].as_posix()


def iter_frames(cond):
    i0 = opening_gates[cond]
    ia = round(i0 - n_seconds_before * fps_in * speed_fast)
    ib = round(i0 + n_seconds_after * fps_in * speed_slow)

    intervals_slow = np.array([[i0, ib]])
    interval_fast = np.array([ia, ib])
    t_out, is_slow = get_t_for_slowed_video(
        fps_in,
        fps_out,
        intervals_slow,
        interval_fast,
        speed_slow,
        speed_fast,
    )
    idx = get_nearest_indices(t_in, t_out)
    t_out = t_out - t_in[i0]
    mp4_path = video_paths[cond]

    vr = VideoReader(mp4_path)
    h, w = vr[0].asnumpy().shape[:2]

    h -= 7
    h = h // 2
    w = w // 2
    title = "Single housed" if cond == "i" else "Group housed"
    im_title = draw_text(np.zeros((28, w), np.uint8), title, w // 2, 14, "mm")

    im_ff = imread("../data/fast_forward.png")[..., 3]
    hn, wn = 64, 64
    im_ff = cv2.resize(im_ff, (wn, hn))
    hpad = h - hn
    wpad = w - wn
    hpad = hpad // 2, hpad - hpad // 2
    wpad = wpad // 2, wpad - wpad // 2
    im_ff = np.pad(im_ff, (hpad, wpad))

    for i, ti, s_i in tqdm(zip(idx, t_out, is_slow), total=len(t_out)):
        im = vr[i].asnumpy()[7:, ..., 0]
        im = cv2.resize(im, (w, h))

        if cond == "i":
            if s_i:
                t_str = format_time(ti) + f" ({speed_slow}×)"
            else:
                t_str = format_time(ti) + f" ({speed_fast}×)"

            x = 4
            y = im.shape[0] - 4
            im = draw_text(im, t_str, x, y, anchor="lb", font_size=18)

        im = np.concatenate([im_title, im], axis=0)
        yield im


with get_writer((save_dir / save_name).with_suffix(".mp4"), fps=fps_out) as writer:
    iter_i = iter_frames("i")
    for im_i in iter_i:
        writer.append_data(im_i)
