from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from decord import VideoReader
from imageio import get_writer
from Sociability_Learning.utils_files import get_learning_dark_dataset
from Sociability_Learning.utils_videos import draw_text
from tqdm import tqdm


def format_time(n_secs):
    dt = datetime.fromtimestamp(n_secs)
    t_str = dt.strftime("01:%M:%S.%f")[:-4]
    return t_str


n_rows_preview = 2
n_cols_preview = 4
n_seconds = 4
fps = 80
n_frames = n_seconds * fps
n_videos_preview = n_rows_preview * n_cols_preview
data_dir = Path("../data/learning-dark")
data_dir.mkdir(exist_ok=True, parents=True)
save_path = Path("../outputs/videos/vid6_learning_dark.mp4")
save_path.parent.mkdir(exist_ok=True, parents=True)
df = get_learning_dark_dataset()
groupby = (
    df["clips"]
    .query("type == 'distancing'")
    .groupby(["condition", "datetime", "arena"])
)

for key, df_ in tqdm(groupby, total=len(groupby)):
    mp4_path = list(Path(df["arenas"].loc[key[1:], "path"]).glob("*80fps.mp4"))[0]
    cap = cv2.VideoCapture(mp4_path.as_posix())
    frames = []

    for i in (
        df_.sort_values("auc", ascending=False)
        .head(n_videos_preview)["ind_min_dist"]
        .values
    ):
        frames.append([])
        cap.set(cv2.CAP_PROP_POS_FRAMES, i - n_frames // 2)
        for _ in range(n_frames):
            ret, frame = cap.read()
            if ret:
                frames[-1].append(frame[..., 0])
            else:
                frames[-1].append(np.zeros_like(frames[-1][-1]))

    frames = np.array(frames)
    frames = np.concatenate(
        np.concatenate(
            frames.reshape((n_rows_preview, n_cols_preview, *frames.shape[1:])), axis=2
        ),
        axis=2,
    )
    filename = data_dir / f"{key[0]}-{key[1].strftime('%Y%m%d%H%M%S')}-{key[2]}.mp4"
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    with get_writer(filename, fps=80, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    cap.release()
s = {}

s[
    "i"
] = """i-20230201140340-3 1
i-20230202171424-4 1
i-20230130173701-3 1
i-20230201140340-4 1"""

s[
    "g"
] = """g-20230130105949-4 3
g-20230131141555-2 1
g-20230131141555-4 1
g-20230201171926-1 1"""

all_frames = {}

for c, sc in s.items():
    sc = [((a := i.split(" "))[0], int(a[1])) for i in sc.split("\n")]
    all_frames_c = []
    hn, wn = 384, 416

    for filename, k in sc:
        vr = VideoReader((data_dir / f"{filename}.mp4").as_posix())
        frames = vr[:].asnumpy()[..., 0]
        T, H, W = frames.shape
        h = H // n_rows_preview
        w = W // n_cols_preview
        i = k // n_cols_preview
        j = k % n_cols_preview
        frames = frames[:, i * h : (i + 1) * h, j * w : (j + 1) * w]
        frames = np.array([cv2.resize(frame, (wn, hn)) for frame in frames])
        all_frames_c.append(frames)

    all_frames_c = np.array(all_frames_c)
    all_frames_c[:, hn - 1 : hn + 1] = 128
    all_frames_c[:, :, wn - 1 : wn + 1] = 128
    all_frames[c] = np.concatenate(
        np.concatenate(all_frames_c.reshape((2, 2, -1, hn, wn)), axis=2), axis=2
    )
titles = {
    "i": "Single housed flies in dark",
    "g": "Group housed flies in dark",
}

im_title = draw_text(
    np.zeros((32, wn * 4), dtype=np.uint8),
    titles["i"],
    wn,
    16,
    anchor="mm",
    font_size=28,
)
im_title = draw_text(
    im_title,
    titles["g"],
    wn * 3,
    16,
    anchor="mm",
    font_size=28,
)
all_frames = np.concatenate((all_frames["i"], all_frames["g"]), axis=-1)
with get_writer(save_path, fps=80) as writer:
    for j, im in enumerate(all_frames):
        im[hn - 1 : hn + 1] = 128
        im[:, wn - 1 : wn + 1] = 128
        im[:, wn * 3 - 1 : wn * 3 + 1] = 128

        im_time2 = draw_text(
            im_title.copy(),
            f"t = {j / 80 - 1:+0.2f} s".replace("-", "−"),
            wn * 4 - 2,
            16,
            anchor="rm",
            font_size=28,
        )
        im = np.row_stack([im_time2, im])
        im[:, wn * 2 - 1 : wn * 2 + 1] = 255

        writer.append_data(im)
