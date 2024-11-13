from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from decord import VideoReader
from imageio import get_writer
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D
from mplex import Grid
from numpy.lib.stride_tricks import sliding_window_view as swv
from Sociability_Learning.utils_embedding import c2xy
from Sociability_Learning.utils_files import get_10min_control_dataset

vid_nums = {"distancing": 4, "standstill": 5}
titles = {
    "distancing": "Example distancing events (moving after closest distance)\n"
    "above and below distancing efficiency threshold ({:.2f})",
    "standstill": "Example standstill events (stationary after closest distance)\n"
    "above and below immobile freezing threshold ({:.2f})",
}

outputs_dir = Path("../outputs/")
size = 240
fps_in = 80
fps_out = 40
n_rows, n_cols = 2, 4
n_vids = n_rows * n_cols

df = get_10min_control_dataset()
df["clips"]["duration"] = df["clips"]["stop"] - df["clips"]["start"]
df["clips"]["before"] = df["clips"]["start"] - df["clips"]["ind_min_dist"]
df["clips"]["after"] = df["clips"]["stop"] - df["clips"]["ind_min_dist"]
intervals = swv(np.pad(df["clips"]["duration"].cumsum(), 1)[:-1], 2)
thresholds = dict(df["threshold"])
bodyparts = ["head", "thorax", "abdomen"]
optobot_dir = Path(
    "/mnt/upramdya_files/LOBATO_RIOS_Victor/Experimental_data/Optogenetics/Optobot/"
)


def get_video_path(datetime, arena, optobot_dir: Path):
    ymd = datetime.strftime("%y%m%d")
    hms = datetime.strftime("%H%M%S")
    return next(
        optobot_dir.glob(
            f"*-control/*-*/{ymd}/{hms}_s0a0_p0-0/arena{arena}/arena{arena}_*-exp_p0-0_80fps.mp4"
        )
    )


def init_fig(clip_data, event_type, lim=160, axsize=240):
    g = Grid(axsize, (n_rows, n_cols), facecolor="k", dpi=72, border=(0, 0, 116, 0))
    elems = []

    for i, ax in enumerate(g.axs.ravel()):
        d = clip_data[i]
        kw = dict(lw=2)
        elem = {
            "im": ax.imshow(d["frames"][0], cmap="gray", vmin=0, vmax=255),
            "l1": LineCollection(c2xy(swv(d["trajs"][0], 2)), color="m", **kw),
            "l2": LineCollection(c2xy(swv(d["trajs"][1], 2)), color="c", **kw),
        }

        elems.append(elem)
        ax.add_collection(elem["l1"])
        ax.add_collection(elem["l2"])
        ax.axis("off")
        s = f"${d['auc']:0.2f}$"
        ax.add_text(
            1, 1, s, ha="r", va="t", size=22, c="lightgray", transform="a", pad=(-1, -5)
        )

    txt_kw = dict(size=22, x=0, y=0.5, ha="l", transform="a", c="w")
    tit_lw = dict(
        x=0.5, y=1, size=30, c="w", transform="a", ha="c", va="_", pad=(0, 61)
    )

    ax0 = g.make_ax(behind=False)

    for j in range(1, 4):
        ax0.plot([j / 4, j / 4], [0, 1], transform=ax0.transAxes, c="gray", lw=1)

    ax0.plot([0, 1], [0.5, 0.5], transform=ax0.transAxes, c="w", lw=2, ls="--")
    ax0.add_text(s="Below threshold", va="t", pad=(0, -5), **txt_kw)
    ax0.add_text(s="Above threshold", va="b", pad=(0, 5), **txt_kw)
    ax0.add_text(s=titles[event_type].format(thresholds[event_type]), **tit_lw)
    time_text = ax0.add_text(
        0.5, 1, "", size=30, c="w", transform="a", ha="c", va="_", pad=(0, 16)
    )

    ax = g[0, 0]
    r = 0.07
    pad = 0.01
    xy = (r + pad, 1 - r + 116 / size - pad)
    circ = Circle(xy, r, fc="r", lw=0, transform=ax.transAxes, clip_on=False)
    circ.set_visible(False)
    ax.add_patch(circ)

    return g, elems, time_text, circ


clip_ids = {
    "distancing": [1179, 309, 229, 228, 912, 1078, 492, 15],
    "standstill": [70, 150, 1264, 36, 776, 513, 507, 691],
}
flip = {
    "distancing": [False, True, True, True, False, False, False, False],
    "standstill": [False, True, False, True, False, False, False, True],
}
for event_type in ["distancing", "standstill"]:

    if event_type == "distancing":
        t0 = -1 * fps_in
        t1 = 3 * fps_in + 1
    else:
        t0 = -2 * fps_in
        t1 = 4 * fps_in + 1

    clip_data = []

    for j, i in enumerate(clip_ids[event_type]):
        clip_i = df["clips"].iloc[i]

        s_t_frames = np.s_[clip_i.ind_min_dist + t0 : clip_i.ind_min_dist + t1]
        s_t = np.s_[
            clip_i.ind_min_dist + clip_i.before : clip_i.ind_min_dist + clip_i.after
        ]
        df_i = df["data"].loc[clip_i.name].iloc[s_t]
        tmin_i = -clip_i.before
        fly1, fly2 = {False: "lr", True: "rl"}[flip[event_type][j]]

        traj1 = df_i[fly1][bodyparts].mean(1).values
        traj2 = df_i[fly2][bodyparts].mean(1).values

        video_path = get_video_path(clip_i.name[0], clip_i.name[1], optobot_dir)

        clip_data.append(
            dict(
                frames=VideoReader(video_path.as_posix())[s_t_frames].asnumpy()[
                    ..., 7:, :, 0
                ],
                T=np.arange(clip_i.before, clip_i.after),
                duration=clip_i.duration,
                trajs=np.array((traj1, traj2)),
                auc=clip_i.auc,
            )
        )

    filename = outputs_dir / f"videos/vid{vid_nums[event_type]}_{event_type}.mp4"
    (outputs_dir / "videos").mkdir(parents=True, exist_ok=True)
    g, elems, time_text, circ = init_fig(clip_data, event_type)
    with get_writer(filename, fps=fps_out) as writer:
        for i_t, t in enumerate(range(t0, t1)):
            for d, e in zip(clip_data, elems):
                e["im"].set_data(d["frames"][i_t])

                def get_alphas(T, t, k=0.02):
                    alphas = np.zeros(d["duration"])
                    bidx = T <= t
                    alphas[bidx] = np.exp(k * (T[bidx] - t))
                    return alphas

                a = get_alphas(d["T"], t)
                e["l1"].set_alpha(a)
                e["l2"].set_alpha(a)

            circ.set_visible(-20 <= t < 20)
            time_text.set_text(f"t = ${t / fps_in:+0.2f}$ s (0.5×)")
            writer.append_data(g.to_rgba_array()[..., :3])

    plt.close()
