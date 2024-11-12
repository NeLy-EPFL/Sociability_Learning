from pathlib import Path
from subprocess import run

import matplotlib.pyplot as plt
import numpy as np
from decord import VideoReader
from imageio.v2 import get_writer
from joblib import Parallel, delayed
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.transforms import Affine2D
from mplex import Grid
from numpy.lib.stride_tricks import sliding_window_view as swv
from Sociability_Learning.utils_embedding import c2xy
from Sociability_Learning.utils_files import get_10min_control_dataset
from tqdm import trange

save_dir = Path("../outputs/videos")
cache_video_path = Path("../data/10min_control_events.mp4")
save_dir.mkdir(exist_ok=True, parents=True)
cache_video_path.parent.mkdir(exist_ok=True, parents=True)
data_path = Path("../data/dist_vel_angle_20_clusters.npz")
assert data_path.exists(), "Run embedding.ipynb first"

df = get_10min_control_dataset()
win = int(0.25 * 80)


def pad(frames, new_shape):
    ho, wo = frames.shape[1:3]
    hn, wn = new_shape
    assert hn >= ho and wn >= wo
    return np.pad(frames, ((0, 0), (0, hn - ho), (0, wn - wo), (0, 0)))


new_shape = (386, 412)

if not cache_video_path.exists():
    with get_writer(cache_video_path, fps=80, macro_block_size=None) as writer:
        for key, df_ in df["clips"].groupby(["datetime", "arena"]):
            arena_dir = Path(df["arenas"].loc[key, "path"])
            video_paths = list(arena_dir.glob("arena*_80fps.mp4"))
            assert len(video_paths) == 1
            video_path = video_paths[0]
            vr = VideoReader(video_path.as_posix())

            for i in df_["ind_min_dist"].values:
                frames = pad(vr[i - win : i + win].asnumpy(), new_shape)

                for frame in frames:
                    writer.append_data(frame)

fps = 80
fps_out = 20
w = int(fps * 0.5)  # window length in frames
arange = np.arange(-w // 2, w // 2)

data = np.load(data_path)
labels = data["labels"]
need_flip = data["need_flip"]
im_regions = data["im_regions"]
F = data["F"]
Z = data["Z"]

n_clusters = labels.max() + 1
conds = df["clips"]["condition"].values

n_cols = 7
n_rows = 5
size = 160
n_videos = n_cols * n_rows
assert n_videos <= np.bincount(labels).min()

bound = 4.5


def iter_l(k):
    from matplotlib.colors import to_hex
    from matplotlib.patches import Rectangle
    from mplex.colors import change_hsv

    hpx = n_rows * size
    g = Grid(25, (6, 3), sharey="row", space=(2, 6), facecolor="k")
    g[:].set_visible_sides("")
    g[2:, 1].set_visible_sides("l")
    axs = np.concatenate([g.axs[2:, 1], g.axs[3:, 2]])

    exprs = ["$d$", "$v^\\text{ to.}$", "$v^\\perp$", "$\\theta$"]
    units = ["mm", "mm/s", "mm/s", "rad"]
    ylabels = [f"{e}\n({u})" for e, u in zip(exprs, units)]

    clist = list("wmmmccc")
    t = arange / fps
    Fk = F[labels == k]

    for i in range(F.shape[1]):
        ax = axs[i]
        ax.plot(t, Fk[:, i].T, c=change_hsv(clist[i], v=0.5), alpha=0.2, lw=0.5)
        ax.plot(t, Fk[:, i].mean(0), c=clist[i], lw=1)

        for sp in ax.spines.values():
            sp.set_color("w")
            ax.tick_params(axis="y", colors="w", labelsize=4)

    for ax, s in zip(axs, ylabels):
        ax.add_text(
            0, 0.5, s, c="w", transform="a", ha="c", va="c", size=5, pad=(-15, 0)
        )

    g[2, 1].set_ylim(0, 10)
    g[3, 1].set_ylim(-40, 40)
    g[4, 1].set_ylim(0, 40)
    g[5, 1].set_ylim(0, np.pi)

    g[2, 1].set_yticks([0, 10])
    g[3, 1].set_yticks([-40, 40])
    g[4, 1].set_yticks([0, 40])
    g[5, 0].set_yticks([0, np.pi], labels=["0", "$\\pi$"])

    for ax in g.axs[2:, 1]:
        label1, label2 = ax.get_yticklabels()
        label1.set_verticalalignment("baseline")
        label2.set_verticalalignment("top")

    ax = g[:2].make_ax()
    ax.set_xticks([])
    ax.set_yticks([])

    c = np.array([{"i": "C0", "g": "C1"}[i] for i in conds])
    ax.scatter(*Z[labels != k].T, s=2, c=c[labels != k], alpha=0.5, lw=0)
    ax.scatter(*Z[labels == k].T, s=2, c=c[labels == k], alpha=1, lw=0)
    ax.set_aspect("equal")
    ax.set_xmargin(0.12)
    ax.set_ymargin(0.12)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ylim = np.array(ylim) + 0.3
    ax.set_ylim(ylim)

    xmax, ymin = max(xlim) - 0.75, min(ylim) + 0.75
    dx, dy = 1, 1
    ax.plot([xmax, xmax - dx], [ymin, ymin], c="w")
    ax.plot([xmax, xmax], [ymin, ymin + dy], c="w")
    ax.add_text(
        xmax - dx / 2, ymin, "UMAP 1", ha="c", va="t", color="w", size=3, pad=(0, -2)
    )
    xy = (xmax, ymin + dy)
    ax.add_text(*xy, "UMAP 2", ha="l", va="c", c="w", size=3, pad=(2, 0), rotation=90)

    ax.contour(
        im_regions + 1,
        levels=np.arange(n_clusters + 1),
        colors=to_hex((0.3,) * 3),
        extent=(-bound, bound, bound, -bound),
        antialiased=True,
        linewidths=0.5,
    )

    ax.contour(
        im_regions == k,
        colors="w",
        linewidths=0.5,
        zorder=100,
        extent=(-bound, bound, bound, -bound),
        antialiased=True,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    text_kw = dict(transform="a", size=5, va="t")

    for s, y, c in zip(["Single housed", "Group housed"], [0, -6], ["C0", "C1"]):
        ax.add_text(1, 1, s, c=c, ha="r", pad=(-1, y), **text_kw)

    ax = g[2:, 1].make_ax(sharex=True, behind=False)
    ax.set_ylim(0, 1)

    x0, x1 = ax.get_xlim()
    rect_l = Rectangle((x0, 0), x1 - x0, 1, fc="k", alpha=0.7, lw=0)
    ax.add_patch(rect_l)
    ax.axvline(0, color="w", ls="--", alpha=0.5)

    ax = g[3:, 2].make_ax(sharex=True, behind=False)
    ax.set_ylim(0, 1)
    x0, x1 = ax.get_xlim()
    rect_r = Rectangle((x0, 0), x1 - x0, 1, fc="k", alpha=0.7, lw=0)
    ax.add_patch(rect_r)
    ax.axvline(0, color="w", ls="--", alpha=0.5)

    ax = g.axs[2, -1]
    ax.add_text(0, 1, f"Cluster {k + 1}", ha="l", c="w", pad=(0, 0), **text_kw)
    time = ax.add_text(
        0, 1, f"t = ${0:+0.2f}$ s", ha="l", c="w", pad=(0, -7), **text_kw
    )

    g.fig.set_dpi(hpx / (g.height_pt / 72))

    for i in range(w):
        t = i / fps - w / fps / 2
        rect_l.set_x(t)
        rect_r.set_x(t)
        time.set_text(f"t = ${t:+0.2f}$ s")
        yield g.to_rgba_array()[..., :3]


def choice(*args, replace=False, random_state=0, **kwargs):
    rng = np.random.RandomState(random_state)
    return np.sort(rng.choice(*args, replace=replace, **kwargs))


def align_traj(fix, mov, max_shift=np.inf, ord=1):
    from scipy.optimize import minimize

    def f(x, p, q):
        theta, tx, ty = x
        dists = np.abs(p - q * np.exp(1j * theta) - complex(tx, ty))
        return np.linalg.norm(dists, ord)

    bounds = [(-np.pi, np.pi)] + [(-max_shift, max_shift)] * 2
    res1 = minimize(f, (0, 0, 0), bounds=bounds, args=(fix, mov))
    res2 = minimize(f, (0, 0, 0), bounds=bounds, args=(fix, mov.conj()))
    mirror = res2.fun < res1.fun
    theta, tx, ty = res2.x if mirror else res1.x
    c = np.cos(theta)
    s = np.sin(theta)

    if mirror:
        M = np.array(((c, s, tx), (s, -c, ty), (0, 0, 1)))
    else:
        M = np.array(((c, -s, tx), (s, c, ty), (0, 0, 1)))

    return M


def align_trajs(trajs, align_to=0, max_shift=np.inf, return_aligned=False):
    fix = trajs[align_to]
    M1 = np.empty((len(trajs), 3, 3))
    M1[align_to] = np.eye(3)

    for i, mov in enumerate(trajs):
        if i == align_to:
            continue
        M1[i] = align_traj(fix, mov, max_shift=max_shift)

    trajs = c2xy(trajs)
    ones = np.ones((*trajs.shape[:-1], 1))
    trajs = np.concatenate((trajs, ones), axis=-1)
    aligned = np.einsum("nab,ntb->nta", M1[:, :2], trajs)

    t2 = -aligned.mean((0, 1))
    R2 = np.linalg.svd(aligned.reshape((-1, 2)) + t2).Vh
    M2 = np.column_stack((R2, R2 @ t2))
    M = np.einsum("ij,kjl->kil", M2, M1)

    if return_aligned:
        aligned = np.einsum("nab,ntb->nta", M, trajs) @ (1, 1j)
        return M, aligned

    return M


def get_border_masks(n_l, n_rows, n_cols, size, thick):
    import cv2

    mask_l = np.zeros((n_cols, n_rows), dtype=np.uint8)
    mask_l.ravel()[:n_l] = 1
    mask_l = mask_l.T
    mask_l = np.kron(mask_l, np.ones((size, size), dtype=np.uint8))
    mask_r = 1 - mask_l
    kernel = np.ones((thick,) * 2, np.uint8)
    mask_l -= cv2.erode(np.pad(mask_l, 1), kernel)[1:-1, 1:-1]
    mask_r -= cv2.erode(np.pad(mask_r, 1), kernel)[1:-1, 1:-1]
    mask_l = mask_l.astype(bool)
    mask_r = mask_r.astype(bool)
    return mask_l, mask_r


def iter_r(k):
    def get_alphas(t, k=0.05):
        alphas = np.zeros(w)
        arange = np.arange(-w // 2, w // 2)
        leq = arange <= t
        alphas[leq] = np.exp(k * (arange[leq] - t))
        return alphas

    vr = VideoReader(cache_video_path.as_posix(), num_threads=1)

    events = np.where(labels == k)[0]
    cond = df["clips"]["condition"].values[events]
    ni = round((cond == "i").mean() * n_videos)
    ng = n_videos - ni
    events_i = choice(events[cond == "i"], ni)
    events_g = choice(events[cond == "g"], ng)
    events = np.concatenate([events_i, events_g])
    nodes = ["head", "thorax", "abdomen"]
    trajs = []

    for i in events.ravel():
        f1, f2 = {False: "lr", True: "rl"}[need_flip[i]]
        e = df["clips"].iloc[i]  # get event
        s_ = np.s_[e.ind_min_dist - w // 2 : e.ind_min_dist + w // 2]
        df_ = df["data"].loc[e.name]
        trajs.append(
            [
                np.mean([df_[(f1, k)].iloc[s_] for k in nodes], 0),
                np.mean([df_[(f2, k)].iloc[s_] for k in nodes], 0),
            ]
        )

    trajs = np.array(trajs)
    M, aligned = align_trajs(trajs.reshape((n_videos, 2 * w)), return_aligned=True)

    if np.argmin(aligned.reshape((len(events), 2, w))[..., 0].mean(0)):
        M[:, 0] *= -1

    frames = np.array([vr[j * w : j * w + w].asnumpy()[..., 0] for j in events.ravel()])
    frames = frames.transpose((1, 0, 2, 3))
    frames[:, :, :3] = 0
    mask_i, mask_g = get_border_masks(ni, n_rows, n_cols, size, thick=8)

    g = Grid(160, (n_rows, n_cols), dpi=72, facecolor="k")
    g.set_visible_sides("")
    axs = g.axs.T.ravel()
    lim = 200
    elems = []

    for i, ax in enumerate(axs):
        A = Affine2D.from_values(*M[i].T.ravel()) + ax.transData
        ax.set_xlim(-lim, lim)
        ax.set_ylim(lim, -lim)

        l1 = LineCollection(
            c2xy(swv(trajs[i, 0], 2)), color="m", lw=2, transform=A, alpha=0
        )

        l2 = LineCollection(
            c2xy(swv(trajs[i, 1], 2)), color="c", lw=2, transform=A, alpha=0
        )

        elems.append(
            [
                ax.imshow(frames[0, i], cmap="gray", vmax=255, transform=A),
                ax.add_collection(l1),
                ax.add_collection(l2),
            ]
        )

    cig = (np.array([to_rgb("C0"), to_rgb("C1")]) * 255).astype(np.uint8)

    for t in range(len(frames)):
        a = get_alphas(t - w // 2)

        for i, ax in enumerate(axs):
            elems[i][0].set_array(frames[t, i])

            if t > 0:
                elems[i][1].set_alpha(a)
                elems[i][2].set_alpha(a)

        im = g.to_rgba_array()[..., :3]
        im[mask_i] = cig[0]
        im[mask_g] = cig[1]
        yield im


def make_video(k):
    with get_writer(save_dir / f"{k:02d}.mp4", fps=fps_out) as writer:
        for im_l, im_r in zip(iter_l(k), iter_r(k)):
            im = np.concatenate((im_l, im_r), 1)
            writer.append_data(im)

    plt.close("all")


Parallel(n_jobs=-1, max_nbytes=None)(delayed(make_video)(k) for k in trange(n_clusters))
with open("files.txt", "w") as f:
    for k in range(n_clusters):
        f.write(f"file '{save_dir.absolute()}/{k:02d}.mp4'\n")
run(
    [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "files.txt",
        "-c",
        "copy",
        (save_dir / "vid3.mp4").as_posix(),
    ]
)

Path("files.txt").unlink()

for k in range(n_clusters):
    (save_dir / f"{k:02d}.mp4").unlink()
