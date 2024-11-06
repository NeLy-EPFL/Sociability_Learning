import numpy as np
import pandas as pd


def c2xy(obj):
    """Convert complex numbers in a dataframe to x, y coordinates.

    Parameters
    ----------
    obj
        Object containing the complex numbers.

    Returns
    -------
    DataFrame
        Dataframe containing the x, y coordinates.
    """
    if not isinstance(obj, (pd.Series, pd.DataFrame)):
        obj = np.asarray(obj)
        return np.stack((obj.real, obj.imag), axis=-1)

    if isinstance(obj, pd.Series):
        df = obj.to_frame()

    if isinstance(obj, pd.DataFrame):
        df = obj
        data = np.stack([df.values.real, df.values.imag], axis=-1)
        data = data.reshape((len(df), -1))
        tuples = [i if isinstance(i, tuple) else (i,) for i in df.columns]
        tuples = [i + (j,) for i in tuples for j in "xy"]
        names = np.append(df.columns.names, "coord")
        columns = pd.MultiIndex.from_tuples(tuples, names=names)
        return pd.DataFrame(data, df.index, columns)
    else:
        return np.stack((obj.real, obj.imag), axis=-1)
    
def xy2c(obj, coord_level=-1):
    """Convert x, y coordinates in a dataframe to complex numbers.

    Parameters
    ----------
    obj
        Object containing the x, y coordinates.
    coord_level : int or str
        Level of columns that represents the x, y coordinates.

    Returns
    -------
    DataFrame
        Dataframe in which each (x, y) pair is represented by
        a complex number x + y * 1j.
    """
    if not isinstance(obj, (pd.DataFrame, pd.Series)):
        obj = np.asarray(obj)
        return obj @ (1, 1j)

    if isinstance(obj, (pd.DataFrame, pd.Series)):
        try:
            obj = obj.xs("x", 1, coord_level) + obj.xs("y", 1, coord_level) * 1j
        except TypeError:
            obj = obj[["x", "y"]] @ (1, 1j)
    else:
        obj = obj @ (1, 1j)

    return obj

def get_bbox(a):
    from itertools import combinations

    def _get_bbox_1d(a):
        return np.array((a.argmax(), len(a) - a[::-1].argmax()))

    a = np.asarray(a).astype(bool)
    return np.array(
        [
            _get_bbox_1d(a.any(i))
            for i in combinations(reversed(range(a.ndim)), a.ndim - 1)
        ]
    )

def get_kde(points, n_bins=512, bound=None, bw=0.1, border_rel=0.1):
    """Estimate pdf using FFT KDE.

    Parameters
    ----------
    points : array_like
        Datapoints to estimate from, with shape (# of samples, 2).
    n_bins : int
        Number of bins for each dimension.
    bound : float, optional
        Upper bound of the absolute values of the data.
        Will be calculated as max(abs(points)) * (1 + border_rel) if not provided.
    bw : float
        The bandwidth.
    border_rel : float, optional
        See description for bound.

    Returns
    -------
    ndarray
        Estimated 2D pdf.
    """
    from KDEpy.FFTKDE import FFTKDE

    if bound is None:
        bound = np.abs(points).max()
        if border_rel > 0:
            border = bound * border_rel
        else:
            border = 1e-7
        bound += border

    points = points[np.abs(points).max(1) < bound]
    grid = np.mgrid[-bound : bound : n_bins * 1j, -bound : bound : n_bins * 1j]
    grid = grid.reshape((2, -1)).T
    pdf = FFTKDE(bw=bw).fit(points).evaluate(grid)
    return pdf.reshape((n_bins, n_bins), order="F")[::-1], bound
