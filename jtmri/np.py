import logging

from numpy import ma
from numpy.ma import MaskedArray, MAError, add, array, asarray, concatenate, count, \
    filled, getmask, getmaskarray, make_mask_descr, masked, masked_array, \
    mask_or, nomask, ones, sort, zeros
#from core import *

import numpy as np
from numpy import ndarray, array as nxarray
import numpy.core.umath as umath
from numpy.lib.index_tricks import AxisConcatenator
from numpy.linalg import lstsq

import jtmri.utils


log = logging.getLogger(__name__)


def flatten_inplace(seq):
    """Flatten a sequence in place."""
    k = 0
    while (k != len(seq)):
        while hasattr(seq[k], '__iter__'):
            seq[k:(k + 1)] = seq[k]
        k += 1
    return seq


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    (This docstring should be overwritten)
    """
    arr = array(arr, copy=False, subok=True)
    nd = arr.ndim
    if axis < 0:
        axis += nd
    if (axis >= nd):
        raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
            % (axis, nd))
    ind = [0] * (nd - 1)
    i = np.zeros(nd, 'O')
    indlist = list(range(nd))
    indlist.remove(axis)
    i[axis] = slice(None, None)
    outshape = np.asarray(arr.shape).take(indlist)
    i.put(indlist, ind)
    j = i.copy()
    res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
    #  if res is a number, then we have a smaller output array
    asscalar = np.isscalar(res)
    if not asscalar:
        try:
            len(res)
        except TypeError:
            asscalar = True
    # Note: we shouldn't set the dtype of the output from the first result...
    #...so we force the type to object, and build a list of dtypes
    #...we'll just take the largest, to avoid some downcasting
    dtypes = []
    if asscalar:
        dtypes.append(np.asarray(res).dtype)
        outarr = zeros(outshape, object)
        outarr[tuple(ind)] = res
        Ntot = np.product(outshape)
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= outshape[n]) and (n > (1 - nd)):
                ind[n - 1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
            outarr[tuple(ind)] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    else:
        ismasked = np.ma.getmaskarray(arr)[tuple(i.tolist())].all()
        res = array(res, copy=False, subok=True)
        j = i.copy()
        j[axis] = ([slice(None, None)] * res.ndim)
        j.put(indlist, ind)
        Ntot = np.product(outshape)
        holdshape = outshape
        outshape = list(arr.shape)
        outshape[axis] = res.shape
        dtypes.append(asarray(res).dtype)
        outshape = flatten_inplace(outshape)
        outarr = zeros(outshape, object)
        u = tuple(flatten_inplace(j.tolist()))
        outarr[u] = np.ma.masked if ismasked else res
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= holdshape[n]) and (n > (1 - nd)):
                ind[n - 1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            j.put(indlist, ind)
            s = arr[tuple(i.tolist())]
            u = tuple(flatten_inplace(j.tolist()))
            if ~s.mask.all():
                res = func1d(s, *args, **kwargs)
                outarr[u] = res
                dtypes.append(asarray(res).dtype)
            else:
                outarr[u] = np.ma.masked
            k += 1
    max_dtypes = np.dtype(np.asarray(dtypes).max())
    if not hasattr(arr, '_mask'):
        result = np.asarray(outarr, dtype=max_dtypes)
    else:
        result = asarray(outarr, dtype=max_dtypes)
        result.fill_value = ma.default_fill_value(result)
    return result
apply_along_axis.__doc__ = np.apply_along_axis.__doc__


def flatten_axes(a, axis, keepdims=False):
    '''A view of the input array with the specified axes collapsed into
    the first specified axis.

    Args:
    a    -- array to be flattened
    axis -- int or tuple of ints indicating axes to be flattened.

    Returns:
    A view of the input array with the specified axes collapsed into
    the first specified axis.

    Examples:
    >>> import numpy as np
    >>> x = np.ones((5,4,3,2,1))
    >>> flatten_axes(x, axis=[]).shape
    (5, 4, 3, 2, 1)
    >>> flatten_axes(x, axis=(0,1)).shape
    (20, 3, 2, 1)
    >>> flatten_axes(x, axis=(0,1), keepdims=True).shape
    (20, 1, 3, 2, 1)
    >>> flatten_axes(x, axis=(0,1,2,3,4), keepdims=True).shape
    (120, 1, 1, 1, 1)
    >>> flatten_axes(x, axis=(0,1,2,3,4)).shape
    (120,)
    >>> flatten_axes(x, axis=(2,3)).shape
    (5, 4, 6, 1)
    >>> np.array_equal(x.flatten(), flatten_axes(x, axis=(0,1,2,3,4)))
    True
    '''
    axes = np.array(sorted(jtmri.utils.as_iterable(axis)))
    bad_axes = list(axes[axes >= a.ndim])
    assert not bad_axes, 'Axes {!r} are bigger then a.ndim={!r}'.format(bad_axes, a.ndim)

    if len(axes) == 0:
        return a
    out_shape = np.array(a.shape)
    out_shape[axes[0]] = -1
    if keepdims:
        out_shape[axes[1:]] = 1
    else:
        diff = [i for i in range(a.ndim) if i not in axes[1:]]
        out_shape = out_shape[diff]
    # Preserve the ordering from the input array
    return a.reshape(out_shape, order='A')


def apply_to_axes(func, arr, axes, keepdims=False):
    """Apply func to a flattened version of a.

    Parameters
    ==========
    func : (ndarray, axis) -> scalar
        Function to apply to flattened array.
        Takes an ndarray and returns a scalar, must accept
        kwarg axis, indicating which axis to act on.
    arr : ndarray
        array to apply function to
    axes : iterable or scalar
        axis/axes to apply function over

    Returns:
    The result of applying func to the input array flattened along the specified axes.

    Examples:
    >>> import numpy as np
    >>> x = np.ones((5,4,3,2,1))
    >>> apply_to_axes(np.sum, x, axes=(0,1)).shape
    (3, 2, 1)
    >>> apply_to_axes(np.sum, x, axes=(0,1,2))
    np.array([[60.],
              [60.]])
    """
    axes = np.array(sorted(jtmri.utils.as_iterable(axes)))
    return func(flatten_axes(arr, axes, keepdims), axis=axes[0])


def expand(arr, shape):
    """Expand a ndarray to dims by duplicating it
    Parameters
    ==========
    arr : ndarray
    shape : iterable
        Iterable containing the size of each new dimension.
        These dimensions are appended to the existing ones in arr

    Returns
    =======
    ndarray
        `arr` expanded with new dims

    Examples
    ========
    >>> a = np.ones((3, 3));
    >>> a.shape
    [3, 3]
    >>> a = expand(a, (4, 5, 6));
    >>> a.shape
    [3, 3, 4, 5, 6]
    """
    for s in tuple(shape):
        arr = np.tile(arr[..., np.newaxis], s)
    return arr 


def iter_axes(a, axes):
    """Generates arrays by iterating over the specified axis.
    Parameters
    ==========
    a : ndarray
    axes : int or iterable
        axis or axes to iterate over. Axes are iterated over in order, starting from the last one.
        For example, if axes=(3,4,5), then dimension 5 is iterated over first, then 4, then 3.

    Returns
    =======
    Generator over the axes requested to iterate on
    """
    all_axes = np.array(sorted(jtmri.utils.as_iterable(axes)))
    dropped_axes = all_axes[all_axes >= a.ndim]
    if len(dropped_axes) > 0:
        log.warn('Axes larger than a.ndim (%r) have been dropped', dropped_axes, a.ndim)
    axes = all_axes[all_axes < a.ndim]

    if len(axes) == 0:
        raise StopIteration()
    a = flatten_axes(a, axes)
    axis = sorted(axes)[0]
    for arr in np.rollaxis(a, axis):
        yield arr


def mosaic(arr, aspect=16/9., fill=np.nan):
    """Turn a ndarray into a mosaic by reducing all dimensions above the first two
    Parameters
    ==========
    arr : ndndarray
    aspect : float
        Ratio between dimension 1 (width) and dimension 0 (height)
    fill : object
        If the aspect ratio cannot be satisfied given the shape of the input array,
        then fill any missing values with `fill`

    Returns
    =======
    ndarray:
        2D array mosaic from input array with aspect ratio `aspect`
    """
    if arr.ndim < 3:
        return arr
    arr_3d = flatten_axes(arr, range(2, arr.ndim))
    N = arr_3d.shape[2]
    if N == 1:
        nc, nr = 1, 1
    else:
        nc = int(np.sqrt(N / aspect))
        nr = int(np.ceil(N / float(nc)))
    pad = (nr * nc) - N
    if pad > 0:
        pad_im = fill * np.ones(arr_3d.shape[:2] + (pad,))
        arr_3d = np.concatenate((arr_3d, pad_im), 2)
    long_arr = np.vstack(np.rollaxis(arr_3d, -1))
    mos = np.hstack(np.array_split(long_arr, nr))
    if np.ma.isMaskedArray(arr):
        mos = np.ma.array(mos, mask=mosaic(arr.mask, aspect=aspect, fill=True))
    return mos



def checkerboard(shape, cell_len):
    """Create a checkerboard of 0's and 1's
    Args:
        shape    -- shape of N dimensional checkerboard
        cell_len -- length of checkerboard cell along each dimension, either an array with the same
                    length as shape where each value indicates the length for
                    the corresponding dimension, or a singular value indicating the
                    same number length for all dimensions
    Returns:
        A checkerboard of 0's and 1's with specified shape and cell length
    """
    cell_len = jtmri.utils.as_iterable(cell_len)
    if len(cell_len) == 1:
        cell_len = cell_len * len(shape)
    assert len(cell_len) == len(shape)
    ts = [np.arange(n) for n in shape]
    # Index matrix style
    M = np.meshgrid(*ts, indexing='ij')
    return sum(m // s for m, s in zip(M, cell_len)) % 2
