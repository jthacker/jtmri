import itertools
import warnings

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
    >>> flatten_axes(x, axis=(0,1)).shape
    (20, 3, 2, 1)
    >>> flatten_axes(x, axis=(0,1), keepdims=True).shape
    (20, 1, 3, 2, 1)
    >>> flatten_axes(x, axis=(0,1,2,3,4), keepdims=True).shape
    (120, 1, 1, 1, 1)
    >>> flatten_axes(x, axis=(0,1,2,3,4)).shape
    (120,)
    >>> np.array_equal(x.flatten(), flatten_axes(x, axis=(0,1,2,3,4)))
    True
    '''
    axes = np.array(sorted(jtmri.utils.as_iterable(axis)))
    bad_axes = list(axes[axes > a.ndim])
    assert not bad_axes, 'Axes {!r} are bigger then a.ndim={!r}'.format(bad_axes, a.ndim)

    out_shape = np.array(a.shape)
    out_shape[axes[0]] = -1
    if keepdims:
        out_shape[axes[1:]] = 1
    else:
        diff = [i for i in range(a.ndim) if i not in axes[1:]]
        out_shape = out_shape[diff]

    # Preserve the ordering from the input array
    return a.reshape(out_shape, order='A')


def apply_to_axes(func, a, axis, keepdims=False):
    '''Apply func to a flattened version of a.

    Args:
    func    -- function to apply, takes array and returns scalar, must accept
               kwarg axis, indicating which axis to act on.
    a       -- array to apply function to
    axis    -- axis to apply function over

    Returns:
    The result of applying func to the input array flattened along the specified axes.

    Examples:
    >>> import numpy as np
    >>> x = np.ones((5,4,3,2,1))
    >>> apply(np.sum, x, axis=(0,1)).shape
    (3, 2, 1)
    >>> apply(np.sum, x, axis=(0,1,2))
    np.array([[60.],
              [60.]])
    '''
    axes = np.array(sorted(jtmri.utils.as_iterable(axis)))
    return func(flatten_axes(a, axis, keepdims), axis=axes[0])


def iter_axis(a, axis):
    '''Generates arrays by iterating over the specified axis.
    If this axis does not exist, then nothing is generated
    '''
    if axis > a.ndim:
        yield
    for arr in np.rollaxis(a, axis):
        yield arr


def mosaic(a, aspect=16/9.):
    '''Turn 3-D array a into a mosaic
    Args:
    a   -- 3-D ndarray
    Returns a 2-D ndarray
    '''
    _, _, N = a.shape
    nc = int(np.sqrt(N / aspect))
    nr = int(np.ceil(N / float(nc)))
    pad = (nr * nc) - N
    if pad > 0:
        pad_im = np.nan * np.ones(a.shape[:2] + (pad,))
        a = np.concatenate((a, pad_im), 2)
    long_arr = np.vstack(a.swapaxes(0, 2))
    return np.hstack(np.array_split(long_arr, nr))
