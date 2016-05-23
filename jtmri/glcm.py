import itertools

import numpy as np
from skimage.exposure import rescale_intensity
from skimage.feature import greycomatrix, greycoprops

import jtmri.np


def glcm_image(im, extent, distances, angles, props):
    """Given a 2D image, compute the local GLCM for each pixel
    Parameters
    ==========
    im : ndarray (im.ndim == 2)
        2D array to compute local GLCM on
        If `im` is a masked array, only compute results over the unmasked area
    extent: int
        Each local GLCM is computed in a square region around the current voxel,
        with 2*width = 2*height = extent
    distances : iterable<int>
        Iterable of distances to compute local GLCM at
    angles : iterable<float>
        Iterable of angles to compute local GLCM at
    props : iterable<str>
        Iterable of property names to compute
        
    Returns
    =======
    out : ndarray
        5D array of local GLCM results
        Dimensions are: [rows, cols, distances, angles, property]
    """
    assert im.ndim == 2, 'Only supports 2D arrays'
    nrows = im.shape[0]
    ncols = im.shape[1]
    is_masked = np.ma.isMaskedArray(im)
    distances = list(distances)
    angles = list(angles)
    props = list(props)
    if is_masked:
        data = im.data
        indices = zip(*np.where(~im.mask))
    else:
        data = im
        indices = itertools.product(range(nrows), range(ncols))
    g = rescale_intensity(data, out_range=(0, 255)).astype(int)
    out = np.zeros(g.shape + (len(distances), len(angles), len(props)))
    for r, c in indices:
        rmin = np.clip(r-extent, 0, nrows)
        rmax = np.clip(r+extent, 0, nrows)
        cmin = np.clip(c-extent, 0, ncols)
        cmax = np.clip(c+extent, 0, ncols)
        glcm = greycomatrix(g[rmin:rmax, cmin:cmax], distances, angles)
        for i, prop in enumerate(props):
            out[r, c, :, :, i] = greycoprops(glcm, prop)
    if is_masked:
        out = np.ma.array(out, mask=jtmri.np.expand(im.mask, out.shape[2:]))
    return out
