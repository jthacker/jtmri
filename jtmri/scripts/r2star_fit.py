import numpy as np
import skimage.morphology
import jtmri.fit

def fit(t, data, min_threshold=20):
    selem = np.ones((2,2))

    mask = data < min_threshold
    for idx in np.ndindex(*mask.shape[2:]):
        slc = (slice(None,None), slice(None,None)) + idx
        mask[slc] = skimage.morphology.binary_dilation(mask[slc], selem=selem)
    mask = np.tile(mask.any(axis=-1)[...,np.newaxis], mask.shape[-1])
    return jtmri.fit.fit_r2star_fast(t, np.ma.array(data, mask=mask))
