"""
Outline based registration
==========================

Register to images based on their outlines
Find the outline of the reference and 
Finds an affine transform base on the i
"""

from jtmri.image import morphsnakes

from collections import namedtuple
import numpy as np
from scipy.optimize import basinhopping
import skimage.transform


RegistrationResultBase = namedtuple('RegistrationResult', (
    'matrix',
    'reference_mask',
    'image_mask',
    'image_mask_warped',
    'fit'))


def resize_and_warp(img, shape, matrix):
    img = skimage.transform.resize(img, shape)
    return skimage.transform.warp(img, matrix.inverse)


def resize_and_warp_mask(mask, shape, matrix):
    mask = resize_and_warp(mask, shape, matrix)
    return (mask > 0.5).astype(bool)


class RegistrationResult(RegistrationResultBase):
    def warp(self, img):
        img_warped = resize_and_warp(img, self.reference_mask.shape, self.matrix)
        if np.ma.is_masked(img):
            mask_warped = resize_and_warp_mask(img.mask, self.reference_mask.shape, self.matrix)
            return np.ma.array(img_warped, mask=mask_warped)
        return img_warped

    def disp(self):
        """Display a summary of the warp"""
        single = lambda x: '{:.2f}'.format(x)
        double = lambda x: '({:.2f}, {:.2f})'.format(*x)
        props = [
            ('rotation', single),
            ('scale', double),
            ('shear', single),
            ('translation', double)]

        out = ["Transformation Parameters:"]
        for prop, fmt in props:
            val = getattr(self.matrix, prop)
            out.append("{:>12s} -- {}".format(prop, fmt(val)))
        print('\n'.join(out))


def contours(arr, max_iterations=None, threshold=None, smoothing=2, lambda1=1, lambda2=2):
    assert arr.ndim == 2
    m = morphsnakes.MorphACWE(arr,
                              smoothing=smoothing,
                              lambda1=lambda1,
                              lambda2=lambda2)
    m.levelset = np.ones_like(arr, dtype=np.float)
    m.run(max_iterations, threshold)
    return m.levelset


def _x_to_affine(x):
    return skimage.transform.AffineTransform(
        translation=(x[0], x[1]),
        scale=(x[2], x[3]))


class CostFunc(object):
    def __init__(self, ref, img):
        self.ref_mask = ref > 0
        self.img = img

    def cost(self, img):
        """Find the total number of overlapping pixels in two binary arrays"""
        img_mask = img > 0
        return float(np.logical_xor(self.ref_mask, img_mask).sum())

    def __call__(self, x):
        tr = _x_to_affine(x)
        img_tr = skimage.transform.warp(self.img, tr.inverse)
        return self.cost(img_tr)


def register_masked(ref_mask, img_mask, niter=None, x0=None):
    """Register img_mask to ref_mask, assumes inputs are contour masks
    Args:
        ref_mask -- reference image
        img_mask -- image to register to reference
        niter    -- number of iterations
        x0       -- starting value for transform (dx, dy, dsx, dsy)

    Returns: (matrix, image)
        matrix -- skimage AffineTranform object
        image  -- img warped to reference
    """
    assert ref_mask.ndim == img_mask.ndim == 2
    if niter is None:
        niter = 100
    x0 = x0 or (0, 0, 1, 1)
    ny, nx = ref_mask.shape
    img_mask_resized = skimage.transform.resize(img_mask, ref_mask.shape)
    minimizer_kwargs = dict(method='COBYLA')
    res = basinhopping(CostFunc(ref_mask, img_mask_resized),
                       x0=x0,
                       niter=niter,
                       minimizer_kwargs=minimizer_kwargs)
    tr = _x_to_affine(res.x)
    img_mask_warped = resize_and_warp_mask(img_mask, ref_mask.shape, tr)
    reg_res = RegistrationResult(tr, ref_mask, img_mask, img_mask_warped, res)
    return img_mask_warped, reg_res


def register(ref, img, niter=None, x0=None):
    """Register img to ref
    Args:
        ref   -- reference image
        img   -- image to register to reference
        niter -- number of iterations
        x0    -- starting value for transform (dx, dy, dsx, dsy)

    Returns: (matrix, image)
        matrix -- skimage AffineTranform object
        image  -- img warped to reference
    """
    ref_mask = contours(ref)
    img_mask = contours(img)
    _, reg_res = register_masked(ref_mask, img_mask, niter, x0)
    return reg_res.warp(img), reg_res
