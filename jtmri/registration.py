import numpy as np
import jtmri.phantom
import jtmri.plot
import jtmri.dcm
import skimage.transform
from scipy.optimize import basinhopping
from itertools import product


def slide(im1, im2, trans_x=[0], trans_y=[0], epsilon=lambda g: automatic_epsilon(g, noise_level=10)):
    x, y = [], []

    g = normalized_gradient_field(im1, epsilon)
    for dx, dy in product(trans_x, trans_y):
        trans = skimage.transform.AffineTransform(translation=(dx, dy))
        im_mod = skimage.transform.warp(im2, trans)
        x.append((dx, dy))
        g_mod = normalized_gradient_field(im_mod, epsilon)
        y.append(inner_product_distance_measure(g, g_mod))
    return np.array(x), np.array(y)


def normalized_gradient_field(arr, epsilon=1):
    """Find the normalized gradient field of input array
    Args:
     arr     -- ndarray
     epsilon -- estimate of image noise
    
    Returns the normalized gradient field of input array arr
    """
    grad = np.array(np.gradient(arr))
    if callable(epsilon):
        epsilon = epsilon(grad)
    mag = np.sqrt(np.sum(grad**2, axis=0) + epsilon**2)
    return grad / mag


def inner_product_distance_measure(arr0, arr1):
    return -0.5 * np.sum(np.sum(arr0 * arr1, axis=0)**2)


def automatic_epsilon(grad, noise_level=1):
    return noise_level * np.mean(np.sqrt(grad**2))


def _affine_transform(x, template_grad, img, epsilon):
    trans = skimage.transform.AffineTransform(translation=(x[0], x[1]))
    im_mod = skimage.transform.warp(img, trans)
    g_mod = normalized_gradient_field(im_mod, epsilon)
    return inner_product_distance_measure(template_grad, g_mod)
   

def register_linear(template_img, img, niter=1, epsilon = lambda g: automatic_epsilon(g, noise_level=20)):
    template_grad = normalized_gradient_field(template_img, epsilon)
    func = lambda x: _affine_transform(x, template_grad, img, epsilon)
    nx, ny = template_img.shape
    bounds = [(-nx, nx), (-ny, ny)]
    res = basinhopping(func,
                       x0=(0, 0),
                       niter=niter,
                       minimizer_kwargs=dict(method='L-BFGS-B',
                                             bounds=bounds))
    return res
