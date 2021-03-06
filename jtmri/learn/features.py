import numpy as np
import pylab as pl
import itertools
import jtmri.utils


def generate_spatial_context_features(n, min_distance=0, max_distance=100, min_region_len=0, max_region_len=100):
    """Create n features based on Criminsi2009 long range spatial features
    A feature consists of the mean value of some parameterized portion of the image.
    Features are generated based on uniform interval distributions.
    Features consist of a distance vector and a box size.

    Parameters
    ----------
    n : int
        Number of spatial features to create
    min_distance : float (default: 0)
        Minimum distance of a generated feature from the origin
    max_distance : float (default: 100)
        Maximum distance of a generated feature from the origin
    min_region_len : float (default: 0)
        Minimum length of a generated feature side
    max_region_len : float (default: 100)
        Maximum length of a generated feature side
   
    Returns
    -------
    ndarray
        An array of the feature paramters.
        Each set of parameters consists of 4 values: x-distance, y-distance, width, height
    """
    radius = min_distance + (max_distance - min_distance) * np.sqrt(np.random.uniform(low=0, high=1, size=n))
    theta = np.random.uniform(low=0, high=2*np.pi, size=n)
    height = np.random.uniform(low=min_region_len, high=max_region_len, size=n)
    width = np.random.uniform(low=min_region_len, high=max_region_len, size=n)
    return np.vstack((radius * np.cos(theta), radius * np.sin(theta), width, height)).T


def plot_spatial_context_features_distributions(features, bins=25):
    params = [
        ('dx',),
        ('dy',),
        ('w',),
        ('h',)
    ]
    _, axs = pl.subplots(ncols=2, nrows=2)
    axs = jtmri.utils.flatten(axs)
    for (name,), ax, data in zip(params, axs, jtmri.np.iter_axes(features, 1)):
        ax.hist(data, bins=bins)
        ax.set_title(name)
    pl.tight_layout()

def plot_spatial_context_features(origin, features, scale=(1, 1), ax=None, show_arrows=True):
    """Given a origin and an array of features, plot them as rectangles and arrows."""
    from matplotlib.patches import Rectangle, Arrow
    if ax is None:
        _, ax = pl.subplots()
    ax.set_aspect('equal')
    ox, oy = origin
    sx, sy = scale
    for x, y, w, h in features:
        x /= sx
        y /= sy
        w /= sx
        h /= sy
        ax.add_patch(Rectangle((ox + x - w/2, oy + y - h/2), w, h, facecolor=(1,0,0,0.1)))
        if show_arrows:
            ax.add_patch(Arrow(ox, oy, x, y))


def integral_image(image):
    """Create an integral image on the first 2 dimensions from the input.
    Args:
        image -- ndarray with ndim >= 2
    Returns an integral image where every location i,j is the cumulative sum
    of all preceeding pixels.
    """
    return image.cumsum(1).cumsum(0)


def integral_image_sum(ii, r0, c0, r1, c1):
    """Find the sum of a region using an integral image."""
    return ii[r1, c1] + ii[r0, c0] - ii[r0, c1] - ii[r1, c0]


def integral_image_mean(ii, r0, c0, r1, c1):
    """Find the mean of a region using an integral image"""
    N = ((r1 - r0) * (c1 - c0)).astype(float)[:, np.newaxis]
    return integral_image_sum(ii, r0, c0, r1, c1) / N
    

def spatial_context_features_response(image, feature_params, scale, mask=None):
    """Compute the response of each feature for every pixel in the image
    Args:
        image          -- ndarray for computing response from.
                          if ndim is == 3, then the third dimension is treated as
                          extra channels and are aggregated in the response
        feature_params -- ndarray of shape [n_features, 4] (feature units should be in millimeters)
        scale          -- a 2-tuple (sx, sy) that gives the scale
                          from millimeters to pixels (e.g. 10 mm / pixel)
        mask           -- only compute the response for the pixels in this mask
        
    Returns: ndarray of feature responses, shape is [image.size, len(feature_params)]
    
    The response is computed as sum of the mean of each channel in the image.
    """
    assert 2 <= image.ndim <= 3
    assert feature_params.ndim == 2
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    N, M, C = image.shape
    sx, sy = scale
    n_features, _ = feature_params.shape

    # Compute the integral image to save on processing time
    ii = integral_image(image)
   
    def cleanse(x, xmax):
        return np.clip(x.ravel(), 0, xmax).astype(int)

    rr, cc = np.mgrid[:N, :M]
    if mask is not None:
        rr, cc = rr[mask], cc[mask]
    r, c = [x.ravel()[:,np.newaxis] for x in (rr, cc)]
    dx, dy, w, h = feature_params.T
    R0 = cleanse(r + (dy - h/2.) * sy, N - 1)
    R1 = cleanse(r + (dy + h/2.) * sy, N - 1)
    C0 = cleanse(c + (dx - w/2.) * sx, M - 1)
    C1 = cleanse(c + (dx + w/2.) * sx, M - 1)
    return integral_image_mean(ii, R0, C0, R1, C1).reshape(-1, n_features)
