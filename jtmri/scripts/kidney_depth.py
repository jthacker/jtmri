import numpy as np

def distance_from_region(label_mask, distance_mask=None, scale=1, ord=2):
    """Find the distance at every point in an image from a set of labeled points.
    Parameters
    ==========
    label_mask : ndarray
        A mask designating the points to find the distance from. A True value
        indicates that the pixel is in the region, a False value indicates it is not.
    distance_mask : ndarray
        A mask inidicating which regions to calculate the distance in
    scale : int
        Scale the calculated distance to another distance measure (eg. to millimeters)
    ord : int
        Order of norm to use when calculating distance. See np.linalg.norm for more details

    Returns
    =======
    distances : ndarray
        A masked array of the same size as label_mask.
        If distance_mask is passed in then the output array is masked by it.
    """
    if distance_mask is None:
        distance_mask = np.ones(label_mask.shape, dtype=bool)
    assert label_mask.shape == distance_mask.shape
    scale = np.array(scale)
    output = np.zeros(label_mask.shape)

    indxs = np.indices(label_mask.shape)
    X = indxs[:, distance_mask].T
    Y = indxs[:, label_mask].T
    for x in X:
        output[tuple(x)] = np.linalg.norm(scale*(x-Y), ord=ord, axis=1).min()
    return np.ma.array(output, mask=np.logical_not(distance_mask))


def contours(distances, contours=10):
    amin,amax = distances.min(), distances.max()
    edges,step = np.linspace(amin, amax, contours, retstep=True)
    mask = np.logical_not(np.ma.getmaskarray(distances))
    return [np.ma.getdata(mask & (distances >= cntr) & (distances < (cntr+step))) for cntr in edges[:-1]], edges


def plot_by_contours(arr, contour_masks, contour_vals, ax=None):
    if ax is None:
        import pylab as pl
        _,ax = pl.subplots()

    x = contour_vals[:-1]
    y = np.array([arr[mask].mean() for mask in contour_masks])
    ax.set_xlabel('Distance from surface (mm)')
    ax.set_ylabel('Mean R2* value')
    return ax.plot(x, y, 'o--')[0], x, y


def plot_by_distance(arr, distances, ax=None):
    assert arr.shape == distances.shape
    if ax is None:
        import pylab
        _,ax = pylab.subplots()

    mask = np.logical_not(np.ma.getmaskarray(distances))
    x = distances[mask].ravel()
    y = arr[mask].ravel()

    return ax.plot(x,y,'o')
