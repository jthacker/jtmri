import numpy as np


def entropy(a):
    a = a / float(a.sum())
    a = a[np.nonzero(a)]
    return -(a * np.log2(a)).sum()

def _hist_range(a, bins):
    a_min, a_max = a.min(), a.max()
    s = 0.5 * (a_max - a_min) / float(bins - 1)
    return (a_min - s, a_max + s)


def mutual_information(a, b, bins=256):
    a = a.flatten()
    b = b.flatten()
    a_range = _hist_range(a, bins)
    b_range = _hist_range(b, bins)
    ab_hist, _, _ = np.histogram2d(a, b, bins=bins, range=(a_range, b_range))
    a_hist, _ = np.histogram(a, bins, range=a_range)
    b_hist, _ = np.histogram(b, bins, range=b_range)
    return entropy(a_hist) + entropy(b_hist) - entropy(ab_hist)
