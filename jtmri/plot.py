from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as pl

def multi_bar(xlabels, data, yerr, grouplabels, groupcolors, ax=None, padding=0.15):
    Ngroups = len(data)
    assert Ngroups > 0
    assert 0 <= padding <= 1.0
    Nx = len(data[0])
    
    width = (1 - padding) / Ngroups
    
    if ax is None:
        _, ax = pl.subplots()
    
    ind = np.arange(Nx)
    rects = []
    for i in range(Ngroups):
        rects.append(ax.bar(ind+(width*i), data[i], width, 
            color=groupcolors[i], yerr=yerr[i], ecolor='black'))
    ax.set_xticks(ind+((1-padding)/2))
    ax.set_xticklabels(xlabels)
    ax.set_xlim([-padding, Nx])
    ax.legend([r[0] for r in rects], grouplabels)
    return ax,rects


def line(y_intercept, slope, xlimits=None, ax=None, plot_kwds={}):
    '''Draw a line by the y-intercept and slope within the limits of the axes'''
    if ax is None:
        _,ax = pl.subplots()
    if xlimits is None:
        xlimits = ax.get_xlim()
    x = np.linspace(xlimits[0], xlimits[1])
    y = slope * x + y_intercept
    line = ax.plot(x, y, **plot_kwds)
    return ax,line


def adjust_limits(ax, xadj=0.1, yadj=0.1):
    '''Adjust the limits on the specified axes
    Args:
    ax   -- axis to adjust limits on
    xadj -- fraction of x-range to increase(+)/decrease(-) limits by
    yadj -- fraction of y-range to increase(+)/decrease(-) limits by
   
    Use positive(+) adjustment values to increase the limits and
    negative(-) values to decrease the limits'''
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    xrng = xmax - xmin
    yrng = ymax - ymin
    ax.set_xlim(xmin - xadj * xrng, xmax + xadj * xrng)
    ax.set_ylim(ymin - yadj * yrng, ymax + yadj * yrng)


def density(arr, samples=1000, plt_kwds=dict(), ax=None):
    if ax is None:
        _,ax = pl.subplots()
    gkde = stats.gaussian_kde(arr)
    idx = np.linspace(arr.min(), arr.max(), samples)
    ax.plot(idx, gkde.evaluate(idx), **plt_kwds)
    ax.set_ylabel('Density')
