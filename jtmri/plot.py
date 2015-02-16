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


def max_limits(axs):
    '''Find the maximum limits that fit all the provided axes
    Args:
    axs -- iterable of axis objects

    Returns: (xlims, ylims)
    '''
    xmin, xmax = zip(*(ax.get_xlim() for ax in axs))
    ymin, ymax = zip(*(ax.get_ylim() for ax in axs))
    return (min(xmin), max(xmax)), (min(ymin), max(ymax))


def set_limits_equal(axs, square=False):
    '''Set the limits of all axs to be the same.
    Finds the maximum limits that will fit them all.
    Args:
    axs    -- iterable of axes
    square -- (default: False) Set x and y axes to be the same
    '''
    xlim, ylim = max_limits(axs)
    if square:
        xmin, xmax = xlim
        ymin, ymax = ylim
        amin, amax = min(xmin, ymin), max(xmax, ymax)
        xlim, ylim = (amin, amax), (amin, amax)
    for ax in axs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

def density(arr, samples=1000, plt_kwds=dict(), ax=None):
    if ax is None:
        _,ax = pl.subplots()
    gkde = stats.gaussian_kde(arr)
    idx = np.linspace(arr.min(), arr.max(), samples)
    ax.plot(idx, gkde.evaluate(idx), **plt_kwds)
    ax.set_ylabel('Density')


def imshow(img, imshow_args={}, title='', label='', ax=None):
    '''Wrapper around matplotlib.imshow but with better defaults'''
    if ax is None:
        fig,ax = pl.subplots()
    cax = ax.imshow(img, **imshow_args)
    ax.axis('off')
    cb = pl.colorbar(cax, label=label)
    return ax,cb


def mean_difference(x, y, ax=None, scatter_kwargs=None):
    '''Plots mean-difference (bland-altman) of x - y
    Args:
    x               -- iterable
    y               -- iterable
    ax              -- (default: new axis) axis object to plot on
    scatter_kwargs  -- (default: defaults) keyword arguments to scatter

    Returns:
    axis object
    '''
    defaults = {
        'facecolor': 'none',
        'edgecolors': 'k'}
    scatter_kwargs = dict(defaults, **(scatter_kwargs or {}))
    if ax is None:
        fig, ax = pl.subplots()
    sx, sy = ((x + y) / 2.), (x - y)
    m = np.mean(sy)
    sd = np.std(sy)
    
    ax.scatter(sx, sy, **scatter_kwargs)
    ax.axhline(m, linestyle='--', color='gray', label='mean'.format(m))
    ax.axhline(m + 2*sd, linestyle='--', color='r', label='mean +/- 2*std'.format(sd))
    ax.axhline(m - 2*sd, linestyle='--', color='r')
    ax.set_xlabel('mean')
    ax.set_ylabel('diff')

    ax.set_title('Mean-Difference plot\n(mean: {:.3g} std: {:.3g})'.format(m, sd))
    return ax
