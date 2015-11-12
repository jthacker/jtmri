from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as pl
import itertools
from collections import namedtuple, Iterable

import jtmri.np

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)


BBox = namedtuple('BBox', ['x', 'y', 'width', 'height'])

def multi_bar(xlabels, data, yerr=None,
              group_labels=None, group_colors=None, axs=None,
              x_start=0,
              padding=0.15):
    """ Plot multiple bar plots grouped together
    Args:
      xlabels --
      data    -- 2-d array with first dimension representing groups and the second
                 representing points in that group
    """

    Ngroups = len(data)
    assert Ngroups > 0
    assert 0 <= padding <= 1.0
    Nx = len(data[0])

    if yerr is None:
        yerr = [[0] * Nx] * Ngroups
    else:
        assert len(yerr) == Ngroups

    if group_labels is None:
        group_labels = map(str, range(Nx))
    else:
        assert len(group_labels) == Ngroups

    if group_colors is None:
        colors = itertools.cycle(tableau20)
        group_colors = [colors.next() for _ in range(Ngroups)]
    else:
        assert len(group_colors) == Ngroups
    
    width = (1 - padding) / Ngroups
    
    if axs is None:
        _, axs = pl.subplots()

    if not isinstance(axs, Iterable):
        axs = [axs] * Ngroups
    else:
        assert len(axs) == Ngroups
    
    ind = x_start + np.arange(Nx)
    bboxs = {}
    rects = []
    for i, ax in zip(range(Ngroups), axs):
        x, y = ind+(width*i), np.array(data[i])
        rects.append(ax.bar(x, y, width, 
                            color=group_colors[i],
                            yerr=yerr[i],
                            ecolor='black'))
        bboxs[group_labels[i]] = [BBox(x[j], 0, width, y[j] + yerr[i][j])
                                  for j in range(Nx)]
        ax.set_xticks(ind + ((1 - padding) / 2))
        ax.set_xticklabels(xlabels)
        ax.set_xlim([-padding, Nx])
    legend = ax.legend([r[0] for r in rects], group_labels)
    return axs, legend, rects, bboxs


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
    if img.ndim > 2:
        img = jtmri.np.mosaic(img)
    cax = ax.imshow(img, **imshow_args)
    ax.axis('off')
    cb = pl.colorbar(cax, label=label)
    return ax,cb


def label_bars(bbox0, bbox1, text, y_offset=None, y_text_offset=None, ax=None):
    """Place a label between two bars
    Useful for indicating significant differences on a bar plot
    Args:
        bbox0         --
        bbox1         --
        text          --
        y_offset      --
        y_text_offset --
        ax            --
    """
    if ax is None:
        ax = pl.gca()

    props = {
        'connectionstyle': 'bar',
        'arrowstyle': '-',
        'lw': 1
    }

    y = max(bbox0.height, bbox1.height)

    y_lim = ax.get_ylim()
    if y_offset is None:
        y_offset = 0.05 * (y_lim[1] - y_lim[0])

    if y_text_offset is None:
        y_text_offset = 0.1 * (y_lim[1] - y_lim[0])

    xy0 = (bbox0.x + (bbox0.width / 2.), y + y_offset)
    xy1 = (bbox1.x + (bbox1.width / 2.), y + y_offset)
    ax.annotate('', xy=xy0, xytext=xy1, zorder=10, arrowprops=props)
    ax.annotate(text, xy=((xy0[0] + xy1[0]) / 2., xy0[1] + y_text_offset),
                zorder=10,
                ha='center')

def annotate_difference(pos0, pos1, text, text_offset=(0, 0), below=False, ax=None):
    """Links two points with a line and draws text in the middle.
    Useful for annotating significant differences.
    Args:
        pos0        -- tuple of (x, y) coordinates indicating the first point
        pos1        -- tuple of (x, y) cooridnates indicating the second point
        text        -- Text to draw in the middle
        text_offset -- Shift text by a tuple of (x, y) coorinates
        below       -- (default: False) Draw above or below points
        ax          -- (default: pl.gca()) axis to apply annotation to
    """
    ax = ax or pl.gca()
    arrowprops = {
        'connectionstyle': 'bar,fraction={}'.format(-0.3 if below else 0.3),
        'arrowstyle': '-',
        'lw': 1
    }
    ax.annotate('', xy=pos0, xytext=pos1, zorder=10, arrowprops=arrowprops)
    # Draw text in the middle
    mid_point = ((pos0[0] + pos1[0]) / 2. + text_offset[0],
                 (pos0[1] + pos1[1]) / 2. + text_offset[1])
    ax.annotate(text, xy=mid_point, zorder=10, ha='center')


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


def set_subplots_row_titles(axs, titles, pad=5):
    """ Set the titles for each row at the far left of the plot """
    assert len(titles) == len(axs[:, 0]), 'Number of rows must mactch the number of titles'
    for ax, row in zip(axs[:,0], titles):
        ax.annotate(row,
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label,
                    textcoords='offset points',
                    size='large',
                    ha='right',
                    va='center')


def set_subplots_col_titles(axs, titles, pad=5):
    """ Set the titles for each column at the far left of the plot """
    assert len(titles) == len(axs[0]), 'Number of columns must mactch the number of titles'
    for ax, col in zip(axs[0], titles):
        ax.annotate(col,
                    xy=(0.5, 1),
                    xytext=(0, pad),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    size='large',
                    ha='center',
                    va='baseline')
