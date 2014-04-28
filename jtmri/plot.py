from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def multi_bar(xlabels, data, yerr, grouplabels, groupcolors, ax=None, padding=0.15):
    Ngroups = len(data)
    assert Ngroups > 0
    assert 0 <= padding <= 1.0
    Nx = len(data[0])
    
    width = (1 - padding) / Ngroups
    
    if ax is None:
        _, ax = plt.subplots()
    
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

