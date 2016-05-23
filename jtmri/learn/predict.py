import skimage.exposure
from .features import spatial_context_features_response
from skimage.morphology import binary_dilation, disk
import numpy as np
from numpy.random import normal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl

import logging


log = logging.getLogger(__name__)


def train_single_image_classifier(clf, im, pos_mask, neg_mask, features):
    if pos_mask.sum() == 0:
        log.debug('positive mask is empty, skipping')
        return
    mask = pos_mask | neg_mask
    classes = pos_mask[mask]  ## Positive examples are 1, negatives are 0
    clf.fit(features, classes)


norm = lambda loc=0.,scale=1.,size=1: loc if scale==0 else normal(loc, scale, size)


def gen_modified_images(image, mask, 
                        translation=(0, 0), 
                        scale=(0, 0), 
                        rotation=0., 
                        noise_loc=0, 
                        noise_scale=0,
                        bias_loc=0,
                        bias_scale=0):
    while True:
        tr = skimage.transform.AffineTransform(
            translation=(norm(scale=translation[0]), norm(scale=translation[1])),
            scale=(norm(loc=1, scale=scale[0]), norm(loc=1, scale=scale[1])),
            rotation=norm(scale=rotation))
        noise = norm(size=image.shape, loc=noise_loc, scale=noise_scale)
        bias = np.ones_like(image) * norm(loc=bias_loc, scale=bias_scale)
        pos_mask = skimage.transform.warp(mask, tr).astype(bool)
        im = skimage.transform.warp(image, tr) + noise + bias
        yield im, pos_mask
        

def plot_predictions(im, actual_mask, pos_class_prediction, prob_prediction):
    _, axs = pl.subplots(ncols=4, figsize=(25,10))

    axs[0].set_title('input')
    m = axs[0].imshow(im)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(m, cax=cax)
    axs[0].set_axis_off()

    mask_map = np.nan * np.ones_like(actual_mask, dtype=bool)
    mask_map[pos_class_prediction & actual_mask] = 3
    mask_map[pos_class_prediction & ~actual_mask] = 2
    mask_map[~pos_class_prediction & actual_mask] = 1
    
    axs[1].set_title('positive class prediction')
    m = axs[1].imshow(mask_map, interpolation='nearest', vmin=1, vmax=3)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = pl.colorbar(m, cax=cax, ticks=[1,2,3])
    cbar.ax.set_yticklabels(['Actual', 'Predicted', 'Agree'])
    axs[1].set_axis_off()

    axs[2].set_title('positive class probability')
    m = axs[2].imshow(prob_prediction[:,:,1], interpolation='nearest', vmin=0, vmax=1)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(m, cax=cax)
    axs[2].set_axis_off()
    axs[2].set_title

    axs[3].set_title('negative class probability')
    m = axs[3].imshow(prob_prediction[:,:,0], interpolation='nearest', vmin=0, vmax=1)
    divider = make_axes_locatable(axs[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(m, cax=cax)
    axs[3].set_axis_off()
    axs[3].set_title
