import skimage.exposure
from .features import spatial_context_features_response
from skimage.morphology import binary_dilation, disk
import numpy as np
from numpy.random import normal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl


def _features(im, feature_params, mask, scale):
    features = spatial_context_features_response(im, feature_params, scale, mask)
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 0
    return features
    
    
def train_single_image_classifier(clf, im, pos_mask, feature_params, scale):
    if pos_mask.sum() == 0:
        return
    neg_mask = (binary_dilation(pos_mask, selem=disk(20))).astype(bool) - pos_mask
    mask = pos_mask | neg_mask
    classes = pos_mask[mask]  ## Positive examples are 1, negatives are 0
    im = skimage.exposure.rescale_intensity(im, out_range=(0, 1))
    features = _features(im, feature_params, mask, scale)
    clf.fit(features, classes)

norm = lambda loc=0.,scale=1.,size=1: loc if scale==0 else normal(loc, scale, size)


def shepp_logan(n=256, mask_num=6):
    im, masks = jtmri.phantom.phantom(n, ret_masks=True)
    return im, masks[mask_num]


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
        

def predict_class(clf, im, feature_params, scale):
    N, M = im.shape
    im = skimage.exposure.rescale_intensity(im, out_range=(0, 1))
    features = _features(im, feature_params, np.ones_like(im, dtype=bool), scale)
    return clf.predict(features).reshape(N, M)


def predict_proba(clf, im, feature_params, scale):
    N, M = im.shape
    im = skimage.exposure.rescale_intensity(im, out_range=(0, 1))
    features = _features(im, feature_params, np.ones_like(im, dtype=bool), scale)
    C = clf.classes_.size
    return clf.predict_proba(features).reshape(N, M, C)


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
