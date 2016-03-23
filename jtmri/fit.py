import inspect, logging
from collections import namedtuple
import numpy as np
import pylab as plt
import operator as op
import skimage.morphology
from scipy.optimize import curve_fit

from .cache import memoize
from .utils import rep
from .progress_meter import ProgressMeter
from .np import apply_along_axis

log = logging.getLogger(__name__)


def _minValueFailedFitFunc(initialGuess):
    size = len(initialGuess)
    popt = np.zeros(size)
    pcov = np.zeros((size,size))
    pcov[np.diag_indices_from(pcov)] = np.inf
    return popt,pcov


def siemens_t2star_to_r2star(t2star):
    """Mimics Siemens scanner inversion of t2star to r2star.
    Does not provide the correct value for a small segment of the colormap though.

    Parameters
    ----------
    t2star : ndarray
        T2* map that has been loaded from a siemens dicom file. This map is expected
        to have 16 bit values for T2*.

    Returns
    -------
    The result is an inverted T2* map (R2*) that is the same as if it had been done on the scanner.
    The values are not correct in general, as they are just attempting to match what the scanner does.
    """
    r2star = 1000.0 / t2star
    r2star[t2star == 0] = 0
    r2star[t2star == 1] = 1
    return r2star.astype(int)


Fit = namedtuple('Fit', 'x y params paramdict func covmat success')


def plot_fit(fit, ax=None, overshoot=0.2):
    '''Plot the fitted parameters. The data points are placed and then the
    function is plotted over the x-range used during the inital fit.
    Args:
    overshoot -- percentage to extend the x range when plotting the fitted function
    '''
    fig = None
    if ax is None:
        fig,ax = plt.subplots()
    title = " ".join(["%s=%0.3g" % (k,v) for k,v in fit.paramdict.iteritems()])
    title = title if fit.success else "FAILED " + title

    ax.set_title(title)
    ax.plot(fit.x, fit.y, 'bo', label='data')

    xmin = fit.x.min()
    xmax = fit.x.max()
    overshoot = overshoot * (xmax - xmin)
    xFunc = np.linspace(xmin - overshoot, xmax + overshoot, 100)

    ax.plot(xFunc, fit.func(xFunc, *fit.params), 'r-', label='fit')
    ax.set_xlabel('x')
    ax.grid()

    if fig is not None:
        fig.show()

    return ax


class Fitter(object):
    def __init__(self, fitFunc, guess, failedFitFunc=_minValueFailedFitFunc):
        '''Create a new Fitter object.
        fitFunc should be of the form:
            lambda xData, param0, ... : function(xData, param0, ...)
        The first argument should be the xData used for fitting.
        The Fitter object will attempt to find estimates for rest 
        of the specified parameters.
        If the Fitter fails to find a fit, then failedFitFunc is 
        called to get default values for those points.
        '''
        fitFuncArgs = inspect.getargspec(fitFunc).args

        assert len(fitFuncArgs) > 1, "The fitFunc should take at least 2 arguments\
            (the first is the xdata and the rest are parameters to be fit),\
            only %d specified." % len(fitFuncArgs)

        self._xname,self._paramsToFit = fitFuncArgs[0],fitFuncArgs[1:]

        assert len(guess) == len(self._paramsToFit), 'len(guess)=%d \
            should equal the number of free parameters to be fit (%d)' % \
            (len(guess), len(self._paramsToFit))
        
        self._fitFunc = fitFunc
        if np.isscalar(guess):
            guess = np.array(guess)
        self._guess = guess
        self._failedFitFunc = failedFitFunc

    def __call__(self, x, arr, axis=None, disp=None):
        kwargs = {}
        if axis is not None:
            kwargs['axis'] = axis
        if disp is not None:
            kwargs['disp'] = disp

        return fit_along_axis(self, x, arr, **kwargs)

    def fit(self, xdata, ydata, disp=True):
        '''Find estimates for the parameters of the fitFunc given the 
        initalGuess, xdata and ydata. Thie initalGuess is needed inorder 
        for the nonlinear fitting function to converge. It should be the 
        same length as the number of parameters specifed in the fitFunc.
        '''
        xdata = np.array(xdata)
        ydata = np.array(ydata)

        assert xdata.ndim == 1, "xdata must be 1D but has %d dims" % xdata.ndim
        assert ydata.ndim == 1, "ydata must be 1D but has %d dims" % ydata.ndim

        assert len(xdata) == len(ydata), \
                "xdata and ydata must have the same length, %d != %d" % \
                (len(xdata), len(ydata))

        successfulFit = True

        try:
            popt,pcov = curve_fit(self._fitFunc, xdata, ydata, self._guess)
        except RuntimeError as e:
            successfulFit = False
            if disp:
                log.warn('Failed to find an appropriate fit, using the default')
                log.debug(e)
            popt,pcov = self._failedFitFunc(self._guess)

        if np.isscalar(pcov):
            successfulFit = False
            assert pcov == np.inf
            if disp:
                log.warn("Failed to find an appropriate fit, using the default value.")
            pcov = self._failedFitFunc(self._guess)[1]
        
        paramdict = dict(zip(self._paramsToFit, popt))

        fit = Fit(xdata, ydata, popt, paramdict, self._fitFunc, pcov, successfulFit)
        status = 'succeeded' if fit.success else 'FAILED'
        return fit


def fit_along_axis(fitter, x, arr, axis=-1, disp=True):
    '''Apply fitter function to arr
    Args:
    fitter -- a function that takes a numpy array of y values and 
              returns the fit value of paramters.
    arr    -- multi-dimenision array to fit over
    axis   -- (default: -1) the axis to fit the data across
    '''
    axis = np.arange(arr.ndim)[axis]
    assert len(x) == arr.shape[axis], 'The length of the x values must be the same ' \
        'as the length of the array along the axis being fit. ' \
        'len(x) != arr.shape[axis] len(x)=%d arr.shape[%d]=%d' % (len(x), axis, arr.shape[axis])
    #total = reduce(op.mul, (d for i,d in enumerate(arr.shape) if i != axis), 1)
    res = apply_along_axis(lambda y: fitter.fit(x,y,disp).params, axis, arr)
    return res


def fit_r2star_fast(t, arr, axis=-1):
    '''Fit an r2star decay across the specified axis using a linear least squares fit.
    arr is transformed to a linear equation by taking the natural logrithm of the
    R2* decay equation (log(S) = log(So) - log(TE) * R2*).
    Args:
    t    -- Echo times in seconds
    arr  -- N dimensional array of GRE data
    axis -- (default: -1) Axis that R2* decay occurs on

    Returns:
    An N-1 dimensional array of R2* fits.
    An N-1 dimensional array of residuals.
    '''
    arr_dims = range(arr.ndim)
    axis = arr_dims[axis]
    assert arr.shape[axis] == len(t), 'The length of t (%d) must be equal to the ' \
            'length of the arr dimension designated by axis (%d).' % (len(t), axis)

    out_shape = [arr.shape[i] for i in arr_dims if i != axis]

    A = np.c_[-1*t, np.ones_like(t)]
    b = np.log(arr.swapaxes(axis, -1).reshape((-1, len(t)))).T
    b[np.isinf(b)] = 0
    fits, residuals, rank, singvals = np.linalg.lstsq(A, b)
    r2star = fits[0].reshape(out_shape)
    residuals = residuals.reshape(out_shape)

    if np.ma.isMaskedArray(arr):
        r2star = np.ma.array(r2star, mask=np.ma.getmaskarray(arr).all(axis=-1))
        residuals = np.ma.array(residuals, mask=np.ma.getmaskarray(arr).all(axis=-1))
    return r2star, residuals


def fit_r2star_with_threshold(t, data, min_threshold=20):
    '''Fit an R2* map using a fast fitting method and thresholded GRE images.
    Assumes that the last dimension holds the echos'''
    selem = np.ones((2,2))

    mask = data < min_threshold
    for idx in np.ndindex(*mask.shape[2:]):
        slc = (slice(None,None), slice(None,None)) + idx
        mask[slc] = skimage.morphology.binary_dilation(mask[slc], selem=selem)
    mask = np.tile(mask.any(axis=-1)[...,np.newaxis], mask.shape[-1])
    return fit_r2star_fast(t, np.ma.array(data, mask=mask))


fit_r2 = Fitter(lambda te, so, r2: so * np.exp(-1 * r2 * te), (1,1))
fit_r2star = Fitter(lambda te, so, r2star: so * np.exp(-1 * r2star * te), (1,1))
fit_r2prime = Fitter(lambda tau, so, r2prime: so * np.exp(-2 * r2prime * tau), (1,1))
