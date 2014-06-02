import inspect, logging
from collections import namedtuple
import numpy as np
import pylab as plt
import operator as op
from scipy.optimize import curve_fit

from .cache import memoize
from .utils import ProgressMeter, rep

log = logging.getLogger('jtmri.fitting')

def _minValueFailedFitFunc(initialGuess):
    size = len(initialGuess)
    popt = np.zeros(size)
    pcov = np.zeros((size,size))
    pcov[np.diag_indices_from(pcov)] = np.inf
    return popt,pcov

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

        return apply_along_axis(self, x, arr, **kwargs)

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


def apply_along_axis(fitter, x, arr, axis=-1, disp=True):
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
        'len(x) != arr.shape[axis] x=%d arr.shape[%d]=%d' % (len(x), axis, arr.shape[axis])
    total = reduce(op.mul, (d for i,d in enumerate(arr.shape) if i != axis), 1)
    if disp:
        pm = ProgressMeter(total, 'Calculating map')
    def fit(y):
        if disp:
            pm.increment()
        return fitter.fit(x, y, disp=disp).params
    res = np.apply_along_axis(fit, axis, arr)
    if disp:
        pm.finish()
    return res


### Common Fitters ###
class fitters(object):
    r2 = Fitter(lambda te, so, r2: so * np.exp(-1 * r2 * te), (1,1))
    r2star = Fitter(lambda te, so, r2star: so * np.exp(-1 * r2star * te), (1,1))
    r2prime = Fitter(lambda tau, so, r2prime: so * np.exp(-2 * r2prime * tau), (1,1))
