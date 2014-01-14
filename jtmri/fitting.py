import inspect
import numpy as np
import pylab as plt
import logging
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

class Fit(object):
    def __init__(self, x, y, func, params, covMatrix, success):
        self.x = x
        self.y = y
        self.func = func
        self.params = params
        self.covMatrix = covMatrix
        self.success = success

        # First argument is the x-data, so throw it out
        # The rest should be the parameters that were fit
        funcArgs = inspect.getargspec(self.func).args
        xName,paramNames = funcArgs[0],funcArgs[1:]
        self._xlabel = xName
        self.paramDict = dict(zip(paramNames, params))

    def plot(self, funcOvershoot=0.2):
        '''Plot the fitted parameters. The data points are placed and then the
        function is plotted over the x-range used during the inital fit.
        Args:
        funcOvershoot -- amount to extend the x range when plotting 
                         the fitted function
        '''
        plt.figure()
        title = 'Fit: '
        title += " ".join(["%s=%s" % (k,v) for k,v in self.paramDict.iteritems()])
        title = title if self.success else "FAILED " + title

        plt.title(title)
        plt.plot(self.x, self.y, 'bo', label='data')

        xmin = self.x.min()
        xmax = self.x.max()
        overshoot = funcOvershoot * (xmax - xmin)
        xFunc = np.linspace(xmin - overshoot, xmax + overshoot, 100)

        plt.plot(xFunc, self.func(xFunc, *self.params), 'r-', label='fit')
        plt.xlabel(self._xlabel)
        plt.grid()
        plt.legend()
        plt.show()

    def __repr__(self):
        return rep(self, ('x','y','func','paramDict','covMatrix','success'))


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

        self._paramsToFit = fitFuncArgs[1:]

        assert len(guess) == len(self._paramsToFit), 'len(guess)=%d \
            should equal the number of free parameters to be fit (%d)' % \
            (len(guess), len(self._paramsToFit))
        
        self._fitFunc = fitFunc
        if np.isscalar(guess):
            guess = np.array(guess)
        self._guess = guess
        self._failedFitFunc = failedFitFunc

    def __call__(self, xdata, ydata):
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
            log.warn('Failed to find an appropriate fit, using the default')
            log.debug(e)
            popt,pcov = self._failedFitFunc(self._guess)

        if np.isscalar(pcov):
            successfulFit = False
            assert pcov == np.inf
            log.warn("Failed to find an appropriate fit, using the default value.")
            pcov = self._failedFitFunc(self._guess)[1]

        fit = Fit(xdata, ydata, self._fitFunc, popt, pcov, successfulFit)
        status = 'succeeded' if fit.success else 'FAILED'
        log.debug("Fit [%s] params = %s" % (status, fit.paramDict))
        return fit


def fitMapper(fitter, arr, axis=None):
    '''Apply fitter function on arr
    Args:
    fitter -- a function that takes a numpy array of y values and 
              returns the fit value of paramters.
    arr    -- multi-dimenision array to fit over
    axis   -- the axis to fit the data across
    '''
    if axis is None:
        axis = arr.ndim - 1
    total = reduce(op.mul, (d for i,d in enumerate(arr.shape) if i != axis))
    pm = ProgressMeter(total, 'Calculating map')
    def fit(ydata):
        pm.increment()
        return fitter(ydata)
    res = np.apply_along_axis(fit, axis, arr)
    pm.finish()
    return res


### Common Fitters ###
r2Fitter = Fitter(lambda te, so, r2: so * np.exp(-1 * r2 * te), (1,1))
r2StarFitter = Fitter(lambda te, so, r2star: so * np.exp(-1 * r2star * te), (1,1))
r2PrimeFitter = Fitter(lambda tau, so, r2prime: so * np.exp(-2 * r2prime * tau), (1,1))
