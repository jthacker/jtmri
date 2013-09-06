import inspect
import numpy as np
import pylab as plt
import logging
import operator as op
from scipy.optimize import curve_fit
from jtmri.cache import memoize
from jtmri.utils import ProgressMeter

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
        
        funcOvershoot: amount to extend the x range when plotting the fitted function'''

        plt.figure()
        title = 'Fit: ' + " ".join(["%s=%s" % (k,v) for k,v in self.paramDict.iteritems()])
        title = title if self.success else "FAILED " + title

        plt.title(title)
        plt.plot(self.x, self.y, 'bo', label='data')

        # Plot the fitted function with fixed sampling and a range extended by funcOvershoot
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
        return "Fit(x=%s, y=%s, func=%s, fittedParams=%s, covMatrix=%s, success=%s)" % (self.x, self.y, self.func, self.paramDict, self.covMatrix, self.success)


class Fitter(object):
    def __init__(self, fitFunc, failedFitFunc=_minValueFailedFitFunc):
        '''Create a new Fitter object.
        fitFunc should be of the form lambda xData, param0, param1, ... : function(xData, param0, param1, ...)
        The first argument should be the xData used for fitting.
        The Fitter object will attempt to find estimates for rest of the specified parameters.
        If the Fitter fails to find a fit, then failedFitFunc is called to get default values for those points.
        '''
        fitFuncArgs = inspect.getargspec(fitFunc).args

        assert len(fitFuncArgs) > 1, "The fitFunc should take at least 2 arguments\
            (the first is the xdata and the rest are parameters to be fit),\
            only %d specified." % len(fitFuncArgs)
        
        self._fitFunc = fitFunc
        self._paramsToFit = fitFuncArgs[1:]
        self._failedFitFunc = failedFitFunc

    def __call__(self, initialGuess, xdata, ydata):
        '''Find estimates for the parameters of the fitFunc given the initalGuess, xdata and ydata.
        Thie initalGuess is needed inorder for the nonlinear fitting function to converge.
        It should be the same length as the number of parameters specifed in the fitFunc.'''

        xdata = np.array(xdata)
        ydata = np.array(ydata)

        assert xdata.ndim == 1, "xdata must be one dimensional but has %d dimensions" % xdata.ndim
        assert ydata.ndim == 1, "ydata must be one dimensional but has %d dimensions" % ydata.ndim

        assert len(xdata) == len(ydata), "xdata and ydata must have the same length, %d != %d" % (len(xdata), len(ydata))

        assert len(initialGuess) == len(self._paramsToFit), """len(initalGuess)=%d \
            should equal the number of free parameters to be fit (%d) \
            which were specified by the fitFunc""" % (len(initalGuess), len(self._paramsToFit))

        if np.isscalar(initialGuess):
            initialGuess = np.array(initialGuess)

        successfulFit = True

        try:
            popt,pcov = curve_fit(self._fitFunc, xdata, ydata, initialGuess)
        except RuntimeError as e:
            successfulFit = False
            log.warn("Failed to find an appropriate fit, using the default value. %s" % e)
            popt,pcov = self._failedFitFunc(initialGuess)

        if np.isscalar(pcov):
            successfulFit = False
            assert pcov == np.inf
            log.warn("Failed to find an appropriate fit, using the default value.")
            pcov = self._failedFitFunc(initialGuess)[1]

        fit = Fit(xdata, ydata, self._fitFunc, popt, pcov, successfulFit)
        log.debug("Fit [%s] params = %s" % ('succeeded' if fit.success else 'FAILED', fit.paramDict))
        return fit

def fitMapper(fitter, images, x, guess):
    '''Fitter is a function that takes a numpy array of y values
    and returns the parameter that was fit.
    TODO: fit func needs to return the proper thing
    '''
    pm = ProgressMeter(reduce(op.mul, images[0].shape), 'Calculating map')

    def fit(ydata):
        pm.increment()
        return fitter(guess, x, ydata).params[1]

    res = np.apply_along_axis(fit, 0, images)
    pm.finishSuccess()
    
    return res


### Common Fitters ###
r2Fitter = Fitter(lambda te, so, r2: so * np.exp(-1 * r2 * te))
r2StarFitter = Fitter(lambda te, so, r2star: so * np.exp(-1 * r2star * te))
r2PrimeFitter = Fitter(lambda tau, so, r2prime: so * np.exp(-2 * r2prime * tau))
