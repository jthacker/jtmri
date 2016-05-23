from ..fit import fit_r2star_fast
import numpy as np
from numpy.testing import assert_almost_equal

def test_fit_r2star_fast():
    t = np.linspace(0,1,8)
    r2star = np.array([[[10.0, 42.2],
                        [ 2.4, 52.9]],
                       [[16.7,  1.0],
                        [22.5, 34.6]]])
    arr = np.exp(-r2star[...,np.newaxis] * t)
    r2star_fit, residuals = fit_r2star_fast(t, arr)
    assert_almost_equal(r2star, r2star_fit)


def test_fit_r2star_fast_nondefault_axis():
    t = np.linspace(0,1,8)
    r2star = np.arange(3*4*5).reshape((3,4,5))
    arr = np.exp(-r2star[...,np.newaxis] * t).swapaxes(2,-1)

    r2star_fit, residuals = fit_r2star_fast(t, arr, axis=2)
    assert_almost_equal(r2star, r2star_fit)


def test_r2star_fit_fast_masked_input():
    t = np.linspace(0,1,8)
    mask = np.ones((3,4))
    mask[:,0] = False
    r2star = np.ma.array(np.arange(3*4).reshape((3,4)), mask=mask)
    arr = np.exp(-r2star[...,np.newaxis] * t)

    r2star_fit, residuals = fit_r2star_fast(t, arr)
    assert_almost_equal(r2star, r2star_fit)
    assert_almost_equal(r2star.mask, r2star_fit.mask)
