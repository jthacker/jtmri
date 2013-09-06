import numpy as np
from numpy.fft import ifftn, fftn, irfftn, rfftn, ifftshift
from numpy import array
from itertools import *
import operator as op

GS = 1

#def grid(shape):
#    mz,my,mx = map(lambda x: x/2, shape)
#    return np.mgrid[-mz:mz,-my:my,-mx:mx] 

def grid((nz,ny,nx)):
    j = 1.j
    return np.mgrid[-GS:GS:nz*j, -GS:GS:ny*j, -GS:GS:nx*j]

def _l2_norm(*args):
    return np.sum(np.array(args)**2, axis=0)

def _linf_norm(*args):
    return np.max(np.array(map(np.abs , args)), axis=0)

def nu((nz,ny,nx)):
    z,y,x = grid((nz,ny,nx))
    nu = (np.pi * 4)**-1 * (2*z**2 - x**2 - y**2) / _l2_norm(z,y,x)**(5/2.0)
    nu[nz/2, ny/2, nx/2] = 0
    return nu

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) / 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def sphere_theory(shape, radius):
    z,y,x = grid(shape)
    r2 = _l2_norm(z,y,x)
    sol = (1/3.0) * radius**3 * (2*z**2 - x**2 - y**2) / r2**(5/2.0)
    sol[ r2 <= radius**2 ] = 0
    return sol

def sphere(shape,radius):
    sphere = np.zeros(shape)
    sphere[ _l2_norm(*grid(shape)) <= radius**2 ] = 1
    return sphere

def square(shape, radius):
    sq = np.zeros(shape)
    sq[ _linf_norm(*grid(shape)) <= radius ] = 1
    return sq

def random(shape, num):
    chi = np.zeros(shape)
    idx = np.arange(chi.size)
    np.random.shuffle(idx)
    chi.ravel()[idx[:num]] = 1
    return chi

def wiener_deconvolve(x, kernel, l):
    s1 = array(x.shape)
    s2 = array(kernel.shape)
    size = s1 + s2 - 1

    X = fftn(x, size)
    H = fftn(kernel, size)
    Hc = np.conj(H)

    ret = ifftshift(ifftn( X*Hc / (H*Hc*X + l**2))).real
    return _centered(ret, s1)

def fftconvolve(in1, in2, mode='same'):
    """Convolve two N-dimensional arrays using FFT. See convolve.

    """
    s1 = array(in1.shape)
    s2 = array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))
    size = s1 + s2 - 1

    # Always use 2**n-sized FFT
    fsize = (2 ** np.ceil(np.log2(size))).astype('int')
    IN1 = fftn(in1, fsize)
    IN1 *= fftn(in2, fsize)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = ifftn(IN1)[fslice].copy()
    del IN1 
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if np.product(s1, axis=0) > np.product(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return _centered(ret, osize)
    elif mode == "valid":
        return _centered(ret, abs(s2 - s1) + 1)

    return conv[:s[0], :s[1], :s[2]]
