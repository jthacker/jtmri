import numpy as np

gamma = 2 * np.pi * 42.57 # MHz / T

def _minPositive(x):
    """Returns the smallest positive number in the array"""
    return min(x[x > 0])

def getGradientWidth_rect(b, separation, gradAmplitude):
    """Get the width of the rectangular gradient pulses from the given b-value.
    b and separation should be in milliseconds and the gradient 
    amplitude (gradAmplitude) should be in milliTesla per millimeter

    b = gamma**2 * G**2 * width**2 * ( separation - width/3 )
    
    returns the smallest positive solution
    """

    c = gamma**2 * gradAmplitude**2
    roots = np.roots([-1 * c/3.0, c*separation, 0, -b * 10**9])
    return _minPositive(roots)



def getGradientWidth_trap(b, separation, gradAmplitude, slewrate):
    """Get the width of the trapezoidal gradient pulses from the given b-value.
    b, separation and slewrate should be in milliseconds
    gradAmplitude should be in millitesla per millimeter
    
    returns the smallest positive solution"""

    c = gamma**2 * gradAmplitude**2
    roots = np.roots([-c/3.0, c*separation, -c*slewrate**2/6.0, -b*10**9 + c*slewrate**3/30.0])
    return _minPositive(roots)
