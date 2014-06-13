import numpy as np

def cohens_d(a, b):
    '''Cohen's D computed between two list like sequences''' 
    return (np.mean(a) - np.mean(b)) / ((np.var(a) + np.var(b)/2.0))**0.5
