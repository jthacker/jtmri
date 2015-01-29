import numpy as np 
import pylab as plt 
import collections
from fuzzywuzzy import fuzz
import os, sys, logging, csv, re, itertools, collections, copy


log = logging.getLogger('jtmri.utils')


class ProgressMeter(object):
    '''Displays a CLI progress meter.
    maxVal is the final value of the parameter being incremented.
    msg will be displayed while the meter is progressing.'''

    def __init__(self, count, msg='working', width=20):
        # Progress is in the range [0, count)
        self._fd = sys.stderr
        self._progress = 0
        self._count = float(count)
        self._msg = msg
        self._width = width
        self._finished = False
        self._errorCount = 0
        self._display(self._msg)

    def _display(self, msg):
        self._fd.write('\x1b[2K') # Delete current line
        if self._count != 0:
            progress = self._progress / self._count
        else:
            progress = 1.0
        markerWidth = int(progress*self._width)
        progressStr =  '#' * markerWidth
        progressStr += ' ' * (self._width - markerWidth)
        self._fd.write("\r[%s] %5.1f%% -- %s" % (progressStr, 100*progress, msg))
        self._fd.flush()
        
    def _end(self, msg):
        '''Call this if the task finishes earlier than expected'''
        self._progress = self._count
        self._display('%s (errors: %d)\n' % (msg, self._errorCount))

    def increment(self):
        '''Increment the internal value and update the display'''
        if self._progress + 1 >= self._count:
            self._progress = self._count
        else:
            self._progress += 1
        self._display(self._msg) 

    def finish(self, success=True):
        '''Call this if the task finishes succssefully earlier than expected'''
        msg = 'finished' if success else 'failed'
        self._end('(%s) %s' % (msg, self._msg))

    def error(self, msg="Error!"):
        '''Call this if there is a recoverable error while procressing'''
        self._errorCount += 1
        self._display(msg)


def unique(seq):
    '''find unique elements in iterable while preserving order'''
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]


def interleave(*args):
    '''Interleave two or more lists into a single list.
    If the lists are uneven then the remaining values are appened to the
    end of the output list.
    Returns a generator on the input sequences.'''
    return (i for j in itertools.izip_longest(*args) for i in j if i != None)


def chunks(l, n):
    ''' Yield successive n-sized chunks from l. '''
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def extract(arr, threshold=0, padding=0, return_idx=False):
    '''Find the smallest rectangular region in the array that can hold
    all values > thresh
    Args:
    arr        -- Array to extract values from
    threshold  -- Values <= threshold are considered background
    padding    -- When extracting, adding padding around the final region
    return_idx -- (default False) When True, returns a tuple of
                  the indicies used to slice the array and the sliced array.
                  Otherwise, only the sliced array is returned.
    Returns:
    The smallest rectangular region, extracted from arr with padding 
    added to the border.
    When return_idx is True, than a tuple of the indicies used for sliceing
    and the sliced array are returned.
    '''
    def slicer(idxs, dim_len):
        min = np.clip(idxs.min() - padding, 0, dim_len)
        max = np.clip(idxs.max() + 1 + padding, 0, dim_len)
        return slice(min, max)

    indices = [slicer(a, dim_len) for a,dim_len in zip((arr > threshold).nonzero(), arr.shape)]
    if return_idx:
        return arr[indices], indices
    else:
        return arr[indices]


def flatten(iterable):
    '''Flattens the iterable by one level'''
    return (x for sublst in iterable for x in sublst)


class Lazy(object):
    '''A lazy loaded field'''
    def __init__(self, func):
        self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class AttributeDict(collections.MutableMapping):
    '''A dictionary that can have its keys accessed as if they are attributes'''
    # Prevents an inifinite recursion in __getattr__ when loaded by pickle.
    # Pickle does not call the __init__ method and so there will be no _store
    # attribute when __getattr__ is called. __getattr__ requests the _store
    # attribute and since this does not exist, __getattr__ will be called again
    # to resolve the missing attribute, leading to an infinite recursion.
    _store = {}  

    def __init__(self, dic):
        self.__dict__['_store'] = dic

    def __getitem__(self, key):
        val = self._store[key]
        if isinstance(val, Lazy):
            val = val()
            self.__setitem__(key, val)
        return val

    def __setitem__(self, key, val):
        self._store[key] = val

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __getattr__(self, key):
        if key in self._store:
            return self.__getitem__(key)
        else:
            raise AttributeError("'AttributeDict' object has no attribute %r" % key)

    def __setattr__(self, key, val):
        if not key.startswith('_'):
            self._store[key] = val
        else:
            super(object, self).__setattr__(key, val)

    def __dir__(self):
        return sorted(set(dir(type(self)) + self.keys()))

    def dict(self):
        return copy.deepcopy(self._store)
   
    def __str__(self):
        return 'AttributeDict(' + str(self._store) + ')'


class DefaultAttributeDict(AttributeDict):
    '''An AttriuteDict that returns attribute dicts that return attribute dicts ...
    when keys are missing'''
    def __init__(self, *args, **kwargs):
        self.__dict__['_store'] = collections.defaultdict(DefaultAttributeDict, dict(*args, **kwargs))
        

def as_iterable(val):
    '''If val is not iterable then return it wrapped in a list,
    otherewise just return val'''
    if not isinstance(val, collections.Iterable):
        return [val]
    else:
        return val


def rep(self, props):
    '''Create a repr of a property based class quickly
    Args:
    self  -- pass the self reference for the class here
    props -- list of properties to add to the representation

    Returns:
    A string representing the class
    '''
    s = self.__class__.__name__
    s += '(%s)' % ','.join(['%s=%r' % (prop,getattr(obj,prop)) for prop in props])
    return s


class GenLen(object):
    def __init__(self, iterator, length):
        self._iterator = iterator
        self._length = length

    def __iter__(self):
        return self._iterator

    def __len__(self):
        return self._length


class ListAttrAccessor(object):
    '''Access the same attribute from each object in a list'''

    def __init__(self, lst):
        self._obj = lst[0] if len(lst) > 0 else {}
        self._attrs = []
        self._attrs = self._obj.keys()
        self._lst = lst 

    def __dir__(self):
        return self.__dict__.keys() + self._attrs

    def __getattr__(self, attr):
        return np.array([d[attr] if attr in d else None for d in self._lst])


def similar(s1, s2, threshold=90):
    '''Fuzzy comparison function
    Args:
    s1,s2     -- strings to compare
    threshold -- (default: 90) threshold for signifiance, 
                 should be between 0 and 100
    Returns:
    A predicate function that returns True if the 
    argument is similar to the matcher otherwise False.
    '''
    return fuzz.partial_ratio(s1, s2) >= threshold


def filter_error(func, catch=(RuntimeError,), retval=False):
    '''Wraps a function and performs the following:
    Func is called and if an exception is thrown, then return retval,
    otherwise return the result of calling func.
    Args:
    func   -- Function to wrap
    catch  -- List of exceptions to catch
    retval -- (default False) Returned value when an exception is caught'''

    def _filter_error(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
        except catch:
            return retval
        return res
    return _filter_error
