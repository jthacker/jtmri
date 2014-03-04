import numpy as np 
import pylab as plt 
from fuzzywuzzy import fuzz
import os, sys, logging, csv, re, itertools, collections


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


def extract(m, thresh=0):
    '''Find the smallest rectangular region in the matrix that can hold
    all values > thresh'''
    indx = [slice(a.min(),a.max()+1) for a in (m > thresh).nonzero()]
    return m[indx]


def flatten(iterable):
    '''Flattens the iterable by one level'''
    return (x for sublst in iterable for x in sublst)


class AttributeDict(object):
    '''A dictionary that can have its keys accessed as if they are attributes'''
    def __init__(self, dic):
        self.__dict__['_dict'] = dic
        self.values = dic.values
        self.keys = dic.keys
        self.get = dic.get
        self.iteritems = dic.iteritems
    
    def __dir__(self):
        return sorted(set(dir(type(self)) + self._dict.keys()))

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val

    def __getattr__(self, key):
        if key in self._dict:
            return self._dict[key]
        else:
            super(AttributeDict, self).__getattr__(key)

    def __setattr__(self, key, val):
        if not key.startswith('_'):
            self._dict[key] = val
        else:
            return super(AttributeDict, self).__setattr__(key, val)

    def update(self, *args, **kwargs):
        self._dict.update(*args, **kwargs)
        return self

    def __iter__(self):
        return self._dict.__iter__()

    def __delitem__(self, key):
        del self._dict[key]
   
    def __str__(self):
        return 'AttributeDict(' + str(self._dict) + ')'


class DefaultAttributeDict(AttributeDict):
    '''An AttriuteDict that returns attribute dicts that returns attribute dicts ...
    when keys are missing'''
    def __init__(self, *args, **kwargs):
        self.__dict__['_dict'] = collections.defaultdict(DefaultAttributeDict, dict(*args, **kwargs))
        

def asiterable(val):
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
    '''Access the same attribute in a list of objects'''

    def __init__(self, lst):
        obj = lst[0] if len(lst) > 0 else {}
        self._attrs = []
        self._attrs = obj.keys()
        self._lst = lst 

    def __dir__(self):
        return self.__dict__.keys() + self._attrs

    def __getattr__(self, attr):
        if len(self._lst) == 0:
            return []

        if attr not in self._attrs:
            raise AttributeError("%r has no attribute %r" % attr)

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

