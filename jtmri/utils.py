import collections
import copy
from glob import iglob
import itertools
import logging
import numpy as np
import pandas as pd
import os

log = logging.getLogger(__name__)


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
    padding    -- When extracting, adding padding around the final region.
                  Padding can be an array of len arr.ndim, where each value
                  indicates the amount of padding for that dimension
    return_idx -- (default False) When True, returns a tuple of
                  the indicies used to slice the array and the sliced array.
                  Otherwise, only the sliced array is returned.
    Returns:
    The smallest rectangular region, extracted from arr with padding 
    added to the border.
    When return_idx is True, than a tuple of the indicies used for sliceing
    and the sliced array are returned.
    '''
    if isinstance(padding, collections.Iterable):
        assert len(padding) == arr.ndim, 'padding must be same length as arr.ndim'
    else:
        padding = [padding] * arr.ndim

    def slicer(idxs, dim_len, pad):
        min = np.clip(idxs.min() - pad, 0, dim_len)
        max = np.clip(idxs.max() + 1 + pad, 0, dim_len)
        return slice(min, max)

    indices = [slicer(a, dim_len, pad)
               for a, dim_len, pad
               in zip((arr > threshold).nonzero(), arr.shape, padding)]
    if return_idx:
        return arr[indices], indices
    else:
        return arr[indices]


def flatten(iterable):
    '''Flattens the iterable by one level.
    If there are non-iterable items in the zero-th level,
    they are returned unaltered.
    '''
    for sublst in iterable:
        try:
            for x in sublst:
                yield x
        except TypeError:
            yield sublst


class Lazy(object):
    '''A lazy loaded field'''
    def __init__(self, func):
        self._func = func

    def __call__(self, *args, **kwargs):
        log.debug('Lazy class called {!r}'.format(self))
        return self._func(*args, **kwargs)


class AttributeDict(collections.MutableMapping):
    '''A dictionary that can have its keys accessed as if they are attributes'''
    # Prevents an inifinite recursion in __getattr__ when loaded by pickle.
    # Pickle does not call the __init__ method and so there will be no _store
    # attribute when __getattr__ is called. __getattr__ requests the _store
    # attribute and since this does not exist, __getattr__ will be called again
    # to resolve the missing attribute, leading to an infinite recursion.
    _store = {}

    # collection.MutableMapping requires:
    # __delitem__, __setitem__, __getitem__, __iter__, __len__
    #
    # and provides:
    # __contains__, keys, items, values, get, __eq__, __ne__, pop, popitem, clear, update, and setdefault

    def __init__(self, dic):
        self.__dict__['_store'] = dic

    def __getitem__(self, key):
        val = self._store[key]
        if isinstance(val, Lazy):
            val = val()
            self.__setitem__(key, val)
        return val

    # Contains is implemented here (as opposed to using the mixin from MutableMapping),
    # because the default implementation calls __getitem__ and this will auto load Lazy
    # classes. So a version of __contains__ is implemented here that just checks the _store
    def __contains__(self, key):
        return key in self._store

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

    def __repr__(self):
        return 'AttributeDict(' + repr(self._store) + ')'


class DefaultAttributeDict(AttributeDict):
    '''An AttriuteDict that returns attribute dicts that return attribute dicts ...
    when keys are missing'''
    def __init__(self, *args, **kwargs):
        self.__dict__['_store'] = collections.defaultdict(DefaultAttributeDict, dict(*args, **kwargs))


def as_iterable(val):
    '''If val is not iterable then return it wrapped in a list,
    otherewise just return val, strings will be wraped in a list as well'''
    return val if is_sequence(val) else [val]


def rep(self, attrs):
    '''Create a repr of a property based class quickly
    Args:
        self  -- pass the self reference for the class here
        attrs -- list of attributes to add to the representation

    Returns:
    A string representing the class
    '''
    s = self.__class__.__name__
    s += '(%s)' % ','.join(['%s=%r' % (attr, getattr(obj, attr)) for attr in attrs])
    return s


class GenLen(object):
    """Wrapper for a generator that has a known length"""

    def __init__(self, iterator, length):
        self._iterator = iterator
        self._length = length

    def __iter__(self):
        return self._iterator

    def __len__(self):
        return self._length


class ListAttrAccessor(object):
    '''Access the same attribute from each object in a list'''

    def __init__(self, item_list, attributes=dir, unique=False):
        '''Init ListAttrAccessor.
        Args:
            item_list  -- list of items that have common attributes
            attributes -- (default: None) explicit list of attributes, None (all attributes) or a function
            unique     -- (default: dir) Only return unique values, preservering order
        The first item is used to get the attributes that are supported unless the attributes
        option is supplied.
        Usage:
        >>> l = ListAttributeAccess([{'a':i, 'b':i**2} for i in range(5)], attributes=lambda o: o.keys())
        >>> l.a
        np.array([0, 1, 2, 3, 4, 5])
        >>> l.b
        np.array([0, 1, 4, 9, 16, 25])

        Attributes are automatically pulled from the first item in item_list using dir.
        Attributes starting with and underscore ('_') are ignored.
        '''
        self._attributes = []
        if isinstance(attributes, collections.Iterable):
            self._attributes = attributes
        elif len(item_list) > 0:
            self._attributes = attributes(item_list[0])
        # Filter out private attributes
        self._attributes = filter(lambda s: not s.startswith('_'), self._attributes)
        self._unique = unique
        self._item_list = item_list

    def __dir__(self):
        return self.__dict__.keys() + self._attributes

    def __getattr__(self, attr):
        vals = []
        for obj in self._item_list:
            val = getattr(obj, attr)
            vals.append(val)
        if self._unique:
            vals = unique(vals)
        return np.array(vals)


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
    from fuzzywuzzy import fuzz
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


def let(*args, **kwargs):
    '''bind a value in the arguments and pass it to function'''
    func = kwargs.pop('func')
    return func(*args, **kwargs)


def is_sequence(x):
    '''Returns true for all iterables that are not Strings'''
    return isinstance(x, collections.Iterable) and not isinstance(x, basestring)


def config_logger(level=logging.INFO):
    logging.basicConfig(format="%(asctime)s.%(msecs)03d::%(name)s::%(levelname)s -- %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().setLevel(level)
    log.info('Log level set to: %s' % level)


def path_generator(path, recursive=False):
    """Generate paths starting at the base path and recursing as requested
    Args:
        path      -- glob like path describer (~ is expanded), or iterable of paths
                     if None, the current directory is used
        recursive -- (default: False) recurse into all paths described by path
    
    Return: Generates a sequence of paths
    """
    if path is None:
        path = os.path.abspath(os.path.curdir)

    for path in as_iterable(path):
        path = os.path.expanduser(path)

        if os.path.isdir(path):
            path = os.path.join(path, '*') 
        
        for p in iglob(path):
            # Generate all paths for directories too if recursive is enabled
            if recursive and os.path.isdir(p):
                for root,_,files in os.walk(p):
                    for f in files:
                        yield os.path.join(root,f)
            else:
                yield p


class OrderedDefaultDict(collections.OrderedDict):
    def __init__(self, *args, **kwargs):
        if not args:
            self.default_factory = None
        else:
            if not (args[0] is None or callable(args[0])):
                raise TypeError('first argument must be callable or None')
            self.default_factory = args[0]
            args = args[1:]
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)

    def __missing__ (self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = default = self.default_factory()
        return default

    def __reduce__(self):  # optional, for pickle support
        args = (self.default_factory,) if self.default_factory else ()
        return self.__class__, args, None, None, self.iteritems()


def getattr_nested(obj, attrs):
    """Support getting nested attributes
    Args:
        obj   -- object to get attributes from
        attrs -- attribute accessor string
                 This should be a dot separated list of attributes to access,
                 just like when typeing the code out

    Examples
    >>> A = namedtuple('A', 'x,y')
    >>> a = A(A(1, 2), A(3, 4))
    >>> getattr_nested(a, 'x.x')
    1
    >>> getattr_nested(a, 'y.x')
    3
    """
    for attr in attrs.split('.'):
        obj = getattr(obj, attr)
    return obj


class Grouped(collections.Mapping):
    """A grouped object is created from a dictionary whose keys are tuples and values
    are iterables

    Keys can be accessed sequentially.
    >>> g = Grouped({(0,1):['a','b'], (0,2):['a,'c']})
    >>> g[0][1]
    ['a', 'b']
    >>> g[0,1]
    ['a', 'b']
    """

    def __init__(self, dic, _keylen=None):
        self._dic = collections.OrderedDict(dic)
        self._keylen = _keylen or len(self._dic.iterkeys().next())

    def __iter__(self):
        return iter(self._dic)

    def __len__(self):
        return len(self._dic)

    def __getitem__(self, key):
        key = tuple(as_iterable(key))
        keylen = len(key)
        if keylen < self._keylen:
            return Grouped(((k[keylen:], v) for k, v in self._dic.iteritems() if k[:keylen] == key),
                           self._keylen - keylen)
        return self._dic[key]

    def apply(self, applier):
        """Apply a function to each value and return a Grouped object
        Args:
            applier -- a func or a list of funcs or a iterable of name, func pairs
                       func can also be a string for attribute access
        """
        def func(v, applier=applier):
            vals = []
            for f in as_iterable(applier):
                try:
                    key, f = f
                    vals.append((key, f(v)))
                except (TypeError, ValueError):
                    if callable(f):
                        vals.append(f(v))
                    else:
                        vals.append(getattr_nested(v, f))
            if len(vals) == 1:
                return vals[0]
            return vals
        return Grouped(((k, func(v)) for k, v in self._dic.iteritems()), self._keylen)
        


def groupby(iterable, grouper=lambda x: x, group_type=list):
    """Group items in an iterable together by key, no presorting necessary
    Args:
        iterable    -- an iterable
        grouper     -- (default: ident func) a callable, a single attribute or a list of attributes and callables
        group_type  -- (default: list) set the container type for each group returned
        nested      -- (default: False) This will nest each group in another dictionary
    
    Returns:
        OrderedDict sorted on the group keys

    Examples:
    # Group integers by parity
    >>> groupby(range(10), lambda x: x % 2 == 0)
    OrderedDict([((False,), [1, 3, 5, 7, 9]), ((True,), [0, 2, 4, 6, 8])])

    # Group objects by an attribute
    >>> from collections import namedtuple
    >>> A = namedtuple('A', 'field')
    >>> groupby(map(A, range(3)), 'field')
    OrderedDict([((0,), [A(field=0)]), ((1,), [A(field=1)]), ((2,), [A(field=2)])])

    # Groupby by multiple attributes or functions
    >>> Square = namedtuple('Square', 'color,h,w')
    >>> lst = [Square('red', 10, 20), Square('green', 10, 20), Square('red', 20, 10), Square('green', 20, 10)]
    >>> # Group by color and area
    >>> groupby(lst, ['color', lambda s: s.h * s.w])
    OrderedDict([(('green', 200),
                  [Square(color='green', h=10, w=20),
                   Square(color='green', h=20, w=10)]),
                 (('red', 200),
                  [Square(color='red', h=10, w=20),
                   Square(color='red', h=20, w=10)])])

    # Output can be controlled with group_type and nested.
    # Nesting allows each selection of groups
    >>> groupby(lst, ['color', lambda s: s.h * s.w], group_type=tuple, nested=True)
    OrderedDict([('red',
                  OrderedDict([(200,
                                [Square(color='red', h=10, w=20),
                                 Square(color='red', h=20, w=10)])])),
                 ('green',
                  OrderedDict([(200,
                                [Square(color='green', h=10, w=20),
                                 Square(color='green', h=20, w=10)])]))])
    """
    keyfuncs = []
    for f in as_iterable(grouper):
        if callable(f):
            keyfuncs.append(f)
        elif isinstance(f, basestring):
            keyfuncs.append(lambda x, f=f: getattr_nested(x, f))
        else:
            raise TypeError('{!r} of type {!r} cannot be used in a grouper'.format(f, type(f)))
    out = OrderedDefaultDict(list)
    for item in iterable:
        key = tuple(f(item) for f in keyfuncs)
        out[key].append(item)
    return Grouped((k, group_type(v)) for k, v in out.iteritems())
