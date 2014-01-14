import cPickle as pickle
import numpy as np
from decorator import decorator
from UserDict import DictMixin
from hashlib import sha1
import logging

log = logging.getLogger('jtmri.cache')

class Cache(DictMixin):
    def __init__(self):
        self._redis = redis.Redis()

    def __contains__(self, key):
        try:
            hasKey = key in self._redis
        except redis.exceptions.ConnectionError as e:
            hasKey = False
            log.warn("[Cache Error] Unable to connect to redis: (%s)" % e.message)
        
        return hasKey

    def _stats(self):
        info = self._redis.info()
        msg = "mem usage: %s (peak %s)" % \
                (info['used_memory_human'], info['used_memory_peak_human'])
        return msg

    def _log(self, key, msg):
        log.debug("[CACHE %s] %s [%s]" % (key, msg, self._stats()))

    def __getitem__(self, key):
        '''According to the doc for __getitem__, if a key is missing then
        a KeyError should be raised.'''
        self._log('GET', "key=%s" % str(key))

        val = self._redis.get(key)
        if not val:
            raise KeyError(key)
        return pickle.loads(val) if val else None

    def __setitem__(self, key, item):
        self._log('SET', "key=%s" % str(key))
        self._redis.set(key, pickle.dumps(item, pickle.HIGHEST_PROTOCOL))

    def __delitem__(self, key):
        '''According to the doc for __delitem__, if a key is missing then
        a KeyError should be raised'''
        self._log('REM', "key=%s" % str(key))
        self._redis.delete(key)

    def keys(self):
        return self._redis.keys()


DictProxyType = type(object.__dict__)

# TODO: This function sucks
def make_hash(o):
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that 
    contains only other hashable types (including any lists, tuples, sets, and
    dictionaries). In the case where other kinds of objects (like classes) need 
    to be hashed, pass in a collection of object attributes that are pertinent. 
    For example, a class can be hashed in this fashion:
  
    make_hash([cls.__dict__, cls.__name__])

    A function can be hashed like so:

    make_hash([fn.__dict__, fn.__code__])
    """
    
    if type(o) == DictProxyType:
        o2 = {}
        for k, v in o.items():
            if not k.startswith("__"):
                o2[k] = v
        o = opy2

    if isinstance(o, set) or isinstance(o, tuple) or isinstance(o, list):
        return tuple([make_hash(e) for e in o])

    if isinstance(o, np.ndarray):
        return sha1(o).hexdigest()
        
    if not isinstance(o, dict):
        return hash(o)

    for k, v in o.items():
        o[k] = make_hash(v)

    return hash(tuple(frozenset(o.items())))  


try:
    import redis
    _cache = Cache()
except:
    log.warn('redis module not installed, using local cache only')
    _cache = {}


def func_hash(func, args, kwargs):
    '''hash a function and its arg and kwargs.
    Uses just the byte code of the function for comparison so two different
    functions that have the same bytecode will hash to the same function.
    '''
    keyHash = sha1(str(make_hash((func.__code__.co_code, args, kwargs))))
    return keyHash.hexdigest()


@decorator
def memoize(func, *args, **kwargs):
    '''Use this decorator to give you function persistent 
    across interpreter sessions. If the same arguments are 
    given to your function then this should result in the 
    same value being returned. For this reason, any function 
    being memoized should be a pure function with no side effects, 
    at least if you expect consistent behaviour.
    '''
    fhash = func_hash(func, args, kwargs)
    key = '%s:%s' % (func.func_name, fhash)

    if key in _cache:
        result = _cache[key]
    else:
        result = _cache[key] = func(*args, **kwargs)
    return result
