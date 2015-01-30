import cPickle as pickle
import numpy as np
from decorator import decorator
from UserDict import DictMixin
from hashlib import sha1
from collections import namedtuple
import logging
import shutil
import os.path

log = logging.getLogger('jtmri.cache')


StoreInfo = namedtuple('StoreInfo', 'disk_used,mem_used')


class DirectoryStore(object):
    '''DirectoryStore uses the filesystem as a hash table.
    It is currently not safe for concurrent usage.
    '''
    def __init__(self, path):
        self.path = os.path.expanduser(path)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _key_path(self, key):
        return os.path.join(self.path, str(key))

    def _item_path(self, key):
        return os.path.join(self._key_path(key), 'data')

    def get_size(self, path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

    def contains(self, key):
        return os.path.isdir(self._key_path(key))

    def delete_item(self, key):
        shutil.rmtree(self._key_path(key))

    def set_item(self, key, item):
        if not self.contains(key):
            os.makedirs(self._key_path(key))
        with open(self._item_path(key), 'w') as f:
            f.write(item)
            f.flush()
            f.close()

    def get_item(self, key):
        item = None
        if self.contains(key):
            with open(self._item_path(key)) as f:
                item = f.read()
                f.close()
        return item

    def info(self):
        return StoreInfo(disk_used=self.get_size(self.path), mem_used=0)

    def keys(self):
        return os.listdir(self.path)


class Cache(DictMixin):
    def __init__(self, store):
        self._store = store

    def __contains__(self, key):
        return self._store.contains(key)

    def stats(self):
        return self._store.info()

    def _log(self, key, msg):
        log.debug("[CACHE %s] %s [%s]" % (key, msg, self.stats()))

    def __getitem__(self, key):
        '''According to the doc for __getitem__, if a key is missing then
        a KeyError should be raised.'''
        self._log('GET', "key=%s" % str(key))
        val = self._store.get_item(key)
        if not val:
            raise KeyError(key)
        return pickle.loads(val) if val else None

    def __setitem__(self, key, item):
        self._log('SET', "key=%s" % str(key))
        self._store.set_item(key, pickle.dumps(item, pickle.HIGHEST_PROTOCOL))

    def __delitem__(self, key):
        '''According to the doc for __delitem__, if a key is missing then
        a KeyError should be raised'''
        self._log('REM', "key=%s" % str(key))
        self._store.delete_item(key)

    def keys(self):
        return self._store.keys()

    def purge(self):
        '''Delete all cached items'''
        for key in self.keys():
            self._store.delete_item(key)


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


def func_hash(func, args, kwargs):
    '''hash a function and its arg and kwargs.
    Uses just the byte code of the function for comparison so two different
    functions that have the same bytecode will hash to the same function.
    '''
    keyHash = sha1(str(make_hash((func.__code__.co_code, args, kwargs))))
    return keyHash.hexdigest()


cache = Cache(DirectoryStore('~/.local/share/jtmri/cache'))


@decorator
def persistent_memoize(func, *args, **kwargs):
    '''Use this decorator to give you function persistent 
    across interpreter sessions. If the same arguments are 
    given to your function then this should result in the 
    same value being returned. For this reason, any function 
    being memoized should be a pure function with no side effects, 
    at least if you expect consistent behaviour.
    '''
    fhash = func_hash(func, args, kwargs)
    key = '%s:%s' % (func.func_name, fhash)

    if key in cache:
        result = cache[key]
    else:
        result = cache[key] = func(*args, **kwargs)
    return result


def _memoize(func, *args, **kwargs):
    key = args, frozenset(kw.iteritems()) if kwargs else args
    cache = func.cache
    if key in cache:
        return cache[key]
    else:
        cache[key] = result = func(*args, **kwargs)
        return result


def memoize(func):
    '''Use this decorator to cache return values'''
    func.cache = {}
    return decorator(_memoize, func)
