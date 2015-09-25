from __future__ import absolute_import
from glob import iglob
import os, dicom, logging
import numpy as np
import copy
import shutil
import cPickle as pickle

from itertools import ifilter, chain
from collections import defaultdict, Iterable
from prettytable import PrettyTable
import arrow

from .siemens import SiemensProtocol
from ..progress_meter import progress_meter_ctx
from ..utils import (unique, AttributeDict, ListAttrAccessor, Lazy,
                     is_sequence)
from . import dcminfo


log = logging.getLogger('jtmri:dicom')


CACHE_FILE_NAME = '.cache'

class DicomParser(object):
    @staticmethod
    def to_attributedict(dcm):
        return DicomParser.to_dict(dcm, AttributeDict)

    @staticmethod
    def to_dict(dcm, wrap=dict):
        d = DicomParser._dicom_to_dict(dcm, wrap)
        t = arrow.get(d['InstanceCreationDate'] + d['InstanceCreationTime'], 'YYYYMMDDHHmmss.SSSSSS')
        d['InstanceTimestamp'] = t.timestamp + t.microsecond / 1e6
        d['Siemens'] = wrap(SiemensProtocol.from_dicom(dcm))
        pixel_array = None
        try:
            pixel_array = dcm.pixel_array
        except TypeError:
            pass
        d['pixel_array'] = pixel_array
        return d
    
    @staticmethod
    def _convert(val):
        if isinstance(val, list):
            val = map(DicomParser._convert, val)
        elif isinstance(val, dicom.valuerep.DSfloat):
            val = float(val)
        elif isinstance(val, dicom.valuerep.IS):
            val = int(val)
        elif isinstance(val, dicom.UID.UID):
            val = str(val)
        else:
            val = str(val)
        return val

    @staticmethod
    def _dicom_to_dict(dcm, wrap):
        d = {}
        # :WARNING:
        # dcm must be iterated over in this manner with
        # the elem being accessed through __getitem__ because
        # otherwise the raw element is returned and it hasn't
        # been parsed yet
        for tag in dcm.iterkeys():
            elem = dcm[tag]
            name = dicom.datadict.keyword_for_tag(tag)
            if name: # remove blank tags
                val = elem.value
                if tag != (0x7fe0, 0x0010):
                    if isinstance(val, dicom.sequence.Sequence):
                        val = [DicomParser._dicom_to_dict(ds, wrap=lambda x:x) for ds in val]
                    else:
                        val = DicomParser._convert(val)
                    d[name] = val
        return wrap(d)


class DicomSet(object):
    def __init__(self, dcms=tuple()):
        self._dcms = list(dcms)
        key = lambda d: (d.StudyInstanceUID, d.SeriesNumber, d.InstanceNumber, d.InstanceTimestamp)
        self._dcms.sort(key=key)

        self._cache_series = defaultdict(list)
        for d in self._dcms:
            self._cache_series[d.SeriesNumber].append(d)

    @property
    def first(self):
        return self[0]

    @property
    def all(self):
        '''Access an attribute from all objects in this set'''
        return ListAttrAccessor(self) 

    @property
    def all_unique(self):
        '''Access an attribute from all objects, but only return unique values,
        preserves the order they are passed in.
        '''
        return ListAttrAccessor(self, unique=True)

    def filter(self, func):
        '''Filter the dicoms in this dataset
        Args:
        func -- predicate function for filtering

        Returns:
        A DicomSet with dicoms matching the predicate
        '''
        return DicomSet(ifilter(func, self))

    def groupby(self, key):
        return groupby(self, key=key, outtype=DicomSet)

    def by_series(self, *nums):
        return DicomSet(chain.from_iterable(self._cache_series.get(n, tuple()) for n in nums))

    def by_studyid(self, *studyids):
        return self.filter(lambda d: d.StudyID in studyids)

    def series(self):
        for num in unique(self.all.SeriesNumber):
            yield self.by_series(num)

    def studies(self):
        for sid in unique(self.all.StudyInstanceUID):
            yield self.filter(lambda d: d.StudyInstanceUID == sid)

    def disp(self, extra_columns=tuple()):
        disp(self, extra_columns)

    def view(self, groupby=tuple(), roi_filename=None, roi_tag=None):
        return view(self, groupby, roi_filename, roi_tag)

    def data(self, groupby=tuple()):
        return data(self, field='pixel_array', groupby=groupby)

    def cp(self, dest):
        '''Copy dicom files to dest directory'''
        return dcm_copy(self, dest)
   
    @property
    def count(self):
        return len(self)

    def __iter__(self):
        return iter(self._dcms)

    def __getitem__(self, key):
        return self._dcms[key]

    def __len__(self):
        return len(self._dcms)


def isdicom(path):
    '''Check if path is a dicom file based on the file magic'''
    isdcm = False
    if os.path.isfile(path):
        with open(path,'r') as fp:
            fp.seek(0x80)
            magic = fp.read(4)
            isdcm = (magic == b"DICM")
    return isdcm


def groupby(dicoms, key=lambda x: x, sort=True, outtype=list):
    '''Group dicoms together by key, no presorting necessary
    Args:
    dicoms  -- an iterable of dicoms
    key     -- a callable, a single key or a list of keys and callables
    sort    -- sort the groups by key

    Returns:
    A list of tuples with the structure ((key0, group0), (key1, group1), ... )

    Examples:
    # Group by the key SeriesNumber
    >>> groupby(dicoms, 'SeriesNumber')
    [(0, [dcm1, dcm2, ...]), (1, [dcm3, dcm4]), (2, [dcm5, dcm6]), ...]
    
    # Group even and odd SeriesNumbers, results are based on the series numbers
    # from the previous examples (eg. dcm1 and dcm2 have series number 0)
    >>> groupby(dicoms, lambda x: x.SeriesNumber % 2 == 0)
    [(True, [dcm1, dcm2, dcm5, dcm6, ...]), (False, [dcm3, dcm4, dcm7, dcm8, ...])]

    # Groupy by the key StudyInstanceUID then SeriesInstanceUID
    >>> groupby(dicoms, ('StudyInstanceUID', 'SeriesInstanceUID'))
    [(0, [(0, [dcm1]), (1, [dcm2])]), (1, [(0, [dcm3]), (1, [dcm4])])]
    '''
    keyfunc = None
    extrakeys = None
    if callable(key):
        keyfunc = key
    else:
        if isinstance(key, Iterable) and not isinstance(key, str):
            key,extrakeys = key[0],key[1:]
        keyfunc = lambda x, key=key: x.get(key)

    d = defaultdict(outtype)
    for dcm in dicoms:
        d[keyfunc(dcm)].append(dcm)
    
    items = []
    for k,v in sorted(d.items(), key=lambda x: x[0]):
        if extrakeys:
            items.append((k, groupby(v, extrakeys)))
        else:
            items.append((k, v))
    return tuple(items)


def _newshape(initialshape, keys, groupby):
    newshape = list(initialshape)
    for i,s in enumerate(groupby):
        if s == '*':
            newshape.append(-1)
        else:
            newshape.append(len(set(k[i] for k in keys)))
    return tuple(newshape)


def data(iterable, field, groupby=tuple(), reshape=True):
    '''Get the field attribute from all objects, grouping and reshaping if specified
    Args:
    field   -- Name of a field on the object to get value from
    groupby -- A list or tuple of fields to group the data with
    reshape -- (default True) Reshape the data if specified

    Returns:
    A numpy array grouped by groupby and reshaped to fit if specified.
    If reshape is False, then data will be appended on the last
    dimension (second dimension for 1D data and third for 3D data), 
    and sorted by the groupby fields.

    If the data from all the objects in the set do not have the same 
    dimensions then this method will fail. Instead, 
    use [o.field for o in subset] to get a list.

    Examples for a set of dicoms with 5 SliceLocations and 3 
    images at each location
    >>> # Now apply data to all dicoms in the set
    >>> data(dcms, field='pixel_array').shape
    (64,64,15)
    >>> data(dcms, field='pixel_array', groupby=('SliceLocation',)).shape
    (64,64,5,3)
    '''
    groupby = (groupby,) if isinstance(groupby, basestring) else tuple(groupby)
    obj = list(iterable)
    if len(obj) > 0:
        groupby = groupby + ('*',) if '*' not in groupby else groupby
        key = lambda i,d: [i if s=='*' else d[s] for s in groupby]
        arraykeys = ((getattr(d, field),key(i,d)) for i,d in enumerate(obj))
        arrays, keys = zip(*sorted(arraykeys, key=lambda d: d[1]))
        shape = tuple() if not hasattr(arrays[0], 'shape') else arrays[0].shape
        newshape = _newshape(shape, keys, groupby)
        arrays = np.dstack(arrays)
        return arrays.reshape(newshape) if reshape else arrays
    else:
        return np.array([])


def _path_gen(path, recursive):
    path = os.path.expanduser(path) if path else os.path.abspath(os.path.curdir)

    if os.path.isdir(path):
        path = os.path.join(path, '*') 
    
    for path in iglob(path):
        if recursive and os.path.isdir(path):
            for root,_,files in os.walk(path):
                for f in files:
                    yield os.path.join(root,f)
        else:
            yield path


def _store_cache(dicomset, filename):
    '''Store a list of dicoms as a pickle object to filename
    Args:
        dicoms   -- An iterable of dicoms
        filename -- Name of file to store the dicoms in
    The pixel arrays are dropped from the dicoms to minimize the
    cached file size. They are then lazy loaded after being unpickled.
    The dicom filename attribute is stored as being relative to the .cache
    directory. Upon loading, it is restored to an absolute path.
    '''
    dcms = []
    cache_dir = os.path.dirname(filename)
    for dcm in dicomset:
        d = copy.deepcopy(dcm)
        d.filename = os.path.relpath(d.filename, cache_dir)
        del d['pixel_array']
        dcms.append(d)
    with open(filename, 'w') as f:
        pickle.dump(dcms, f, pickle.HIGHEST_PROTOCOL)


def _load_cache(filename):
    '''Load a list of dicoms from a pickle object
    Args:
        filename -- Filename of pickle object to load dicoms from

    Returns: A list of dicoms from the objects stored in the pickle object
    '''
    dcms = []
    with open(filename, 'r') as f:
        dcms = pickle.load(f)

    cache_directory = os.path.dirname(filename)
    for dcm in dcms:
        dcm.filename = os.path.abspath(os.path.join(cache_directory, dcm.filename))
        dcm.pixel_array = Lazy(lambda dcm=dcm: dicom.read_file(dcm.filename).pixel_array)
    return dcms


def _get_cached(cache, path):
    '''get path from the dicom cache, loading cache files as needed
    Args:
        cache -- dictionary like cache to check if path is cached there
        path  -- path to a dicom
    Returns:
        return a cached dicom or None if it was unable to find the dicom
    '''
    path = os.path.abspath(path)
    if path in cache:
        return cache[path]
    dirname = os.path.dirname(path)
    cache_filename = os.path.join(dirname, CACHE_FILE_NAME)
    if cache_filename not in cache and os.path.exists(cache_filename):
        log.debug('loading cache %s' % cache_filename)
        cache[cache_filename] = True
        for dcm in _load_cache(cache_filename):
            key = os.path.relpath(dcm.filename, cache_filename)
            cache[dcm.filename] = dcm
        return cache.get(path, None)
    log.debug('cache not found at {}'.format(cache_filename))
    return None


def read(path=None, disp=True, recursive=False, progress=lambda x:x, use_info=True,
         use_cache=True, make_cache=False, overwrite=False):
    '''Read dicom files from path and print a summary
    Args:
        path       -- (default: cwd) glob like path of dicom files
        disp       -- (default: True) Print a summary
        recursive  -- (default: False) Recurse into subdirectories
        progress   -- (default: None) One arg callback function (# dicoms read)
        use_info   -- (default: True) Load info from dicom info files (info.yaml)
        use_cache  -- (default: True) Use cached files to quickly load dicoms
        make_cache -- (default: False) Create a cache of dicoms if it doesn't exist
        overwrite  -- (default: False) Overwrite an existing cache
    Returns: A list of dicom objects. Prints a summary of the dicom objects if disp is True
    '''
    dcmlist = []
    dcm_cache = {}
    path = path or os.path.curdir
    paths = _path_gen(path, recursive)

    if make_cache:
        cache(path=path, recursive=recursive, disp=disp, progress=progress, overwrite=overwrite)

    with progress_meter_ctx(description='read', disp=disp) as pm:
        for p in paths:
            pm.increment()
            dcm = _get_cached(dcm_cache, p) if use_cache else None
            if dcm is None:  # path is not in cache
                if not isdicom(p):
                    log.debug('ignoring file %s' % p)
                    continue
                dcm = DicomParser.to_attributedict(dicom.read_file(p))
                dcm.filename = os.path.abspath(p)
                log.debug('loaded from disk %s' % p)
            else:
                log.debug('loaded from cache %s' % p)
            dcmlist.append(dcm)
            progress(len(dcmlist))

    dicomset = DicomSet(dcmlist)

    if use_info:
        infos = dcminfo.read(path, recursive) 
        dcminfo.update_metadata(infos, dicomset)

    if disp:
        dicomset.disp()
    return dicomset


def _path_gen_dirs(path, recursive):
    path = os.path.expanduser(path) if path else os.path.abspath(os.path.curdir)
    for p in iglob(path):
        if os.path.isdir(p):
            if recursive:
                for root,_,_ in os.walk(p):
                    yield root
            else:
                yield p


def cache(path=None, recursive=False, disp=True, overwrite=False, progress=lambda x:x):
    '''Generate dicom cache files for each directory
    Args:
        path      -- (default: cwd) path to find dicoms in.
        recursive -- (default: False) Recurse into subdirectories
        disp      -- (default: True) Display the progress of the cache generation
        overwrite -- (default: False) Overwrite existing cache files
        progress  -- (default: None) Callback for the progress, returns #dicoms read

    This command will search for directories containing dicom files and
    create a cache file for each directory.
    '''
    paths = _path_gen_dirs(path, recursive)
    with progress_meter_ctx(description='cache', disp=disp) as pm:
        def _progress(n):
            pm.increment(n)
            progress(n)

        cnt = 0
        for d in paths:
            pm.increment()
            cache_filename = os.path.join(d, CACHE_FILE_NAME)
            if not overwrite and os.path.exists(cache_filename):
                msg = 'skipping existing cache %s. ' % cache_filename
                msg += 'run with overwrite=True to refresh the cache'
                pm.set_message(msg)
                continue
            dcms = read(d, use_info=False, use_cache=False, disp=False,
                        progress=_progress)
            if len(dcms) > 0:
                _store_cache(dcms, cache_filename)
                pm.set_message('cache written to %s' % cache_filename)


def _column_name(col):
    if is_sequence(col):
        key, val = col[0], col[1]
        return key
    return col

def _column_value(col, series):
    if is_sequence(col):
        key, val = col[0], col[1]
        if callable(val):
            return val(series)
        else:
            return series.first.get(val)
    return series.first.get(col)

def disp(dicomset, extra_columns=tuple()):
    '''Display an iterable of dicoms, removing redundant information
    Args:
    dicomset      -- dicomset
    extra_columns -- columns to display

    Returns: Prints a summary of the dicom data
    '''
    columns = [('#', 'SeriesNumber'), ('Description', 'SeriesDescription')] \
            + list(extra_columns) \
            + [('Count', lambda s: s.count),
               ('ROI', lambda s: '*' if hasattr(s.first, 'meta') and s.first.meta.get('roi') else '')]
    
    if dicomset.count > 0:
        for study in dicomset.studies():
            st = study.first
            print('Patient: %r' % st.PatientName)
            print('StudyID: %r' % st.StudyID)
            print('StudyInstanceUID: %r' % st.StudyInstanceUID)
            if hasattr(st, 'meta'):
                print('Meta: %r' % st.meta.dict())
           
            t = PrettyTable([_column_name(col) for col in columns])
            t.align = 'l'
            for series in study.series():
                t.add_row([_column_value(col, series) for col in columns])
            print('%s\n' % t)
    else:
        print('Dicom list is empty')


def view(dicoms, groupby=tuple(), roi_filename=None, roi_tag='/'):
    '''Display a dicomset with arrview
    Args:
    dicoms  -- An iterable of dicoms
    groupby -- Before displaying, group the dicoms (see data function)
    roi_filename -- Filename of rois to load. If not specified and the dicomset
                    has rois, then those are loaded instead.
    roi_tag -- (default: '/') When using the rois from the dicomset, 
               this is the roi tag to load.

    Returns:
    Displays the dicoms using arrview and returns the instance once
    the window closes
    '''
    import arrview
    arr = data(dicoms, field='pixel_array', groupby=groupby)
    df = dicoms.first
    if roi_filename is None and df.meta.get('roi_filename'):
        roi_filename = df.meta.roi_filename.get(roi_tag)
    else:
        roi_filename = os.path.dirname(df.filename)
    return arrview.view(arr, roi_filename=roi_filename)


def dcm_copy(dicoms, dest):
    '''Copy the dicoms from their current directory to the dest directory.
    File basenames are unchanged.
    Args:
    dicoms -- An interable of dicoms
    dest   -- Path to copy dicoms to

    Returns: Nothing
    '''
    assert os.path.isdir(dest), 'dest must be a directory that exists'
    for dcm in dicoms:
        filename = dcm.filename
        base = os.path.basename(filename)
        shutil.copy(filename, os.path.join(dest, base))
