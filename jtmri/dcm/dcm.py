from __future__ import absolute_import
from glob import iglob
import os, dicom, logging
import numpy as np

from itertools import ifilter
from collections import defaultdict, Iterable
from prettytable import PrettyTable
from tqdm import tqdm

from .siemens import SiemensProtocol
from .info import load_infos
from ..utils import unique, AttributeDict, ListAttrAccessor


log = logging.getLogger('jtmri:dicom')


class DicomParser(object):
    @staticmethod
    def to_attributedict(dcm):
        return DicomParser.to_dict(dcm, AttributeDict)

    @staticmethod
    def to_dict(dcm, wrap=dict):
        d = DicomParser._dicom_to_dict(dcm, wrap)
        d['Siemens'] = wrap(SiemensProtocol.from_dicom(dcm))
        d['pixel_array'] = dcm.pixel_array
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

    @property
    def first(self):
        return self[0]

    @property
    def all(self):
        '''Access an attribute from all objects in this set'''
        return ListAttrAccessor(self) 

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
        return self.filter(lambda d: d.SeriesNumber in nums)

    def series(self):
        for num in unique(self.all.SeriesNumber):
            yield self.by_series(num)

    def disp(self):
        disp(self)

    def data(self, groupby=tuple()):
        return data(self, field='pixel_array', groupby=groupby)

    def view(self, groupby=tuple()):
        return view(self, groupby=groupby)

    @property
    def count(self):
        return len(self)

    def append(self, dcm):
        self._dcms.append(dcm)

    def __iter__(self):
        return iter(self._dcms)

    def __getitem__(self, key):
        return self._dcms[key]

    def __len__(self):
        return len(self._dcms)


def isdicom(path):
    '''Check if path is a dicom file'''
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
    groupby = tuple(groupby)
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
    if os.path.isdir(path):
        path = os.path.join(path, '*') 

    for path in iglob(path):
        if recursive and os.path.isdir(path):
            for root,_,files in os.walk(path):
                for f in files:
                    yield os.path.join(root,f)
        else:
            yield path


def read(path=None, disp=True, recursive=False):
    '''Read dicom files from path and print a summary
    Args:
    path      -- glob like path of dicom files, if None then the current dir is used
    info_path -- path to read info.yaml file from
    disp      -- (default: True) Print a summary
    recursive -- (default: False) Recurse into subdirectories

    Returns:
    A list of dicom objects
    Prints a summary of the dicom objects
    '''
    path = path if path else os.path.abspath(os.path.curdir)

    dcms = []
    for p in tqdm(_path_gen(path, recursive)):
        if isdicom(p):
            dcm = DicomParser.to_attributedict(dicom.read_file(p))
            dcm['filename'] = p
            dcms.append(dcm)

    dcms = load_infos(dcms)

    key = lambda d: (d.StudyInstanceUID, d.SeriesNumber, d.InstanceNumber)
    dicomSet = DicomSet(sorted(dcms, key=key))
    if disp:
        dicomSet.disp()
    return dicomSet


def disp(dicoms, headers=tuple()):
    '''Display an iterable of dicoms, removing redundant information
    Args:
    dicoms  -- iterable of dicoms
    headers -- additional headers to display data for

    Returns:
    Returns nothing.
    Prints a summary of the dicom data
    '''
    _headers = ('SeriesNumber', 'SeriesDescription','RepetitionTime') + headers

    if len(dicoms) > 0:
        groups = ('PatientName', 'StudyInstanceUID', 'SeriesNumber')
        for patientName,studies in groupby(dicoms, groups):
            for studyID,series in studies:
                t = PrettyTable(_headers + ('Count',))
                t.align = 'l'

                print('Patient: %s' % patientName)
                print('Study: %s' % studyID)
                for seriesNum,dcms in series:
                    row = [dcms[0].get(h) for h in _headers] + [len(dcms)]
                    t.add_row(row)
                print('%s\n' % t)
    else:
        print('Dicom list is empty')


def view(dicoms, groupby=tuple()):
    '''Display a dicomset with arrview
    Args:
    dicoms  -- An iterable of dicoms
    groupby -- Before displaying, group the dicoms (see data function)

    Returns:
    Displays the dicoms using arrview and returns the instance once
    the window closes
    '''
    import arrview
    arr = data(dicoms, field='pixel_array', groupby=groupby)
    return arrview.view(arr)
