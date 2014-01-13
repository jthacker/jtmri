from __future__ import absolute_import
from glob import glob
import os, dicom, logging
import numpy as np

from itertools import ifilter
from collections import defaultdict, Iterable
from prettytable import PrettyTable

from .siemens import SiemensProtocol
from .utils import unique, AttributeDict, ProgressMeter

try:
    import arrview
except:
    print('Missing arrview module. view methods will not work without it.')

log = logging.getLogger('jtmri:dicom')

class DicomParser(object):
    @staticmethod
    def to_dict(dcm):
        d = DicomParser._dicom_to_dict(dcm)
        d['Siemens'] = SiemensProtocol.fromDicom(dcm)
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
    def _dicom_to_dict(dcm):
        d = AttributeDict()
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
                        val = [DicomParser._dicom_to_dict(ds) for ds in val]
                    else:
                        val = DicomParser._convert(val)
                    d[name] = val
        return d


class DicomSet(object):
    def __init__(self, dcms):
        self._dcms = list(dcms)

    def series(self, *nums):
        return DicomSet(ifilter(lambda d: d.SeriesNumber in nums, self))

    def disp(self):
        disp(self)

    def data(self):
        return data(self)

    def view(self):
        return view(self)

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


def groupby(dicoms, key=lambda x: x, sort=True):
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

    d = defaultdict(list)
    for dcm in dicoms:
        d[keyfunc(dcm)].append(dcm)
    
    items = []
    for k,v in sorted(d.items(), key=lambda x: x[0]):
        if extrakeys:
            items.append((k, groupby(v, extrakeys)))
        else:
            items.append((k, v))
    return tuple(items)


def data(dicoms):
    '''Get the pixel array from each dicom and concatentate them together
    Args:
    dicoms -- an iterable of dicoms

    Returns:
    numpy array of all the pixel arrays concatenated along the third dimension.
    If they are all the same shape then a simple array is returned,
    if they are not then a record array is returned.
    '''
    arr = []
    for loc,ds in groupby(dicoms, ('SliceLocation','InstanceNumber')):
        inner = []
        for num,dcms in ds:
            assert len(dcms) == 1
            inner.append(dcms[0].pixel_array)
        arr.append(inner)
    return np.array(arr).transpose(2,3,0,1)


def read(path=None, progressfn=None):
    '''Read dicom files from the path
    Args:
    path -- glob style path of dicom files, if a dir then all files in dir are added

    Returns:
    A list of dicom objects
    '''
    path = path if path else os.path.abspath(os.path.curdir)

    if os.path.isdir(path):
        path = os.path.join(path, '*') 
    paths = glob(path)

    def dicom_gen():
        dcmpaths = filter(isdicom, paths)
        total = len(dcmpaths)
        for i,dcmpath in enumerate(dcmpaths, 1):
            if progressfn:
                progressfn(i, total)
            yield DicomParser.to_dict(dicom.read_file(dcmpath))
    return DicomSet(dicom_gen())


def ls(path=None, headers=tuple()):
    '''Read dicom files from path and print a summary
    Args:
    path    -- glob like path of dicom files, if None then the current dir is used
    headers -- headers to use in summary in addition to the default ones

    Returns:
    A list of dicom objects
    Prints a summary of the dicom objects
    '''
    progress = ProgressMeter(1000, 'reading dicoms')
    dicomSet = read(path, progressfn=lambda i,n: progress.setprogress(i/float(n)))
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
        # Groupby Patient, StudyInstanceUID, SeriesInstanceUID
        groups = ('PatientName', 'StudyID', 'SeriesNumber')
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


def view(dicoms):
    arr = data(dicoms)
    return arrview.view(arr)
