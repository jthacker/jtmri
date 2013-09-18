from __future__ import absolute_import
from glob import glob
import numpy as np
import os
import dicom
import logging
from collections import defaultdict, Iterable
from prettytable import PrettyTable
from .siemens import SiemensProtocol
from .utils import unique

log = logging.getLogger('jtmri:dicom')


class DicomParser(object):
    @staticmethod
    def to_dict(dcm):
        d = _dicomToDict(dcm)
        d['SiemensProtocol'] = SiemensProtocol.fromDicom(dcm)
        return d

    def _convert(self,val):
        if isinstance(val, list):
            val = map(self.convert, val)
        elif isinstance(val, dicom.valuerep.DSfloat):
            val = float(val)
        elif isinstance(val, dicom.valuerep.IS):
            val = int(val)
        elif isinstance(val, dicom.UID.UID):
            val = str(val)
        else:
            val = str(val)
        return val

    def _dicom_to_dict(dcm):
        d = {}
        for tag in dcm.keys():
            elem = dcm[tag]
            key = dicom.datadict.keyword_for_tag(tag)
            val = elem.value
            if tag != (0x7fe0, 0x0010):
                if isinstance(val, dicom.sequence.Sequence):
                    val = [_dicomToDict(ds) for ds in val]
                else:
                    val = convert(val)
                d[key] = val
        return d



def isdicom(path):
    '''Check if path is a dicom file'''
    isdcm = False
    if os.path.isfile(path):
        with open(path,'r') as fp:
            fp.seek(0x80)
            magic = fp.read(4)
            isdcm = (magic == b"DICM")
    return isdcm


def read(path=None):
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

    return [dicom.read_file(path) for path in paths if isdicom(path)]


def data(dicoms):
    '''Get the pixel array from each dicom and concatentate them together

    Args:
        dicoms -- a iterable of dicoms

    Returns:
        numpy array of all the pixel arrays concatenated along the third dimension.
        If they are all the same shape then a simple array is returned,
        if they are not then a record array is returned.
    '''
    arrays = []
    for dcm in dicoms:
        array.append(dcm.pixel_array)
    return np.array(arrays)


def groupby(dicoms, key=lambda x: x, sort=True):
    '''Group dicoms together by key, no presorting necessary

    Args:
        dicoms  -- an iterable of dicoms
        key     -- a callable or an object to use as a key
        sort    -- sort the groups by key

    Returns:
        A list of tuples with the structure [(key0, group0), (key1, group1), ...]

    Examples:
        >>> # Group by the key SeriesNumber
        >>> groupby(dicoms, 'SeriesNumber')
        [(0, [dcm1, dcm2, ...]), (1, [dcm3, dcm4]), (2, [dcm5, dcm6]), ...]

        >>> # Groupy by the key StudyInstanceUID then SeriesInstanceUID
        >>> groupby(dicoms, ('StudyInstanceUID', 'SeriesInstanceUID'))
        [(0, [(0, [dcm1]), (1, [dcm2])]), (1, [(0, [dcm3]), (1, [dcm4])])]

        >>> # Group even and odd SeriesNumbers
        >>> groupby(dicoms, lambda x: x.SeriesNumber % 2 == 0)
        [(True, [dcm1, dcm2, dcm5, dcm6, ...]), (False, [dcm3, dcm4, dcm7, dcm8, ...])]
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


def ls(path=None, headers=tuple()):
    '''Read dicom files from path and print a summary
    
    Args:
        path    -- glob like path of dicom files, if None then the current dir is used
        headers -- headers to use in summary in addition to the default ones

    Returns:
        A list of dicom objects
        Prints a summary of the dicom objects
    '''
    dicoms = read(path)
    disp(dicoms, headers)
    return dicoms


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

    # Groupby Patient, StudyInstanceUID, SeriesInstanceUID
    summary = ''
    for patientName,studies in groupby(dicoms, ('PatientName', 'StudyID', 'SeriesNumber')):
        for studyID,series in studies:
            t = PrettyTable(_headers)
            t.align = 'l'
            rows = tuple(tuple(dcms[0].get(h) for h in _headers) for seriesNum,dcms in series)
            for row in unique(rows):
                t.add_row(row)
            summary += 'Patient: %s\n' % patientName
            summary += 'Study: %s\n' % studyID
            summary += '%s\n\n' % t
    print(summary)
