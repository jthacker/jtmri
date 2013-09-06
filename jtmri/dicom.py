import struct
import logging
import re
import collections
from pprint import pprint
from glob import glob
import os
import dicom
import pandas as pd

log = logging.getLogger('jtmri:dicom')

_default_headers = ('PatientName', 'SeriesNumber', 'SeriesDescription')

def isdicom(path):
    isdcm = False
    if os.path.isfile(path):
        with open(path,'r') as fp:
            fp.seek(0x80)
            magic = fp.read(4)
            isdcm = (magic == b"DICM")
    return isdcm

def read(path, extraHeaders=tuple()):
    if os.path.isdir(path):
        path = os.path.join(path, '*') 
    paths = glob(path)

    headers = _default_headers + extraHeaders
    dcms = [dicom.read_file(path) for path in paths if isdicom(path)]
    data = [[dcm.get(h) for h in headers] for dcm in dcms]
    df = pd.DataFrame.from_records(data, columns=headers)
    df.sort(columns=headers, inplace=True)
    return df

def ls(path=None, extraHeaders=tuple()):
    '''Given glob like path, list the dicoms in a given directory, or
    the current one is no arguments are given'''
    path = path if path else os.path.abspath(os.path.curdir)

    df = read(path, extraHeaders)
    df.drop_duplicates(inplace=True)
    print(df.to_string(index=False))

def _null_truncate(s):
    """Given a string, returns a version truncated at the first '\x00' if
    there is one. If not, the original string is returned.
    
    [Taken from VESPA project http://scion.duhs.duke.edu/vespa]
    """
    i = s.find(chr(0))
    if i != -1:
        s = s[:i]
    return s


def _scrub(item):
    """Given a string, returns a version truncated at the first '\x00' and
    stripped of leading/trailing whitespace. If the param is not a string,
    it is returned unchanged.
    
    [Taken from VESPA project http://scion.duhs.duke.edu/vespa]
    """
    if isinstance(item, basestring):
        return _null_truncate(item).strip()
    else:
        return item


def _get_chunks(tag, index, format, little_endian = True):
    """Given a CSA tag string, an index into that string, and a format
    specifier compatible with Python's struct module, returns a tuple
    of (size, chunks) where size is the number of bytes read and
    chunks are the data items returned by struct.unpack(). Strings in the
    list of chunks have been run through _scrub().
    
    [Taken from VESPA project http://scion.duhs.duke.edu/vespa]
    """
    format = ('<' if little_endian else '>') + format
    size = struct.calcsize(format)
    chunks = struct.unpack(format, tag[index:index + size])
    chunks = [ _scrub(item) for item in chunks ]
    return (size, chunks)


def _parse_csa_header(tag, little_endian = True):
    """The CSA header is a Siemens private tag that should be passed as
    a string. Any of the following tags should work: (0x0029, 0x1010),
    (0x0029, 0x1210), (0x0029, 0x1110), (0x0029, 0x1020), (0x0029, 0x1220),
    (0x0029, 0x1120).
    
    The function returns a dictionary keyed by element name.
    
    [Taken from VESPA project http://scion.duhs.duke.edu/vespa]
    """
    DELIMITERS = ('M', '\xcd', 77, 205)
    elements = {}
    current = 0
    size, chunks = _get_chunks(tag, current, '4s4s')
    current += size
    assert chunks[0] == 'SV10'
    assert chunks[1] == '\x04\x03\x02\x01'
    size, chunks = _get_chunks(tag, current, 'L')
    current += size
    element_count = chunks[0]
    size, chunks = _get_chunks(tag, current, '4s')
    current += size
    assert chunks[0] in DELIMITERS
    for i in range(element_count):
        size, chunks = _get_chunks(tag, current, '64s4s4s4sL4s')
        current += size
        name, vm, vr, syngo_dt, subelement_count, delimiter = chunks
        assert delimiter in DELIMITERS
        values = []
        for j in range(subelement_count):
            size, chunks = _get_chunks(tag, current, '4L')
            current += size
            assert chunks[0] == chunks[1]
            assert chunks[1] == chunks[3]
            assert chunks[2] in DELIMITERS
            length = chunks[0]
            size, chunks = _get_chunks(tag, current, '%ds' % length)
            current += size
            if chunks[0]:
                values.append(chunks[0])
            current += (4 - length % 4) % 4

        if len(values) == 0:
            values = ''
        if len(values) == 1:
            values = values[0]
        assert name not in elements
        elements[name] = values

    return elements


class SiemensProtocol(object):
    @staticmethod
    def fromDicom(dicom):
        seriesData = _parse_csa_header(dicom[(0x0029, 0x1020)])
        protocolRaw = seriesData['MrPhoenixProtocol']
        protocol = SiemensProtocol(protocolRaw).asDict()
        seriesData['MrPhoenixProtocol'] = protocol
        return seriesData

    def __init__(self, protocolStr):
        startToken = '### ASCCONV BEGIN ###'
        endToken = '### ASCCONV END ###'
        start = protocolStr.find(startToken) + len(startToken) 
        end = protocolStr.find(endToken) - 1
        self._rawProtocol = protocolStr[start:end].split('\n')
        if self._rawProtocol[0] == '':
            self._rawProtocol = self._rawProtocol[1:]

    def asDict(self):
        return self._parse()

    def _splitLines(self):
        for line in self._rawProtocol:
            splitLine = line.split('=')
            if len(splitLine) == 2:
                yield splitLine
            else:
                log.warn('Line ['+repr(line)+'] is missing the split token (=)')


    def _parse(self):
        arrayPattern = re.compile(r'(\w+)\[(\d+)\]')
        proto = {}
        
        def nameGen(key):
            for name in key.strip().split('.'):
                match = arrayPattern.match(name)
                if match:
                    prefix,idx = match.groups()
                    yield prefix
                    yield idx
                else:
                    yield name

        for key,val in self._splitLines():
            val = val.strip()
            d = d_prev = proto
            for name in nameGen(key):
                if name not in d:
                    d[name] = {}
                d_prev = d
                d = d[name]
            d_prev[name] = val
        return proto

