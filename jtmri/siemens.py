import struct
import re
from collections import namedtuple

_mdhFields = (
    ('FlagsAndDMALength'    ,0,   'L'),
    ('MeasUID'              ,4,   'l'),
    ('ScanCounter'          ,8,   'L'),  
    ('TimeStamp'            ,12,  'L'),
    ('PMUTimeStamp'         ,16,  'L'),
    ('EvalInfoMask'         ,20,  'Q'),
    ('SamplesInScan'        ,28,  'H'),
    ('UsedChannels'         ,30,  'H'),
    ('Line'                 ,32,  'H'),  
    ('Acquisition'          ,34,  'H'),
    ('Slice'                ,36,  'H'),
    ('Partition'            ,38,  'H'),
    ('Echo'                 ,40,  'H'),  
    ('Phase'                ,42,  'H'),
    ('Repetition'           ,44,  'H'),  
    ('Set'                  ,46,  'H'),
    ('Seg'                  ,48,  'H'),  
    ('Ida'                  ,50,  'H'),
    ('Idb'                  ,52,  'H'),
    ('Idc'                  ,54,  'H'),
    ('Idd'                  ,56,  'H'), 
    ('Ide'                  ,58,  'H'),
    ('PreCutOff'            ,60,  'H'),  
    ('PostCutOff'           ,62,  'H'),
    ('KSpaceCentreColumn'   ,64,  'H'),  
    ('CoilSelect'           ,66,  'H'),
    ('MysteryValue'         ,68,  'H'),
    ('ReadOutOffcentre'     ,70,  'f'),  
    ('TimeSinceLastRF'      ,74,  'H'),
    ('KSpaceCentreLineNo'   ,76,  'H'),  
    ('KSpaceCentrePartitionNo' ,78, 'H'),
    ('IceProgramPara1'      ,80,  'H'),  
    ('IceProgramPara2'      ,82,  'H'),
    ('IceProgramPara3'      ,84,  'H'),  
    ('IceProgramPara4'      ,86,  'H'),
    ('FreePara1'            ,88,  'H'),
    ('FreePara2'            ,90,  'H'),
    ('FreePara3'            ,92,  'H'),
    ('FreePara4'            ,94,  'H'),
    ('SlcPosSag'            ,96,  'f'),
    ('SlcPosCor'            ,100, 'f'),
    ('SlcPosTra'            ,104, 'f'), 
    ('Quaternion1'          ,108, 'f'),
    ('Quaternion2'          ,112, 'f'), 
    ('Quaternion3'          ,116, 'f'),
    ('Quaternion4'          ,120, 'f'),  
    ('ChannelId'            ,124, 'H'),
    ('PTABPosNeg'           ,126, 'H'))

structFmt = '<' + ''.join([width for (_,_,width) in _mdhFields])
MDH_SIZE = 128
MDH = namedtuple('MDH', [name for (name,_,_) in _mdhFields])
Raw = namedtuple('Raw', ('header', 'mdh', 'data'))

assert struct.calcsize(structFmt) == MDH_SIZE, 'MDH must be %d bytes wide' % MDH_SIZE

def parse_mdh(mdhData):
    parsedData = struct.unpack(structFmt, mdhData)
    return MDH(*parsedData)

def parse_header(hdrData):
    pass

def unpack(fmt, f):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(size))

def read_raw_data(fileName):
    with open(fileName, 'rb') as f:
        hdrSize = unpack('<I', f)[0]
        hdr = f.read(hdrSize - struct.calcsize('<I'))
        mdh = parse_mdh(f.read(MDH_SIZE))
        return mdh


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
    _dicom_tag = (0x0029, 0x1020)

    @staticmethod
    def has_protocol(dcm):
        '''Check if a dicom file has a Siemens Protocol embedded in it'''
        return SiemensProtocol._dicom_tag in dcm

    @staticmethod
    def from_dicom(dcm):
        seriesData = {}
        if SiemensProtocol.has_protocol(dcm):
            seriesData = _parse_csa_header(dcm[SiemensProtocol._dicom_tag])
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

