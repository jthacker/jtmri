import numpy as np
import pylab as plt
import sys
import logging
import csv
import re
import itertools

log = logging.getLogger('jtmri.utils')

class ProgressMeter(object):
    '''Displays a CLI progress meter.
    maxVal is the final value of the parameter being incremented.
    msg will be displayed while the meter is progressing.'''

    def __init__(self, maxVal, msg):
        self._progress = 0
        self._max = float(maxVal)
        self._msg = msg

    def _display(self):
        i = 100 * (self._progress / self._max)
        sys.stdout.write("\r%s - %d%%" % (self._msg, i))
        sys.stdout.flush()

    def _end(self, msg):
        '''Call this if the task finishes earlier than expected'''
        self._display()
        sys.stdout.write(" -- %s\n" % msg)
        sys.stdout.flush()

    def increment(self):
        '''Increment the internal value and update the display'''
        assert self._progress < self._max, "The progress (%d) should not be bigger than the max value (%d), the class was initialized incorrectly." % (self._progress, self._max)
        self._progress += 1
        self._display() 

    def finishSuccess(self):
        '''Call this if the task finishes succssefully earlier than expected'''
        self._progress = self._max
        self._end("Finished!")

    def error(self, msg="Error!"):
        '''Call this if the task errors out'''
        self._end(msg) 


class DataObj(object):
    def __init__(self, header, data):
        '''data is assumed to be a numpy array'''
        self.header = header
        self.data = data
        self._headerToColumnDict = dict(enumerate(header))

    def columnNames(self):
        return self.header

    def column(self, name):
        '''Select a column of data by its header name.
        If the column does not exist than the empty list is returned'''
        return self.data[:, self._headerToColumnDict[name]]

    def __repr__(self):
        return "DataObj(header=%s, data=%s)" % (self.header, self.data)


def readData(filename, delimiter='\t'):
    '''Reads a Tab-Separated-File (can switch to other delimiters with the keyword arg)
    Assumes first line is the header and separates it from the rest of the data
    Converts the remaining data into a numpy floating point array'''
    rawData = list(csv.reader(open(filename), delimiter=delimiter))
    header = rawData[0]
    data = np.array(rawData[1:], dtype=np.float)
    return DataObj(header=header, data=data)


def _decompressValues(vals):
    i = 0
    N = len(vals)
    while i < N:
        if N - i >= 3:
            v,vn,vnn = vals[i:i+3]
            if vn == v:
                yield v
                yield vn 
                for _ in xrange(vnn):
                    yield v
                i += 3
            else:
                yield vals[i]
                i += 1
        else:
            yield vals[i]
            i += 1


def readDSV(filename):
    '''Read a Siemens DSP-simulator DSV file. 
    Decompresses the data in the [VALUES] section then return it as a dictionary'''
    sectionRegex = re.compile(r'\[(.*)\]')
    keyValPairRegex = re.compile(r'(.*)=(.*)')

    dsv = {}

    with open(filename) as f:
        # 1: Read and parse the configuration until the VALUES section is reached
        section = None
        for line in f:
            if line[0] == ';': # Ignore lines starting with a comma
                continue

            sectionMatch = sectionRegex.match(line)

            # If the line represents a new section, then update the section
            if sectionMatch:
                section = sectionMatch.groups()[0].lower()
                log.debug("Parsing [%s] Section" % section.upper())
                dsv[section] = {}
                if section == 'values':
                    break
                else:
                    continue

            keyValMatch = keyValPairRegex.match(line)
            if keyValMatch:
                key,val = keyValMatch.groups()
                try:
                    val = float(val)
                except ValueError:
                    val = val.strip()
                dsv[section][key.lower()] = val

        assert 'definitions' in dsv.keys(), 'No [DEFINITIONS] section was found'
        definitions = dsv['definitions']
        
        assert section == 'values', "No [VALUES] section was found. Last section seen %s. Configuration so far %s" % (section, dsv)

        # 2: Read all the data from the values section
        rawValues = []
        for line in f:
            if line[0] == ';': # Ignore lines starting with a comma
                continue
            rawValues.append(line)
        # Remove the last line since it is a blank line
        rawValues = np.array(rawValues[:-1], dtype=int)
        rawValuesLen = rawValues.size
        
        # 3: Decompress and scale the values.
        # Decompress. 0 0 3 -> 0 0 0 0 0
        # Convert from deltas to real values
        # Scalesby the vertfactor
        values = np.array(list(_decompressValues(rawValues))).cumsum() / definitions['vertfactor']
        
        log.debug("Found %d raw values and %d decompressed values." % (rawValuesLen, values.size))

        expectedSamples = definitions['samples']
        assert values.size == expectedSamples, "The number of decompressed values, %d, should be the same as the number of data points defined in the header, %d." % (values.size, expectedSamples)
        
        dsv['values'] = values
        return dsv 

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

class AttributeDict(object):
    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
    
    def __dir__(self):
        return sorted(set(dir(type(self)) + self._dict.keys()))

    def __getitem__(self, key):
        return self._dict[key]

    def __getattr__(self, key):
        return self._dict[key]

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def __repr__(self):
        return 'AttributeDict(' + self._dict.__repr__() + ')'
        


