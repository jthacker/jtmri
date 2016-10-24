import datetime
import dicom
import numpy as np
import os
import scipy.io
import time
import jtmri.dcm
from jtmri.utils import filter_error
from terseparse import Parser, Arg, types

description = '''Converts an rbf.mat file to a dicom file
Dicom parameters are read from the specified dicom file, which is assumed to
have only one study in it.
The new ASL dicom is added to the end of the study.
'''

class RBFMatType(types.File):
    def __init__(self, mode):
        super(RBFMatType, self).__init__(mode)
        self.name = 'rbf-mat'

    def convert(self, val):
        f = super(RBFMatType, self).convert(val)
        if f is None:
            return
        rbf_mat = scipy.io.loadmat(f.name)
        try:
            return rbf_mat['rbf']
        except KeyError:
            self.fail(val, "Wrong format, 'rbf' key is missing")


def pick(dcms):
    dcms.disp()
    series_nums = dcms.all_unique.SeriesNumber
    while True:
        series_num = raw_input('pick a series: ')
        if series_num == 'q':
            return None
        try:
            series_num = int(series_num)
            if series_num in series_nums:
                return dcms.by_series(series_num)
        except ValueError:
            pass
        print("%s is an invalid series number, choose one from the preceeding list or 'q' to quit" % series_num)
        return None


def get_asl_series(series_num, dcms):
    if series_num == 'smart-pick':
        return pick(dcms.filter(filter_error(lambda d: d.SequenceName.lower().startswith('tfi2d1'), catch=AttributeError)))
    elif series_num == 'pick':
        return pick(dcms)
    return dcms.by_series(series_num)


def get_output_asl_series(series_num, dcm, study_dcms, overwrite):
    series_numbers = set(study_dcms.all.SeriesNumber)
    if series_num == 'auto':
        series_num = study_dcms.all.SeriesNumber.max() + 1
    elif series_num == 'add-100':
        series_num = dcm.SeriesNumber + 100
    elif series_num == 'pick':
        series_num = dcm.SeriesNumber + 100
        if series_num in series_numbers:
            series_num = max(series_numbers) + 1
        while True:
            inpt = raw_input('pick a series number (default: %d): ' % series_num)
            if inpt != '':
                try:
                    series_num = int(inpt)
                except ValueError:
                    print("please enter a whole number")
                    continue
            if series_num in series_numbers and not overwrite:
                print("series number %d is already taken, choose a different one" % series_num)
            else:
                break
    else:
        raise Exception('unable to parse series number %s' % series_num)
    if series_num in series_numbers and not overwrite:
        raise Exception('output series number ({}) already exists in study. Use --overwrite to override'.format(series_num))
    return series_num


def inc_uid(uid):
    """Increment the UID"""
    ints = map(int, uid.split('.'))
    ints[-1] += 1
    return '.'.join(map(str, ints))


series_num_type = types.Int.positive | 'smart-pick' | 'pick'
output_series_num_type = types.Int.positive | 'auto' | 'add-100' | 'pick'


p = Parser('asl-rbf-to-dicom', description,
    Arg('--disable-cache-update', 'Disable updating of the dicom cache', action='store_true'),
    Arg('--verbose', 'Enable verbose printing', action='store_true'),
    Arg('--asl-series', 'ASL series number', series_num_type, default='smart-pick'),
    Arg('--output-series', 'Series number to save output as', output_series_num_type, default='pick'),
    Arg('--pretend', 'Disable writing', action='store_true'),
    Arg('--overwrite', 'Overwrite an existing file', action='store_true'),
    Arg('rbf', 'ASL rbf.mat file', RBFMatType.r),
    Arg('dicom-dir', 'A directory of dicom files from the series that the rbf.mat was derived from', types.Dir.rw))


def main():
    parser, args = p.parse_args()
    dcms = jtmri.dcm.read(args.ns.dicom_dir, disp=False)
    last_dcm = dcms[-1]
    if len(dcms) == 0:
        parser.error('No dicoms found in %s' % args.ns.dicom_dir)

    asl_series = get_asl_series(args.ns.asl_series, dcms)
    if asl_series is None:
        parser.error('Failed to automatically find the ASL series number')

    # Update the dicom parameters for the new ASL recon file
    dcm = dicom.read_file(asl_series.first.filename)
    dcm.ContentDate = str(datetime.date.today()).replace('-','')
    dcm.ContentTime = str(time.time()) #milliseconds since the epoch
    dcm.SeriesNumber = get_output_asl_series(args.ns.output_series, dcm, dcms, args.ns.overwrite)
    dcm.SOPInstanceUID = inc_uid(last_dcm.SOPInstanceUID)
    dcm.SeriesInstanceUID = inc_uid(last_dcm.SeriesInstanceUID)
    dcm.SeriesDescription = 'ASL recon: %s' % dcm.get('SeriesDescription', 'NONE')

    pixel_array = args.ns.rbf.astype(float)
    rows, cols = pixel_array.shape
    dcm.Rows = rows
    dcm.Columns = cols
    dcm.SamplesPerPixel = 1
    dcm.PixelRepresentation = 0  # Unsigned
    dcm.HighBit = 31
    dcm.BitsStored = 32
    dcm.BitsAllocated = 32
    dcm.SmallestImagePixelValue = 0
    dcm.LargesImagePixelValue = 0xFFFFFFFF
    fmin, fmax = pixel_array.min(), pixel_array.max()
    m = float(fmax - fmin) / (2**32 - 1)
    dcm.RescaleSlope = m
    dcm.RescaleIntercept = float(fmin)
    dcm.PixelData = ((pixel_array - fmin) / m).astype(np.uint32).tostring()

    filename = os.path.join(args.ns.dicom_dir, jtmri.dcm.dcm.canonical_filename(dcm))
    print('writing ASL dicom file to %s' % filename)
    if not args.ns.pretend:
        if os.path.exists(filename) and not args.ns.overwrite:
            raise Exception('File ({}) already exists! Use --overwrite to override'.format(filename))
        dcm.save_as(filename)

    if not args.ns.disable_cache_update:
        print('updating cache')
        jtmri.dcm.cache(args.ns.dicom_dir, full=args.ns.overwrite)
