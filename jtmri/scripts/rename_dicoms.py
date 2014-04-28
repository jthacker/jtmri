import argparse
import os.path
import re
import shutil
import jtmri.dcm

_rgx = re.compile('[^\w\-\_]')

def clean(attr):
    return _rgx.sub('_', attr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename a set of '
            'dicom files')
    parser.add_argument('--outputdir', help='Directory to output renamed files to')
    parser.add_argument('--move', action='store_true',
            help='Move files instead of copying')
    parser.add_argument('--pretend', action='store_true',
            help='Show the renaming that will take place without actually performing it')
    parser.add_argument('--extension', default='ima',
            help='Set the extension to use for the renamed files')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    count = 0
    for f in args.files:
        dcms = jtmri.dcm.read(f, disp=False)
        for dcm in dcms:
            attrs = [dcm.PatientName, dcm.Modality, dcm.StudyDescription, 
                    '%04d' % int(dcm.StudyID), '%04d' % dcm.SeriesNumber, 
                    dcm.SeriesDate, dcm.AcquisitionTime, args.extension]
            filename = '.'.join(clean(attr) for attr in attrs)
            dirname = args.outputdir if args.outputdir else os.path.basename(dcm.filename)
            path = os.path.join(dirname, filename)
            print('%s -> %s' % (os.path.relpath(dcm.filename), os.path.relpath(path)))
            if not args.pretend:
                if args.move:
                    shutil.move(dcm.filename, path)
                else:
                    shutil.copy(dcm.filename, path)
            count += 1
    print('Renamed %d files' % count)
