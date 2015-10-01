#!/usr/bin/env python
import argparse
import os
import shutil
import sys
import jtmri.dcm

def filter_expr(expr_str):
    if expr_str is None:
        return lambda dcm:dcm
    else:
        return eval('lambda dcm: %s' % expr_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename a set of '
            'dicom files')
    parser.add_argument('--ext', default='ima',
            help='Set the extension to use for the renamed files')
    parser.add_argument('--filter', default=None, type=filter_expr,
            help='Filter the dicoms to rename, eg. dcm.SeriesNumber != 2. '
                 'Expression can be any valid python code that works in a lambda expression')
    parser.add_argument('--move', action='store_true',
            help='Move files instead of copying')
    parser.add_argument('--outputdir', help='Directory to output renamed files to',
            metavar='DIR')
    parser.add_argument('--pretend', action='store_true',
            help='Show the renaming that will take place without actually performing it')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    if not args.pretend and args.outputdir and not os.path.exists(args.outputdir):
        os.mkdir(args.outputdir)

    count = 0
    for f in args.files:
        dcms = jtmri.dcm.read(f, disp=False)
        for dcm in dcms.filter(args.filter):
            filename = jtmri.dcm.dcm.canonical_filename(dcm)
            dirname = args.outputdir if args.outputdir else os.path.basename(dcm.filename)
            path = os.path.join(dirname, filename)
            print('%s -> %s' % (os.path.relpath(dcm.filename), os.path.relpath(path)))
            if not args.pretend:
                if os.path.exists(path):
                    print('Warning: Path already exists %s' % path)
                if args.move:
                    shutil.move(dcm.filename, path)
                else:
                    shutil.copy(dcm.filename, path)
            count += 1
    print('Renamed %d files' % count)
