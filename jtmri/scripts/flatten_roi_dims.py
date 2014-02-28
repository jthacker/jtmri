#!/usr/bin/env python
import argparse
import jtmri.roi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take a set of files '\
            'and flatten them by setting the slc dimension to 0 for '\
            'the specified specified and each ROI in the file.')
    parser.add_argument('--dim', default=-1, type=int, help='dimension to flatten')
    parser.add_argument('--size', default=0, type=int, help='final size of slice')
    parser.add_argument('--pretend', action='store_true')
    parser.add_argument('files', nargs='+', help='roi files to flatten')
    args = parser.parse_args()

    if args.pretend:
        print('Pretending, file saving disabled')

    for f in args.files:
        print('Flattening rois in %s' % f)
        rois = jtmri.roi.load(f)
        for roi in rois:
            slc = list(roi.slc)
            slc[args.dim] = 0
            slc = slc + [0] * max(args.size - len(slc), 0)
            slc = jtmri.roi.SliceTuple(slc)
            print('%s -> %s %s' % (roi.slc, slc, roi.name))
            roi.slc = slc
        if not args.pretend:
            jtmri.roi.save(rois, f)
            print('Saving rois to %s' % f)
