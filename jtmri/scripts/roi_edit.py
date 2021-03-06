import re

from terseparse import Arg, Parser, types


description = """\
roi-edit: Edit a list of ROI files

Usage:
# Change the 2nd dimension to 0 and 3rd dimension to 5 and extend the dimensions to 10
> roi-edit --dim=2 --dim=3,5 --ndim=10 rois.h5

# Rename rois named kidney to liver and lung to heart: --rename=pattern,replacement
> roi-edit --rename=kidney,liver --rename=lung,heart rois.h5

# Rename can also take regex matches: \1, \2, ... match groups from the pattern
> roi-edit --rename=left_(\w*),right_\1

# To match names with commas in them, quote the comma
> roi-edit --rename=name\,side,newname\,side
"""

class DimEditType(types.Type):
    name = 'dimension-edit'

    def convert(self, val):
        splits = map(int, val.split(','))
        if len(splits) > 0:
            dim = splits[0]
            val = 0
        if len(splits) == 2:
            val = splits[1]
        if len(splits) > 2:
            raise TypeError('A dim edit should be of the form "dim" or "dim,val"')
        return dim, val


class RenameType(types.Type):
    name = 'rename'

    def convert(self, val):
        m = re.match(r'(.*[^\\]),(.*)', val)
        if m is None:
            raise TypeError('A rename edit should be of the form "old_name,new_name"')
        pattern, replacement = m.groups()
        return '^{}$'.format(pattern), replacement


class Renamer(object):
    def __init__(self, rename_maps):
        self._maps = [(re.compile(pattern), replacement) for pattern, replacement in rename_maps]

    def rename(self, name):
        """If a pattern from rename_maps matches name, then it will be used to substitute text.
        If no patterns match then the original name is returned
        """
        for pattern, replacement in self._maps:
            m = re.match(pattern, name)
            if m:
                name = re.sub(pattern, replacement, name)
                break
        return name


def fmt_slc(slc):
    return ','.join(map(str, slc))


p = Parser('roi-edit', description,
    Arg('--dim', 'dimension to flatten', DimEditType(), default=[], action='append'),
    Arg('--ndim', 'final number of dimensions', types.Int.positive, default=-1),
    Arg('--pretend', 'do not commit any actions to disk, just pretend', action='store_true'),
    Arg('--rename', 'old,new name mapping, multiple --rename args can be specified', RenameType(), default=[], action='append'),
    Arg('files', 'roi files to flatten', nargs='+'))


def main():
    _, args = p.parse_args()

    if args.ns.pretend:
        print('Pretending, file saving is disabled')

    renamer = Renamer(args.ns.rename)

    import jtmri.roi

    for f in args.ns.files:
        print('Editing rois in %s' % f)
        rois = jtmri.roi.load(f)

        edited = False
        for roi in rois:
            slc = list(roi.slc)
            # Edit all dimensions from cli args
            for dim, val in args.ns.dim:
                # Cannot edit a view dimension
                if dim in roi.slc.viewdims:
                    raise Exception("cannot edit dimension {} because it is a view dimension, file:{}, roi:{}"\
                            .format(dim, f, roi))
                if dim >= len(slc):
                    raise Exception("dimension {} is too big for roi with only {} dimensions, file:{}, roi:{}"\
                            .format(dim, len(slc), f, roi))
                slc[dim] = val
            # Add or subtract dimensions from end of slice
            if args.ns.ndim > -1:
                for d in roi.slc.viewdims:
                    if d >= args.ns.ndim:
                        raise Exception("reducing dimesions from {} to {} destroys a view dimension ({}, {}), file:{}, roi:{}"\
                                .format(len(slc), args.ns.ndim, d, slc[d], f, roi))
                if args.ns.ndim > len(slc):
                    slc = slc + [0] * (args.ns.ndim - len(slc))
                else:
                    slc = slc[:args.ns.ndim]

            slc = jtmri.roi.SliceTuple(slc)
            name = renamer.rename(roi.name)

            edits = []
            if roi.slc != slc:
                edits.append('slice: {} -> {}'.format(fmt_slc(roi.slc), fmt_slc(slc)))
                edited = True
            else:
                edits.append('slice: {}'.format(fmt_slc(slc)))

            if roi.name != name:
                edits.append('name: {} -> {}'.format(roi.name, name))
                edited = True
            else:
                edits.append('name: {}'.format(roi.name))
            print('  '.join(edits))

            roi.slc = slc
            roi.name = name

        if not args.ns.pretend and edited:
            jtmri.roi.store(rois, f)
            print('Saving rois to %s' % f)
        else:
            print('ROIs unchanged, skipping file %s' % f)
        print
