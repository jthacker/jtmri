# -*- coding: utf-8 -*-
from collections import namedtuple, defaultdict
import logging
import os

from .slice_tuple import SliceTuple
import h5py
import numpy as np
from prettytable import PrettyTable
import skimage.draw

from .utils import unique, rep, path_generator, groupby, AttributeDict

ROI_FILE_EXT = '.h5'


log = logging.getLogger(__name__)


def create_mask(shape, slc, poly, collapse=False):
    """Convert a polygon to a binary mask according the specified array shape
    Args:
        shape    -- a tuple with the size of each dimension in the mask
        slc      -- SliceTuple describing where to apply the polygon to
        poly     -- A numpy array of (x,y) point pairs describing a polygon
        collapse -- Ignore any dimensions > the number of dimensions in shape (i.e. len(shape))

    Returns:
        binary mask with the region in poly set to False and everywhere else
        set to True
    """
    mask = np.zeros(shape, dtype=bool)
    if len(poly) > 0:
        viewShape = shape[slc.ydim],shape[slc.xdim]
        y,x = skimage.draw.polygon(y=poly[:,1], x=poly[:,0], shape=viewShape)
        idxs = slc.slice_from_screen_coords(x, y, mask)
        if collapse:
            idxs = idxs[:mask.ndim]
        mask[idxs] = True
    return mask


class ROI(object):
    def __init__(self, name, mask, props=None):
        self.name = name
        self.mask = mask
        self.props = AttributeDict(props or {})

    @property
    def tag(self):
        """Used by dicom viewer for distinguishing user ROIs
        TODO: Find a better way to handle this use case
        """
        reldir = self.props['reldir']
        return reldir or '/'

    def to_mask(self, shape, collapse=False):
        """Create a mask from an ROI object.
        Args:
            shape    -- shape of output mask
            collapse -- (default:False) collapse the ROI to fit in mask
        """
        return self.mask

    def __repr__(self):
        return '<ROI name:{!r} props:{!r} shape:{!r} ...>'.format(self.name, self.props, self.mask.shape)



class ROISet(object):
    def __init__(self, rois):
        self.rois = rois

    def filter(self, func):
        """Filter ROIs by a unary predicate func
        Args:
            func -- unary predicate
        Return: ROISet containing ROIs that evaluted to true when passed to the predicate
        """
        return ROISet(filter(func, self.rois))

    @property
    def first(self):
        return self.rois[0]

    def by_name(self, *names):
        """Filter ROIs by names"""
        return self.filter(lambda r: r.name in names)

    def by_property(self, name, value):
        """File ROIs by property
        Args:
            name  -- name of property
            value -- value of property
        Return: ROISet of ROIs with matching properties
        """
        return self.filter(lambda r: (name in r.props) and (r.props[name] == value))

    def by_tag(self, tag):
        """Used by dicom viewer for distinguishing user ROIs"""
        return self.filter(lambda r: r.tag == tag)

    def groupby(self, grouper):
        return groupby(self, grouper, group_type=ROISet)

    def to_mask(self, shape, collapse=False):
        '''Create a mask from the ROIs'''
        masks = map(lambda a: a.to_mask(shape, collapse), self)
        return reduce(np.logical_or, masks, np.zeros(shape, dtype=bool))

    def to_masked(self, array, collapse=False):
        '''Returns a numpy masked array'''
        return np.ma.array(array, mask=~self.to_mask(array.shape, collapse))

    def to_masked_dict(self, array, collapse=False):
        '''Returns a dict mapping roi names to masked numpy arrays'''
        return {name:self.by_name(name).to_masked(array, collapse) for name in self.names}

    def common_prop_names(self):
        """return an iterable of property names that are common to all rois"""
        names = []
        if len(self.rois) > 0:
            names = set(self.rois[0].props.keys())
        for roi in self.rois:
            names = names.intersection(set(roi.props.keys()))
        return names

    def disp(self):
        """Print a pretty display summarizing the ROISet"""

        def _prop_str(dic):
            """return a compact and pretty string representation of a dict"""
            keys = sorted(dic.keys())
            return ' '.join('{}:{!r}'.format(k, dic[k]) for k in keys)

        # Column name, value function pairs
        columns = [
            ('name',  lambda roi: roi.name),
            ('slice', lambda roi: roi.slc)]

        common_prop_names = self.common_prop_names()

        columns += [
            (name, lambda roi, name=name: roi.props[name])
            for name in common_prop_names]

        extra_props_filter = lambda props: {k:v for k,v in props.iteritems() if k not in common_prop_names}
        extra_props = {roi:_prop_str(extra_props_filter(roi.props)) for roi in self.rois}
        if any(extra_props.values()):
            columns += [('props', lambda roi: extra_props[roi])]

        if self.count > 0:
            t = PrettyTable([name for name, _ in columns])
            t.align = 'l'
            for roi in self.rois:
                t.add_row([func(roi) for _, func in columns])
            print(str(t) + '\n')
        else:
            print('ROISet is empty')

    @property
    def names(self):
        '''Returns a unique set of names from the rois in this object'''
        return unique(r.name for r in self.rois)

    @property
    def count(self):
        '''Returns the total number of ROIs in this object'''
        return len(self.rois)

    def __len__(self):
        return self.count

    def __iter__(self):
        '''Returns an iterator over the ROIs in this object'''
        return iter(self.rois)

class ROIFormatError(Exception):
    pass


def store(rois, filename):
    """Save ROIs to filename

    Parameters
    ----------
    rois : iterable
        iterable of ROIs to save
    filename : str
        name of file to save ROIs to
    """
    with h5py.File(filename, 'w') as f:
        f.attrs['version'] = _version
        f.attrs['description'] = _file_description
        f.attrs['creation_time'] = time.time()
        root = f.create_group('rois')
        for i, roi in enumerate(rois):
            roigrp = root.create_group('roi_%d' % i)
            roigrp.attrs['index'] = i
            roigrp.attrs['name'] = roi.name
            roigrp.create_dataset('mask',
                                  data=roi.mask,
                                  dtype=bool,
                                  compression=_compression_type,
                                  compression_opts=_compression_opts)
        log.debug('rois saved to:{!r} count:{!r} version:{!r} time:{!r}'
                .format(filename, len(rois), _version, f.attrs['creation_time']))


def load(filename, extra_props=None, shape=None):
    """Load ROIs from filename

    Parameters
    ----------
    filename : str
        name of file to load ROIs from
    shape : (default: None) tuple
        shape of ROI masks, used only for loading version 0 ROI files
    extra_props : dict
        add extra properties to the loaded ROI
    Returns
    -------
    List of ROIs loaded from file
    """
    extra_props = extra_props or {}
    rois = _load(filename, shape, extra_props)
    for roi in rois:
        props = extra_props.copy()
        props.update(dict(
            abspath=os.path.abspath(filename)
        ))
        roi.props = props
    return ROISet(rois)


def _load(filename, shape, extra_props):
    rois = {}
    with h5py.File(filename, 'r') as f:
        version = f.attrs.get('version')
        log.debug('loading ROIs from {!r}, version: {!r}'.format(filename, version))
        if version == 1:
            return _parse_version1(f)
        return _parse_version0(f, shape)


def _parse_version1(f):
    rois = {}
    creation_time = f.attrs.get('creation_time')
    for roigrp in f['/rois'].itervalues():
        index = roigrp.attrs['index']
        rois[index] = ROI(
                name=roigrp.attrs['name'],
                mask=roigrp['mask'].value)
    return [rois[i] for i in sorted(rois)]


def _convert_version0(roi_dict, shape):
    rois = []
    for name, _rois in roi_dict.items():
        if _rois:
            roi = ROI(name=name, mask=np.zeros(shape, dtype=bool))
            for _roi in _rois:
                mask = create_mask(shape, _roi['slc'], _roi['poly'])
                roi.mask = np.logical_or(roi.mask, mask)
            rois.append((_rois[0]['slc'], roi.name, roi))
    return [r[2] for r in sorted(rois, key=lambda r: (r[0], r[1]))]


def _parse_version0(f, shape):
    assert shape, 'shape must be set to parse version 0 ROI files'
    rois = defaultdict(list)
    for roigrp in f['/rois'].itervalues():
        viewdims = roigrp.attrs['viewdims']
        arrslc = roigrp.attrs['arrslc']
        roi = dict(
                name=roigrp.attrs['name'],
                poly=roigrp['poly'].value,
                slc=SliceTuple.from_arrayslice(arrslc, viewdims))
        rois[roi['name']].append(roi)
    return _convert_version0(rois, shape)


def read(paths, basepath=None, recursive=True):
    """Load all ROI files found in path
    If a directory is given, then ROIs loaded from different files
    are given a path property that is the relative path from the root path

    Given a directory structure as follows:
    >>> tree .
    ./rois/
    ├── rois0.h5
    ├── sub0
    │   ├── rois1.h5
    │   └── rois2.h5
    └── sub1
        ├── rois2.h5
        └── rois3.h5

    Reading this directory (i.e. read('rois')) will consume the six ROI files.
    """
    rois = []
    for path in filter(lambda p: os.path.splitext(p)[1] == ROI_FILE_EXT, path_generator(paths, recursive)):
        if basepath:
            props = { 'reldir': os.path.dirname(os.path.relpath(path, basepath)) }
        else:
            props = {}
        rois.extend(load(path, props))
    return ROISet(rois)


def save(rois):
    """Save an ROIset to the respective paths
    ROIs in the set must have a 'path' property otherwise they cannot be saved using this method, use store instead.
    """
    roi_files = defaultdict(list)
    for roi in rois:
        if 'abspath' not in roi.props:
            raise Exception('Unable to save rois, each roi must have an "abspath" property. Failing roi: {!r}'.format(roi))
        roi_files[roi.props['abspath']].append(roi)
    for path, rois in roi_files.iteritems():
        store(rois, path)
