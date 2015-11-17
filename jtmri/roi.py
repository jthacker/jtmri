# -*- coding: utf-8 -*-
from collections import namedtuple, defaultdict
import h5py
import numpy as np
import os
from prettytable import PrettyTable
import skimage.draw

from .utils import unique, rep, path_generator, groupby, AttributeDict

ROI_FILE_EXT = '.h5'


#TODO: Duplicate code from arrview

class SliceTuple(tuple):
    def __init__(self, *args, **kwargs):
        super(SliceTuple, self).__init__(*args, **kwargs)
        self._xdim = self.index('x')
        self._ydim = self.index('y')

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim
   
    @property
    def viewdims(self):
        return (self.xdim, self.ydim)

    @property
    def is_transposed(self):
        return self.ydim > self.xdim

    def viewarray(self, arr):
        '''Transforms arr from Array coordinates to Screen coordinates
        using the transformation described by this object'''
        assert arr.ndim == len(self), 'dimensions of arr must equal the length of this object'
        viewdims = self.viewdims
        arrayslice = [slice(None) if d in viewdims else x for d,x in enumerate(self)]
        a = arr[arrayslice]
        return a.transpose() if self.is_transposed else a

    def screen_coords_to_array_coords(self, x, y):
        '''Transforms arr of Screen coordinates to Array indicies'''
        r,c = (y,x) if self.is_transposed else (x,y)
        return r,c

    def slice_from_screen_coords(self, x, y, arr):
        slc = list(self)
        rdim,cdim = self.screen_coords_to_array_coords(*self.viewdims)
        slc[rdim],slc[cdim] = self.screen_coords_to_array_coords(x,y)
        return slc

    #TODO: deprecate
    def is_transposed_view_of(self, slc):
        '''Test if slc is equal to this object but with swapped x and y dims swapped
        Args:
        slc -- (SliceTuple)

        Returns:
        True if slc equals this object with swapped view dimensions, otherwise False.

        Examples:
        >>> a = SliceTuple(('x','y',0,1))
        >>> b = SliceTuple(('y','x',0,1))
        >>> c = SliceTuple(('y','x',0,20))
        >>> a.is_transposed_view_of(b)
        True
        >>> b.is_transposed_view_of(a)
        True
        >>> a.is_transposed_view_of(c)
        False
        '''
        s = list(self)
        # Swap the axes
        s[self.xdim],s[self.ydim] = s[self.ydim],s[self.xdim]
        return s == list(slc)

    @property
    def freedims(self):
        return tuple(i for i,x in enumerate(self) if i not in self.viewdims)

    #TODO: @deprecated: Get rid of this method
    @staticmethod
    def from_arrayslice(arrslice, viewdims):
        '''Replace the dims from viedims in arrslice.
        Args:
        arrslice -- a tuple used for slicing a numpy array. The method arrayslice
                    returns examples of this type of array
        viewdims -- a len 2 tuple with the first position holding the dimension
                    number that corresponds to the x dimension and the second is
                    the y dimension.
        Returns:
        arrslice with each dim in viewdims replaced by 'x' or 'y'

        For example:
        >>> arrslice = (0,0,0,0)
        >>> viewdims = (1,0)
        >>> from_arrayslice(arrslice, viewdims)
        ('y','x',0,0)
        '''
        slc = list(arrslice)
        xdim,ydim = viewdims
        slc[xdim],slc[ydim] = 'x','y'
        return SliceTuple(slc)

    def __repr__(self):
        return 'SliceTuple({})'.format(', '.join(map(repr, self)))


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
    def __init__(self, name, poly, slc, props=None):
        self.name = name
        self.poly = poly
        self.slc = slc
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
        return create_mask(shape, self.slc, self.poly, collapse)
    
    def __repr__(self):
        return '<ROI name:{!r} slc:{!r} props:{!r} ...>'.format(self.name, self.slc, self.props)



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


def load(filename, extra_props=None):
    """Load a single ROI file"""
    extra_props = extra_props or {}
    rois = []
    with h5py.File(filename, 'r') as f:
        for roigrp in f['/rois'].itervalues():
            viewdims = roigrp.attrs['viewdims']
            arrslc = roigrp.attrs['arrslc']
            props = extra_props.copy()
            props.update(dict(
                abspath=os.path.abspath(filename)
            ))
            rois.append(
                ROI(name=roigrp.attrs['name'],
                    poly=roigrp['poly'].value,
                    slc=SliceTuple.from_arrayslice(arrslc, viewdims),
                    props=props))
    return ROISet(rois)


def store(rois, filename):
    """ Save ROIs to the file specified by filename
    Args:
      rois     -- should be an iterable of rois
      filename -- string of the name of the file to write too

    Returns: None
    """
    def _filter_viewdims(slc):
        viewdims = slc.viewdims
        return [0 if d in viewdims else v for d,v in enumerate(slc)] 

    with h5py.File(filename, 'w') as f:
        root = f.create_group('rois')
        for i,roi in enumerate(rois):
            roigrp = root.create_group('%d' % i)
            # h5py only support utf8 strings at the moment, need to coerce data to
            # this representation
            roigrp.attrs['name'] = roi.name
            roigrp.attrs['viewdims'] = roi.slc.viewdims
            roigrp.attrs['arrslc'] = _filter_viewdims(roi.slc)
            roigrp['poly'] = roi.poly


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
