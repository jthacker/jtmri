import skimage.draw
import numpy as np
import h5py
from collections import namedtuple

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


def create_mask(shape, slc, poly, collapse=False):
    '''Convert the polygons to a binary mask according the specified array shape
    Args:
    shape -- a tuple with the size of each dimension in the mask
    slc   -- SliceTuple describing where to apply the polygon to
    poly  -- A numpy array of (x,y) point pairs describing a polygon
    
    Returns:
    binary mask with the region in poly set to False and everywhere else
    set to True
    '''
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
    def __init__(self, name, poly, slc):
        self.name = name
        self.poly = poly
        self.slc = slc

    def to_mask(self, shape, collapse=False):
        return create_mask(shape, self.slc, self.poly, collapse)


class ROISet(object):
    def __init__(self, rois):
        self.rois = rois

    def by_name(self, *names):
        return ROISet(filter(lambda r: r.name in names, self.rois))

    def to_mask(self, shape, collapse=False):
        masks = map(lambda a: a.to_mask(shape, collapse), self)
        return reduce(np.logical_or, masks) 

    @property
    def names(self):
        return set(r.name for r in self.rois)

    @property
    def count(self):
        return len(self.rois)

    def __len__(self):
        return self.count

    def __iter__(self):
        return iter(self.rois)


def load(filename):
    rois = []
    with h5py.File(filename, 'r') as f:
        for roigrp in f['/rois'].itervalues():
            viewdims = roigrp.attrs['viewdims']
            arrslc = roigrp.attrs['arrslc']
            rois.append(
                ROI(name=roigrp.attrs['name'],
                    poly=roigrp['poly'].value,
                    slc=SliceTuple.from_arrayslice(arrslc, viewdims)))
    return ROISet(rois)


def _filter_viewdims(slc):
    viewdims = slc.viewdims
    return [0 if d in viewdims else v for d,v in enumerate(slc)] 


def save(rois, filename):
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
