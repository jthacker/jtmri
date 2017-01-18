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

    @property
    def view_slice(self):
        """Returns a tuple that can be used to get the slice of the array"""
        return [slice(None) if d in self.viewdims else x for d, x in enumerate(self)]

    def viewarray(self, arr):
        '''Transforms arr from Array coordinates to Screen coordinates
        using the transformation described by this object'''
        assert arr.ndim == len(self), 'dimensions of arr must equal the length of this object'
        a = arr[self.view_slice]
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
