from .builder import Images, Report, Section, Table
from ..np import flatten_axes, iter_axis
from ..fit import fit_r2star_with_threshold
import numpy as np


class ROITable(Table):
    '''Build a table of ROI values'''
    def __init__(self, title, description='Table of ROIs'):
        headers = ['Series', 'Observer', 'Name', 'Mean', 'Std', 'Size']
        super(ROITable, self).__init__(title, description, headers)

    def add_ndarray(self, name, data, rois):
        '''For each uniquely named roi in roi_set,
        create a new row in the table with columns:
            name, roi_name, mean, std, size
        Args:
            name     -- Name for the rois to be added (e.g. 'series 15')
            data     -- ndarray
            roi_dict -- Dictionary of ROIs
        '''
        def fmt(val):
            return 'nan' if np.ma.is_masked(val) else '{:4.2f}'.format(val)

        for (observer, roi_name), roiset in rois.groupby(('tag', 'name')).iteritems():
            masked = roiset.to_masked(data, collapse=True)
            row = (name,
                   observer,
                   roi_name, 
                   fmt(masked.mean()),
                   fmt(masked.std()),
                   (~masked.mask).sum())
            self.add_row(row)

    def add_series(self, name, series):
        '''For each uniquely named roi in roi_set,
        create a new row in the table with columns:
            name, roi_name, mean, std, size
        Args:
            name    -- Name for the rois to be added (e.g. 'series 15')
            series  -- ndarray or DicomSet
        '''
        rois = series.first.meta.roi
        data = series.data(['SliceLocation'])
        self.add_ndarray(name, data, rois)


class DicomStudySummaryReport(Report):
    '''Genearte a summary report of a dicom study'''

    def __init__(self, dcms):
        self._dcms = dcms
        s = dcms.first
        title = 'ID: {}'.format(s.PatientID)
        description = \
            'PatientName: {}\n' \
            'Study Description: {}\n' \
            'Study Date: {}\n' \
            'Study Time: {}\n'.format(
                s.PatientName,
                s.StudyDescription,
                s.StudyDate,
                s.StudyTime)
        super(DicomStudySummaryReport, self).__init__(title, description)
        self._roi_table = ROITable('ROI Summary')
        self.add_section(self._roi_table)

    def add_images(self, title, description, data):
        '''Add an image for every image in the axes after the first two.
        For example, an ndarray with dims (64,64,5,4) would add 20 images,
        5 images for the third dimension times 4 images for the fourth dimension.
        '''
        images = Images(title, description, [data.min(), data.max()])
        arrs = flatten_axes(data, range(2, data.ndim))
        for arr in iter_axis(arrs, 2):
            images.add_image(arr)
        self.add_section(images)

    def add_series(self, series_number, title=None, description=None):
        '''Add images and ROIs from the specified series number'''
        dcms = self._dcms.by_series(series_number)
        s = dcms.first
        if title is None:
            title = 'Series %d' % s.SeriesNumber
        if description is None:
            description = s.SeriesDescription
        data = dcms.data(['SliceLocation'])
        self.add_images(title, description, data)
        self._roi_table.add_series(title, dcms)

    def add_series_r2star(self, series_number, title=None, description=None):
        '''Generate an R2* map from series number and add the images and
        ROI values to the report
        '''
        dcms = self._dcms.by_series(series_number)
        echo_times = dcms.all_unique.EchoTime / 1000.
        s = dcms.first
        if title is None:
            title = 'Series %d R2*' % s.SeriesNumber
        if description is None:
            description = 'R2* fit. TE: ' + str(echo_times)
        data = dcms.data(['SliceLocation'])
        r2star = fit_r2star_with_threshold(echo_times, data)
        self.add_images(title, description, r2star)
        self._roi_table.add_ndarray(title, r2star, s.meta.roi)
