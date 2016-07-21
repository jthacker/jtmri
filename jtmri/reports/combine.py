from .builder import Report, Table
from jtmri.fit import fit_r2star_with_threshold

from datetime import datetime


_adc_rows = (
    ('c', 'ADC (cortex) (10^-3 mm^2/s)', lambda x: '%1.2f' % (x / 1000.)),
    ('k', 'ADC (kidney) (10^-3 mm^2/s)', lambda x: '%1.2f' % (x / 1000.)),
)

_r2s_pre_rows = (
    ('c', 'Baseline (pre-furosemide) R2* (cortex) (s^-1)', lambda x: '%2.1f' % x),
    ('m', 'Baseline (pre-furosemide) R2* (medulla) (s^-1)', lambda x: '%2.1f' % x),
    ('k', 'Baseline (pre-furosemide) R2* (kidney) (s^-1)', lambda x: '%2.1f' % x),
)

_r2s_post_rows = (
    ('c', 'Post-furosemide R2* (cortex) (s^-1)', lambda x: '%2.1f' % x),
    ('m', 'Post-furosemide R2* (medulla) (s^-1)', lambda x: '%2.1f' % x),
    ('k', 'Post-furosemide R2* (kidney) (s^-1)', lambda x: '%2.1f' % x),
)


class CombineForm722Report(Report):
    '''Genearte a report that mimicks the COMBINE 722 form'''

    def __init__(self, dcms, observer):
        self._dcms = dcms
        self._observer = observer
        s = dcms.first
        title = 'ID: {}'.format(s.PatientID)
        description = \
            'Observer: {}\n' \
            'Patient Name: {}\n' \
            'Study Description: {}\n' \
            'Study Date: {}\n' \
            'Study Time: {}\n' \
            'Report Created On: {}\n' \
            .format(
                observer,
                s.PatientName,
                s.StudyDescription,
                s.StudyDate,
                s.StudyTime,
                datetime.now()
                )
        super(CombineForm722Report, self).__init__(title, description)
        headers = ['Description', 'Measurement']
        self._rtable = Table('Measurements of Right Kidney:', '', headers)
        self._ltable = Table('Measurements of Left Kidney:', '', headers)
        self.add_section(self._rtable)
        self.add_section(self._ltable)

    def set_series(self, series_adc, series_r2s_pre, series_r2s_post):
        def add_roi(table, rois, side, region, data, description):
            rois = rois.by_tag(self._observer)
            masked = rois.by_name(side+region).to_masked(data, collapse=True)
            table.add_row((description, fmt(masked.mean())))

        for side, table in [('r', self._rtable), ('l', self._ltable)]:
            for region, description, fmt in _adc_rows:
                dcms = self._dcms.by_series(series_adc)
                if dcms.count == 0:
                    table.add_row((description, ''))
                else:
                    data = dcms.data('SliceLocation')
                    add_roi(table, dcms.first.meta.roi, side, region, data, description)
        
            for series_number, rows in [(series_r2s_pre, _r2s_pre_rows),
                                        (series_r2s_post, _r2s_post_rows)]:
                dcms = self._dcms.by_series(series_number)
                for region, description, fmt in rows:
                    if dcms.count == 0:
                        table.add_row((description, ''))
                    else:
                        echo_times = dcms.all_unique.EchoTime / 1000.
                        data = dcms.data('SliceLocation')
                        r2star, _ = fit_r2star_with_threshold(echo_times, data)
                        add_roi(table, dcms.first.meta.roi, side, region, r2star, description)
