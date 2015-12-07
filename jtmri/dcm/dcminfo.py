import collections
import copy
from glob import iglob
import os
import os.path
import re
import yaml

from ..utils import AttributeDict, flatten, Lazy
from ..cache import memoize
from ..roi import read as read_roi
from ..fit import fit_r2star_with_threshold
from .. import asl


_sequences = {}
def _register_seq(name, seqclass):
    _sequences[name] = seqclass


def _update_meta(meta_new, meta_old):
    '''Updates keys in meta_old with values in meta_new.
    If the key in meta_old is a list, then the values in
    meta_old for that key are appended to meta_old[key].
    '''
    for key, val_new in meta_new.items():
        if key in meta_old:
            val_old = meta_old[key]
            if isinstance(val_old, list):
                assert isinstance(val_new, list)
                val_old.extend(val_new)
            else:
                meta_old[key] = copy.copy(val_new)
        else:
            meta_old[key] = copy.copy(val_new)


class Study(object):
    def __init__(self, info):
        self.study = info.get('study', None)
        self.meta = info.get('meta', {})
        self.series = []

    def matches(self, dcm):
        return dcm.StudyInstanceUID == self.study

    def add_metadata(self, meta, dcm, study_dcms):
        '''Add the metadata in this object to meta.
        If the old value is a list then the new value is appended to it,
        rather than just replace the value as would be the case
        for atomic values.
        '''
        _update_meta(self.meta, meta)

    def update_metadata(self, study_dcms):
        for dcm in study_dcms:
            meta = dcm.get('meta', AttributeDict({}))
            self.add_metadata(meta, dcm, study_dcms)
            for series in self.series:
                if series.matches(dcm):
                    series.add_metadata(meta, dcm, study_dcms)
            dcm['meta'] = meta


class Series(object):
    def __init__(self, seriesinfo):
        sid = seriesinfo['id']
        if isinstance(sid, collections.Sequence):
            self.matches = lambda dcm, sid=sid: dcm.SeriesNumber in sid
        else:
            self.matches = lambda dcm, sid=sid: dcm.SeriesNumber == sid
        self.sequence = _sequences.get(seriesinfo['seq'], None)
        self.meta = seriesinfo.get('meta', {})
        self.seq_meta_cache = {}

    def add_metadata(self, meta, dcm, study_dcms):
        _update_meta(self.meta, meta)
        self._add_metadata_seq(meta, dcm, study_dcms) 

    def _add_metadata_seq(self, meta, dcm, study_dcms):
        if self.sequence is not None:
            key = dcm.SeriesInstanceUID
            if key in self.seq_meta_cache:
                seq_meta = self.seq_meta_cache[key]
            else:
                seq_meta = self.sequence.create_metadata(dcm, study_dcms, meta)
                self.seq_meta_cache[key] = seq_meta
            _update_meta(seq_meta, meta)


class GRE(object):
    @staticmethod
    def create_metadata(dcm, study_dcms, old_meta):
        meta = AttributeDict({})
        meta.r2star_series = None
        meta.t2star_series = None
        def fit(dcm=dcm, study_dcms=study_dcms):
            series = study_dcms.by_series(dcm.SeriesNumber)
            data = series.data('SliceLocation')
            echo_times = series.all_unique.EchoTime / 1000.
            return fit_r2star_with_threshold(echo_times, data)
        meta.r2star_offline = fit

        meta.sequence = 'gre'
        if dcm.SoftwareVersions == 'syngo MR D13':
            phase_saved = dcm.Siemens.MrPhoenixProtocol['ucReconstructionMode'] == '8'
            for num in [1,2]:
                num = dcm.SeriesNumber + (num + 1 if phase_saved else num)
                possible_map = study_dcms.by_series(num)
                if possible_map.count > 0:
                    if possible_map.first.SeriesDescription == 'R2Star_Images':
                        meta.r2star_series = num
                    if possible_map.first.SeriesDescription == 'T2Star_Images':
                        meta.t2star_series = num
        else:
            # If ucReconstructionMode == '0x8' then phase recon is enabled and there will be another
            # series following the gre images, which puts the R2* map one more down
            phase_saved = dcm.Siemens.MrPhoenixProtocol['ucReconstructionMode'] == '0x8'
            t2star_series_num = dcm.SeriesNumber + (2 if phase_saved else 1)
            r2star_series_num = dcm.SeriesNumber + (3 if phase_saved else 2)
            
            meta.t2star_series = t2star_series_num
            r2star = study_dcms.by_series(r2star_series_num)
            if len(r2star) > 0 and 'ImageComments' in r2star.first and r2star.first.ImageComments == 'r2star image':
                meta.r2star_series = r2star_series_num
        return meta
_register_seq('gre', GRE)


class ADC(object):
    @staticmethod
    def create_metadata(dcm, study_dcms, old_meta):
        meta = AttributeDict({})
        meta.sequence = 'adc'
        return meta
_register_seq('adc', ADC)


class SE(object):
    @staticmethod
    def create_metadata(dcm, study_dcms, old_meta):
        meta = AttributeDict({})
        meta.sequence = 'se'
        def fit(dcm=dcm, study_dcms=study_dcms):
            series = study_dcms.by_series(dcm.SeriesNumber) 
            data = series.data('SliceLocation')
            echo_times = series.all_unique.EchoTime / 1000.
            return fit_r2star_with_threshold(echo_times, data)
        meta.r2_offline = fit
        return meta
_register_seq('se', SE)


class ASL(object):
    @staticmethod
    def create_metadata(dcm, study_dcms, old_meta):
        meta = AttributeDict({})
        meta.sequence = 'asl'
        meta.inv_delay = int(dcm.Siemens.MrPhoenixProtocol['sWipMemBlock']['alFree']['1']) / 1000.
        meta.m0 = study_dcms.by_series(old_meta.m0)
        meta.pwi = lambda study_dcms=study_dcms, asl=asl, series_num=dcm.SeriesNumber: \
                asl.pwi(study_dcms.by_series(series_num).data('SliceLocation'))
        def fit_rbf(subject, m0=None, dcm=dcm, study_dcms=study_dcms, meta=meta):
            """Find the renal blood flow map
            Args:
                subject -- either 'human' or 'rat'
            """
            subject = subject.lower()
            assert subject in asl.rbf_params.keys()
            if m0 is None:
                m0 = meta.m0.data('SliceLocation').mean(axis=-1)
            return asl.rbf(meta.pwi(), m0, meta.inv_delay, asl.rbf_params[subject])
        meta.rbf_offline = fit_rbf
        return meta
_register_seq('asl', ASL)


class ASLNAV(object):
    """ASLNAV is Huan's sequence that must be reconstructed from raw data in Matlab.
    Use this to label the dicom converted rbf.mat.
    """
    @staticmethod
    def create_metadata(dcm, study_dcms, old_meta):
        meta = AttributeDict({})
        meta.sequence = 'asl-nav'
        # TODO: Some CKD-LONG scans do not seem to have this parameter (e.g. RH217, VW405)
        #meta.inv_delay = int(dcm.Siemens.MrPhoenixProtocol['sWipMemBlock']['alFree']['1']) / 1000.
        return meta
_register_seq('asl-nav', ASLNAV) 


def _convert(raw_infos):
    '''Convert a raw info (yaml file) to a function that combines information
    from a dcm and the raw info'''
    infos = []
    for info in raw_infos:
        study = Study(info)
        if 'series' in info:
            study.series.extend(Series(series_info) for series_info in info['series'])
        infos.append(study)
    return infos


def _dirs(path, recursive):
    if os.path.isdir(path):
        path = os.path.join(path, '*')

    dirs = set()
    for p in iglob(path):
        if os.path.isdir(p) and recursive:
            for rootdir,_,_ in os.walk(p):
                dirs.add(rootdir)
        dirs.add(os.path.dirname(p))
    return dirs


def read(path, recursive):
    infos = []
    for dirpath in _dirs(path, recursive):
        path = os.path.join(dirpath, 'info.yaml')
        if os.path.exists(path):
            infos.extend(_convert(yaml.load_all(open(path))))
    return infos


class ROIReader(object):
    """Lazily load rois
    The call method on this class is used to return Lazy objects that will
    load ROIs from disk. Lazy objects need to be placed in attributes in the AttributeDict
    in order to be auto loaded when requested.

    Call is memoized as well so it will return the same object when called with the 
    same arguments
    """
    def __init__(self):
        self._paths = {}
        self._pattern = re.compile(r'series_([0-9]+)\.h5')

    def _get_paths(self, rois_dir):
        if rois_dir in self._paths:
            return self._paths[rois_dir]

        series_files = collections.defaultdict(list)
        for dirpath, dirnames, filenames in os.walk(rois_dir):
            for filename in filenames:
                m = self._pattern.match(filename)
                if m:
                    series_number = int(m.group(1))
                    path = os.path.abspath(os.path.join(dirpath, filename))
                    series_files[series_number].append(path)
        self._paths[rois_dir] = series_files
        return series_files
    
    @memoize
    def __call__(self, roidir, series_number):
        """Find all rois for the given series_number under the rois directory
        """
        roidir = os.path.abspath(roidir)
        # TODO: This method should be converted to using the new ROI props method
        paths = self._get_paths(roidir).get(series_number)
        if paths:
            return Lazy(lambda paths=paths, roidir=roidir: read_roi(paths, roidir))
        return None


def update_metadata_rois(dcms):
    """Reread rois from disk
    Looks for ROIs in a directory called 'roi' in the same directory as the dicom file.
    ROIs are lazily loaded so this should have minimal impact on performance.
    """
    roi_reader = ROIReader()
    for dcm in dcms:
        rois_dir = os.path.join(os.path.dirname(dcm['filename']), 'rois')
        rois = roi_reader(rois_dir, dcm.SeriesNumber)
        if rois:
            dcm.meta['roi'] = rois


def update_metadata(study_infos, dcms):
    for dcm in dcms:
        dcm.meta = AttributeDict({'sequence': None})
    update_metadata_rois(dcms)
    for study in dcms.studies():
        for info in study_infos:
            if info.matches(study.first):
                info.update_metadata(study)
