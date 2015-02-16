from glob import iglob
import os, os.path
import yaml, copy, collections

from ..utils import AttributeDict, flatten, Lazy
from ..roi import load as load_roi 


_sequences = {}
def _register_seq(name, seqclass):
    _sequences[name] = seqclass


def _update_meta(meta_new, meta_old):
    '''Updates keys in meta_old with values in meta_new.
    If the key in meta_old is a list, then the values in
    meta_old for that key are appended to meta_old[key].
    '''
    for key,val_new in meta_new.items():
        if key in meta_old:
            val_old = meta_old[key]
            if isinstance(val_old, list):
                assert isinstance(val_new, list)
                val_old.extend(val_new)
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
        self._add_metadata_rois(meta, dcm)

    def _add_metadata_seq(self, meta, dcm, study_dcms):
        if self.sequence is not None:
            key = dcm.SeriesInstanceUID
            if key in self.seq_meta_cache:
                seq_meta = self.seq_meta_cache[key]
            else:
                seq_meta = self.sequence.create_metadata(dcm, study_dcms)
                self.seq_meta_cache[key] = seq_meta
            _update_meta(seq_meta, meta)

    def _add_metadata_rois(self, meta, dcm):
        # TODO: Read all rois from here that match the series number
        # If there are directories then recurse down into them
        # Use the path as a tag for the rois
        rois = {}
        roi_files = {}
        rois_dir = os.path.join(os.path.dirname(dcm['filename']), 'rois')
        for dirpath, dirnames, filenames in os.walk(rois_dir):
            for filename in filenames:
                if filename == 'series_%02d.h5' % dcm.SeriesNumber:
                    roi_file = os.path.join(dirpath, filename)
                    tag = os.path.relpath(dirpath, rois_dir)
                    tag = '/' if tag == '.' else '/' + tag
                    rois[tag] = Lazy(lambda: load_roi(roi_file))
                    roi_files[tag] = roi_file
        # ROI is set with an AttributeDict because of they are lazily loaded
        meta['roi'] = AttributeDict(rois)
        meta['roi_filename'] = roi_files


class GRE(object):
    @staticmethod
    def create_metadata(dcm, study_dcms):
        meta = AttributeDict({})
        meta.r2star_series = None
        meta.t2star_series = None
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


def update_metadata(study_infos, dcms):
    for dcm in dcms:
        dcm.meta = AttributeDict({})
    for study in dcms.studies():
        for info in study_infos:
            if info.matches(study.first):
                info.update_metadata(study)
