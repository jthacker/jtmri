from collections import Sequence
from glob import iglob
import os, os.path
import yaml

from ..utils import AttributeDict, flatten
from ..roi import load as load_roi 


_sequences = {}
def _register_seq(name, seqclass):
    _sequences[name] = seqclass


class Study(object):
    def __init__(self, info):
        self.study = info.get('study', None)
        self.meta = info.get('meta', {})
        self.series = []

    def matches(self, dcm):
        return dcm.StudyInstanceUID == self.study

    def add_metadata(self, meta, dcm, study_dcms):
        meta.update(self.meta) 

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
        if isinstance(sid, Sequence):
            self.matches = lambda dcm, sid=sid: dcm.SeriesNumber in sid
        else:
            self.matches = lambda dcm, sid=sid: dcm.SeriesNumber == sid
        self.sequence = _sequences.get(seriesinfo['seq'], None)
        self.meta = seriesinfo.get('meta', {})
   
    def add_metadata(self, meta, dcm, study_dcms):
        meta.update(self.meta)
        self._add_metadata_seq(meta, dcm, study_dcms) 
        self._add_metadata_rois(meta, dcm)

    def _add_metadata_seq(self, meta, dcm, study_dcms):
        if self.sequence is not None:
            self.sequence.add_metadata(meta, dcm, study_dcms)

    def _add_metadata_rois(self, meta, dcm):
        # TODO: Read all rois from here that match the series number
        # If there are directories then recurse down into them
        # Use the path as a tag for the rois
        roi_filename = os.path.join(os.path.dirname(dcm['filename']), 'rois', 'series_%2d.h5' % dcm.SeriesNumber)
        if os.path.exists(roi_filename):
            meta['rois'] = lambda roi_filename=roi_filename: load_roi(roi_filename)


class GRE(object):
    @staticmethod
    def add_metadata(meta, dcm, study_dcms):
        # If ucReconstructionMode == '0x8' then phase recon is enabled and there will be another
        # series following the gre images, which puts the R2* map one more down
        phase_saved = dcm.Siemens.MrPhoenixProtocol['ucReconstructionMode'] == '0x8'
        t2star_series_num = dcm.SeriesNumber + (2 if phase_saved else 1)
        r2star_series_num = dcm.SeriesNumber + (3 if phase_saved else 2)
        
        meta.sequence = 'gre'
        meta.t2star = study_dcms.by_series(t2star_series_num)
        r2star = study_dcms.by_series(r2star_series_num)
        if len(r2star) > 0 and 'ImageComments' in r2star.first and r2star.first.ImageComments == 'r2star image':
            meta.r2star = r2star
        else:
            meta.r2star = None

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
    for study_dcms in dcms.studies():
        for info in study_infos:
            if info.matches(study_dcms.first):
                info.update_metadata(study_dcms)
