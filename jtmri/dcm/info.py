from collections import namedtuple
import os.path
import yaml

from ..utils import AttributeDict, flatten
from ..roi import load as load_roi 


_series_converters = {}
def _register(name, fn):
    _series_converters[name] = fn


class Study(object):
    def __init__(self, study, meta):
        self.study = study
        self.meta = meta

    def matches(self, dcm):
        return dcm.StudyInstanceUID == self.study

    def __call__(self, dcm):
        return AttributeDict(self.meta)


class MGRE(object):
    def __init__(self, study, series, orientation, stage):
        self.study = study
        self.series = series
        self.orientation = orientation
        self.stage = stage

    def matches(self, dcm):
        return dcm.StudyInstanceUID == self.study \
            and dcm.SeriesNumber == self.series

    def __call__(self, dcm):
        # If ucReconstructionMode == '0x8' then phase recon is enabled and there will be another
        # series following the gre images, which puts the R2* map one more down
        r2star_series_inc = 3 if dcm.Siemens.MrPhoenixProtocol['ucReconstructionMode'] == '0x8' else 2

        meta = AttributeDict({
                'orientation': self.orientation,
                'stage': self.stage,
                'r2star_series': self.series + r2star_series_inc})
        # TODO: Read all rois from here that match the series number
        # If there are directories then recurse down into them
        # Use the path as a tag for the rois
        roi_filename = os.path.join(os.path.dirname(dcm['filename']), 'rois', 'series_%2d.h5' % dcm.SeriesNumber)
        if os.path.exists(roi_filename):
            meta['rois'] = lambda roi_filename=roi_filename: load_roi(roi_filename)
        return meta

_register('mgre', MGRE)


def _convert(raw_infos):
    infos = []
    for info in raw_infos:
        infos.append(Study(info['study'], info.get('meta', AttributeDict({}))))
        for s in info['series']:
            for name,args in s.iteritems():
                if name in _series_converters:
                    infos.append(_series_converters[name](info['study'], *args))
    return infos


def _path_gen(dcms):
    infopaths = set()
    for dcm in dcms:
        dirname = os.path.dirname(dcm['filename'])
        infopaths.add(os.path.join(dirname, 'info.yaml'))
    return (path for path in infopaths if os.path.exists(path))


def load_infos(dcms):
    infos = _convert(flatten(yaml.load_all(open(p)) for p in _path_gen(dcms)))
    for d in dcms:
        for info in infos:
            if info.matches(d):
                d['meta'] = d.get('meta', AttributeDict({})).update(info(d))
    return dcms

