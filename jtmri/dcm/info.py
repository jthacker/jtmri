from collections import namedtuple
import os.path
import yaml

from ..utils import AttributeDict, flatten


_series_converters = {}
def _register(name, fn):
    _series_converters[name] = fn


class mgre(object):
    def __init__(self, study, series, orientation, stage):
        self.study = study
        self.series = series
        self.orientation = orientation
        self.stage = stage

    def matches(self, dcm):
        return dcm.StudyInstanceUID == self.study \
            and dcm.SeriesNumber == self.series

    def apply_to(self, dcm):
        dcm['meta'] = AttributeDict({
                'orientation': self.orientation,
                'stage': self.stage,
                'r2star_series': self.series + 2})
_register('mgre', mgre)


def _convert(raw_infos):
    infos = []
    for info in raw_infos:
        for s in info['series']:
            for name,args in s.iteritems():
                if name in _series_converters:
                    infos.append(_series_converters[name](info['study'], *args))
    return infos

def path_gen(path, recursive):
    if recursive and os.path.isdir(path):
        for root,_,files in os.walk(path):
            for f in files:
                if os.path.basename(f) == 'info.yaml':
                    yield os.path.join(root,f)
    else:
        info = os.path.join(os.path.dirname(path), 'info.yaml')
        if os.path.exists(info):
            yield info


def load_infos(dcms, path, recursive):
    infos = _convert(flatten(yaml.load_all(open(p)) for p in path_gen(path, recursive)))
    for d in dcms:
        for info in infos:
            if info.matches(d):
                info.apply_to(d)
    return dcms

