import os

import jtmri.dcm

empty_dirname = os.path.join(os.path.dirname(__file__), 'data/empty')

def test_read():
    dcms = jtmri.dcm.read(empty_dirname)
    assert len(dcms) == 0
