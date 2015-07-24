import unittest

import numpy as np
import jtmri.roi

class TestROI(unittest.TestCase):
    def setUp(self):
        pass

    def test_roi_to_masked(self):
        rois = jtmri.roi.ROISet([])
        img = np.ones((10, 10))
        masked = rois.to_masked(img)
        self.assertTrue(masked.mask.all())
