import jtmri.cache
from jtmri.cache import func_hash
import unittest

class TestFuncHash(unittest.TestCase):
    def setUp(self):
        jtmri.cache._cache = {}

    def test_different_names(self):
        def test1(a,b,c): return a*b*c
        def test2(a,b,c): return a*b*c
        args = (1,2,3)
        self.assertEqual(
                func_hash(test1, args, {}),
                func_hash(test2, args, {}))
    
    def test_default_args(self):
        '''Since the actual keyword args that get passed to
        the function will be passed to func_hash, these two
        functions should hash to the samething for the same
        args and kwargs.
        '''
        def test1(a,b,c): return a*b*c
        def test2(a,b,c=1): return a*b*c
        args = (1,2,3)
        self.assertEqual(
                func_hash(test1, args, {}),
                func_hash(test2, args, {}))

    def test_different_args(self):
        def test(a,b,c): return a*b*c
        self.assertNotEqual(
                func_hash(test, (1,2,3), {}),
                func_hash(test, (3,2,1), {}))
