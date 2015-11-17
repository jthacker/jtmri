from collections import namedtuple
import unittest

from ..utils import as_iterable, Lazy, AttributeDict, flatten, getattr_nested


class TestUtils(unittest.TestCase):
    def test_asiterable(self):
        assert ['x'] == as_iterable('x')
        assert [1] == list(as_iterable(1))
        assert [1,2,3] == list(as_iterable([1,2,3]))

    def test_attributedict(self):
        a = AttributeDict({'a': 1, 'b': 2})
        assert a.a == 1
        assert a.b == 2
        assert a['a'] == 1
        assert a['b'] == 2
        assert sorted(a.keys()) == ['a', 'b']
        assert a.get('c', 3) == 3


    def test_attributedict_lazy(self):
        '''A Lazy field should not be loaded, until it is requested'''
        x = []
        def lazy_field():
            x.append(1)
            return 1
        a = AttributeDict({'a': 1, 'b': Lazy(lazy_field)})
        assert x == []
        assert a.b == 1
        assert x == [1]

        # calling the same field again should not cause the lazy field to loaded again
        assert a.b == 1
        assert x == [1]

        # Calling .values() should cause lazy fields to load too
        x = []
        a = AttributeDict({'a': 1, 'b': Lazy(lazy_field)})
        a.values()
        assert x == [1]

        x = []
        a = AttributeDict({'a': 1, 'b': Lazy(lazy_field)})
        items = sorted(list(a.iteritems()), key=lambda v: v[0])
        assert x == [1]
        assert len(items) == 2
        assert items[0] == ('a', 1)
        assert items[1] == ('b', 1)


    def test_attributedict_pickleable(self):
        import pickle
        a = AttributeDict({'a': 1, 'b': 2})
        s = pickle.dumps(a)
        b = pickle.loads(s)
        assert a == b


    def test_flatten(self):
        self.assertEqual([1,2,3,4,5], list(flatten([1, [2, 3], [4], 5])))


    def test_getattr_nested(self):
        P = namedtuple('P', 'x,y')
        p = P(P(1, 2), P(3, P(4, 5)))
        self.assertEqual(getattr_nested(p, 'x'), P(1, 2))
        self.assertEqual(getattr_nested(p, 'x.x'), 1)
        self.assertEqual(getattr_nested(p, 'y.x'), 3)
        self.assertEqual(getattr_nested(p, 'y.y.y'), 5)

