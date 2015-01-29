from ..utils import as_iterable, Lazy, AttributeDict


def test_asiterable():
    assert 'x' == as_iterable('x')
    assert [1] == list(as_iterable(1))
    assert [1,2,3] == list(as_iterable([1,2,3]))


def test_attributedict():
    a = AttributeDict({'a': 1, 'b': 2})
    assert a.a == 1
    assert a.b == 2
    assert a['a'] == 1
    assert a['b'] == 2
    assert sorted(a.keys()) == ['a', 'b']
    assert a.get('c', 3) == 3


def test_attributedict_lazy():
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


def test_attributedict_pickleabl():
    import pickle
    a = AttributeDict({'a': 1, 'b': 2})
    s = pickle.dumps(a)
    b = pickle.loads(s)
    assert a == b
