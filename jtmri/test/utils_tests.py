from ..utils import as_iterable

def test_asiterable():
    assert 'x' == as_iterable('x')
    assert [1] == list(as_iterable(1))
    assert [1,2,3] == list(as_iterable([1,2,3]))
